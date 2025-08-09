#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <iostream>
#include <glm/glm.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <mynydd/mynydd.hpp>

struct RadixParams {
    uint32_t bitOffset;
    uint32_t numBins;
    uint32_t totalSize;
    uint32_t itemsPerGroup;
};

TEST_CASE("Radix histogram compute shader correctly generates bin counts", "[sort]") {
    const size_t n = 256 * 4;
    const uint32_t numBins = 256;
    const uint32_t itemsPerGroup = 256;
    const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(uint32_t), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), true);
    auto uniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(RadixParams), true);

    auto pipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram.comp.spv", 
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{input, output, uniform}
    );

    std::vector<uint32_t> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = i % numBins;
    }

    RadixParams params = {
        .bitOffset = 0,
        .numBins = numBins,
        .totalSize = static_cast<uint32_t>(n),
        .itemsPerGroup = itemsPerGroup
    };

    mynydd::uploadUniformData<RadixParams>(contextPtr, params, uniform);
    mynydd::uploadData<uint32_t>(contextPtr, inputData, input);

    mynydd::executeBatch<float>(contextPtr, {pipeline}, groupCount);

    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, output, groupCount * numBins);

    REQUIRE(groupCount == 4); // since 256 / 64 = 4

    // Combine all workgroup histograms
    std::vector<uint32_t> combinedHistogram(numBins, 0);
    for (uint32_t group = 0; group < groupCount; ++group) {
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            combinedHistogram[bin] += out[group * numBins + bin];
        }
    }

    // Validate combined bin counts
    for (uint32_t bin = 0; bin < numBins; ++bin) {
        REQUIRE(combinedHistogram[bin] == Catch::Approx(n / numBins));
    }
}


TEST_CASE("Histogram summation shader correctly sums partial histograms", "[sort]") {
    const uint32_t numBins = 16;
    const uint32_t groupCount = 2;  // two partial histograms

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    // Predefined partial histograms (groupCount × numBins)
    std::vector<uint32_t> partialHistograms = {
        // group 0
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        // group 1
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };

    // Expected summed histogram (element-wise sum)
    std::vector<uint32_t> expectedHistogram(numBins);
    for (uint32_t i = 0; i < numBins; ++i) {
        expectedHistogram[i] = partialHistograms[i] + partialHistograms[numBins + i];
    }

    // Create buffers
    auto inputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, partialHistograms.size() * sizeof(uint32_t), false);
    auto outputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, numBins * sizeof(uint32_t), true);

    // Uniform struct matching shader uniform block
    struct SumParams {
        uint32_t groupCount;
        uint32_t numBins;
    } sumParams{groupCount, numBins};

    auto uniformBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, sizeof(SumParams), true);

    // Upload partial histograms and uniform params
    mynydd::uploadData<uint32_t>(contextPtr, partialHistograms, inputBuffer);
    mynydd::uploadUniformData<SumParams>(contextPtr, sumParams, uniformBuffer);

    // Load the summation shader (compiled SPIR-V must match the shader code given)
    auto pipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{inputBuffer, outputBuffer, uniformBuffer}
    );

    // Dispatch exactly 1 workgroup with 256 threads
    mynydd::executeBatch<float>(contextPtr, {pipeline}, 1);

    // Fetch the summed histogram result
    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, outputBuffer, numBins);

    // Validate output matches expected sums
    for (uint32_t i = 0; i < numBins; ++i) {
        REQUIRE(out[i] == expectedHistogram[i]);
    }
}

TEST_CASE("Radix histogram + sum shaders chained produce correct combined arbitrary histogram", "[sort]") {
    const size_t n = 1024;
    const uint32_t numBins = 16;
    const uint32_t itemsPerGroup = 256;
    const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    // Buffers:
    // input: input data (n uint32_t)
    // partialHistograms: output of histogram shader (groupCount * numBins uint32_t)
    // finalHistogram: output of sum shader (numBins uint32_t)
    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(uint32_t), false);
    auto partialHistograms = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), true);
    auto finalHistogram = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * sizeof(uint32_t), true);

    // Uniforms for each stage
    auto histUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(RadixParams), true);
    struct SumParams {
        uint32_t groupCount;
        uint32_t numBins;
    };
    auto sumUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(SumParams), true);

    // Pipelines
    auto histPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{input, partialHistograms, histUniform}
    );

    auto sumPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{partialHistograms, finalHistogram, sumUniform}
    );

    // 1. Define arbitrary histogram counts (non-uniform, sum to n)
    std::vector<uint32_t> expectedHistogram = {50, 200, 5, 0, 123, 87, 150, 42, 12, 30, 77, 89, 25, 10, 2, 122};
    REQUIRE(std::accumulate(expectedHistogram.begin(), expectedHistogram.end(), 0u) == n);

    // 2. Build input data from the histogram
    std::vector<uint32_t> inputData;
    inputData.reserve(n);
    for (uint32_t bin = 0; bin < numBins; ++bin) {
        for (uint32_t count = 0; count < expectedHistogram[bin]; ++count) {
            inputData.push_back(bin);
        }
    }

    // 3. Shuffle the data to remove any ordering bias
    std::mt19937 rng(42);
    std::shuffle(inputData.begin(), inputData.end(), rng);

    // 4. Upload uniform data and input
    RadixParams histParams{
        .bitOffset = 0,
        .numBins = numBins,
        .totalSize = static_cast<uint32_t>(n),
        .itemsPerGroup = itemsPerGroup
    };
    mynydd::uploadUniformData<RadixParams>(contextPtr, histParams, histUniform);
    mynydd::uploadData<uint32_t>(contextPtr, inputData, input);

    SumParams sumParams{groupCount, numBins};
    mynydd::uploadUniformData<SumParams>(contextPtr, sumParams, sumUniform);

    // 5. Execute both pipelines in sequence (chained)
    // histPipeline writes partialHistograms
    // sumPipeline reads partialHistograms and writes finalHistogram
    mynydd::executeBatch<float>(contextPtr, {histPipeline, sumPipeline}, groupCount);

    // 6. Fetch final summed histogram from GPU
    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, finalHistogram, numBins);

    // 7. Check GPU summed histogram matches expected
    for (uint32_t bin = 0; bin < numBins; ++bin) {
        REQUIRE(out[bin] == expectedHistogram[bin]);
    }
}