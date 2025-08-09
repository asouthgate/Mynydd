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


TEST_CASE("Radix histogram matches arbitrary bin distribution", "[sort]") {
    const size_t n = 1024;
    const uint32_t numBins = 16;
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
    std::mt19937 rng(42); // fixed seed for repeatability
    std::shuffle(inputData.begin(), inputData.end(), rng);

    // 4. Upload uniform and data
    RadixParams params = {
        .bitOffset = 0,
        .numBins = numBins,
        .totalSize = static_cast<uint32_t>(n),
        .itemsPerGroup = itemsPerGroup
    };

    mynydd::uploadUniformData<RadixParams>(contextPtr, params, uniform);
    mynydd::uploadData<uint32_t>(contextPtr, inputData, input);

    // 5. Execute shader
    mynydd::executeBatch<float>(contextPtr, {pipeline}, groupCount);

    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, output, groupCount * numBins);

    // 6. Combine all workgroup histograms
    std::vector<uint32_t> combinedHistogram(numBins, 0);
    for (uint32_t group = 0; group < groupCount; ++group) {
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            combinedHistogram[bin] += out[group * numBins + bin];
        }
    }

    // 7. Verify output matches expected histogram
    for (uint32_t bin = 0; bin < numBins; ++bin) {
        REQUIRE(combinedHistogram[bin] == expectedHistogram[bin]);
    }
}