#include <cstdint>
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

struct SumParams {
    uint32_t groupCount;
    uint32_t numBins;
};

struct PrefixParams {
    uint32_t groupCount;
    uint32_t numBins;
};

struct SortParams {
    uint bitOffset;
    uint numBins;
    uint totalSize;
    uint workgroupSize;
    uint groupCount;
};


// Test function for validation
template<typename T>
std::vector<size_t> compute_wg_histogram(
    const std::vector<T>& input, 
    uint32_t numBins, 
    uint32_t itemsPerGroup,
    uint32_t bitOffset // NEW: which bits to start from
) {
    size_t n = input.size();
    size_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;
    std::vector<size_t> histogram(groupCount * numBins, 0); // <-- histogram should be size_t, not T

    // Mask optimization if numBins is power of two
    bool powerOfTwo = (numBins & (numBins - 1)) == 0;
    uint32_t mask = powerOfTwo ? (numBins - 1) : 0;

    for (size_t i = 0; i < n; ++i) {
        uint32_t shifted = static_cast<uint32_t>(input[i] >> bitOffset);
        uint32_t bin = powerOfTwo ? (shifted & mask) : (shifted % numBins);
        size_t groupIndex = i / itemsPerGroup;
        histogram[groupIndex * numBins + bin]++;
    }

    return histogram;
}

// Test function for validation
template<typename T>
std::vector<size_t> compute_full_histogram(
    const std::vector<T>& input, 
    uint32_t numBins, 
    uint32_t bitOffset // NEW: which bits to start from
) {
    size_t n = input.size();
    std::vector<size_t> histogram(numBins, 0); // <-- histogram should be size_t, not T

    // Mask optimization if numBins is power of two
    bool powerOfTwo = (numBins & (numBins - 1)) == 0;
    uint32_t mask = powerOfTwo ? (numBins - 1) : 0;

    for (size_t i = 0; i < n; ++i) {
        uint32_t shifted = static_cast<uint32_t>(input[i] >> bitOffset);
        uint32_t bin = powerOfTwo ? (shifted & mask) : (shifted % numBins);
        histogram[bin]++;
    }

    return histogram;
}


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
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{input, output, uniform},
        groupCount
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

    mynydd::executeBatch<float>(contextPtr, {pipeline});

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
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{inputBuffer, outputBuffer, uniformBuffer},
        1
    );

    // Dispatch exactly 1 workgroup with 256 threads
    mynydd::executeBatch<float>(contextPtr, {pipeline});

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
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{input, partialHistograms, histUniform},
        groupCount
    );

    auto sumPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{partialHistograms, finalHistogram, sumUniform},
        groupCount
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
    mynydd::executeBatch<float>(contextPtr, {histPipeline, sumPipeline});

    // 6. Fetch final summed histogram from GPU
    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, finalHistogram, numBins);

    // 7. Check GPU summed histogram matches expected
    for (uint32_t bin = 0; bin < numBins; ++bin) {
        REQUIRE(out[bin] == expectedHistogram[bin]);
    }
}


TEST_CASE("Full 32-bit radix sort pipeline with 8-bit passes", "[sort]") {
    std::cerr << "\nRunning full radix sort test..." << std::endl;
    const size_t n = 1 << 16; // 65536 elements for test
    const uint32_t bitsPerPass = 8;
    const uint32_t numBins = 1 << bitsPerPass; // 256
    const uint32_t itemsPerGroup = 256;
    const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto inputBuffer = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(uint32_t), false);
    // auto outputBuffer = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(uint32_t), false);

    auto perWorkgroupHistograms = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), false);
    auto globalHistogram = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * sizeof(uint32_t), false);
    // auto globalPrefixSum = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * sizeof(uint32_t), false);
    auto transposedHistograms = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);
    // auto workgroupPrefixSums = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);

    auto radixUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(RadixParams), true);
    auto sumUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(SumParams), true);
    // auto prefixUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(PrefixParams), true);
    auto transposeUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(PrefixParams), true);
    // auto sortUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(SortParams), true);

    // Load compute pipelines
    auto histPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{inputBuffer, perWorkgroupHistograms, radixUniform},
        groupCount
    );

    auto sumPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{perWorkgroupHistograms, globalHistogram, sumUniform},
        1
    );

    auto transposePipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/transpose.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{perWorkgroupHistograms, transposedHistograms, transposeUniform},
        (numBins + 15) / 16, (groupCount + 15) / 16, 1);

    // auto prefixPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
    //     contextPtr, "shaders/workgroup_scan.comp.spv",
    //     std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{transposedHistograms, workgroupPrefixSums, prefixUniform},
    //     numBins);

    // auto globalPrefixPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
    //     contextPtr, "shaders/histogram_sum.comp.spv",
    //     std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{globalHistogram, globalPrefixSum, sumUniform},
    //     1);

    // auto sortPipeline = std::make_shared<mynydd::ComputeEngine<float>>(
    //     contextPtr, "shaders/radix_sort.comp.spv",
    //     std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{inputBuffer, outputBuffer, workgroupPrefixSums, globalPrefixSum, sortUniform},
    //     groupCount);

    // std::cerr << "Setup pipelines" << std::endl;

    // Generate random input
    std::vector<uint32_t> inputData(n);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (auto& v : inputData) v = dist(rng);
    mynydd::uploadData<uint32_t>(contextPtr, inputData, inputBuffer);

    // For each radix pass (4 passes, 8 bits each)
    for (uint32_t pass = 0; pass < 4; ++pass) {
        uint32_t bitOffset = pass * bitsPerPass;

        RadixParams radixParams = {
            .bitOffset = bitOffset,
            .numBins = numBins,
            .totalSize = static_cast<uint32_t>(n),
            .itemsPerGroup = itemsPerGroup
        };

        SumParams sumParams = {
            .groupCount = groupCount,
            .numBins = numBins
        };

        PrefixParams prefixParams = {
            .groupCount = groupCount,
            .numBins = numBins
        };

        PrefixParams transposeParams = prefixParams;

        // SortParams sortParams = {
        //     .bitOffset = bitOffset,
        //     .numBins = numBins,
        //     .totalSize = static_cast<uint32_t>(n),
        //     .workgroupSize=itemsPerGroup,
        //     .groupCount=groupCount
        // };

        mynydd::uploadUniformData<RadixParams>(contextPtr, radixParams, radixUniform);
        mynydd::uploadUniformData<SumParams>(contextPtr, sumParams, sumUniform);
        // mynydd::uploadUniformData<PrefixParams>(contextPtr, prefixParams, prefixUniform);
        mynydd::uploadUniformData<PrefixParams>(contextPtr, transposeParams, transposeUniform);
        // mynydd::uploadUniformData<SortParams>(contextPtr, sortParams, sortUniform);

        // std::cerr << "Starting radix pass " << pass << " with bit offset " << bitOffset << std::endl;
        // // 1) Histogram partial counts
        mynydd::executeBatch<float>(
            contextPtr, 
        {
                histPipeline,
                sumPipeline,
                // globalPrefixPipeline, 
                transposePipeline
                // prefixPipeline,
                // sortPipeline
            }
        );

        // Validate intermediate results
        auto expected_histogram = compute_full_histogram(inputData, numBins, bitOffset);
        auto expected_wg_histogram = compute_wg_histogram(inputData, numBins, itemsPerGroup, bitOffset);
        std::vector<uint32_t> out_global_hist = mynydd::fetchData<uint32_t>(contextPtr, globalHistogram, numBins);
        auto out_wg_hist = mynydd::fetchData<uint32_t>(contextPtr, perWorkgroupHistograms, groupCount * numBins);
        REQUIRE(out_global_hist.size() == numBins);
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            REQUIRE(out_global_hist[bin] == expected_histogram[bin]);
        }
        for (uint32_t bin = 0; bin < expected_wg_histogram.size(); ++bin) {
            REQUIRE(out_wg_hist[bin] == expected_wg_histogram[bin]);
        }
        auto out_wg_hist_transposed = mynydd::fetchData<uint32_t>(contextPtr, transposedHistograms, groupCount * numBins);
        for (uint32_t wg = 0; wg < groupCount; ++wg) {
            for (uint32_t bin = 0; bin < numBins; ++bin) {
                REQUIRE(out_wg_hist_transposed[wg * numBins + bin] == out_wg_hist[bin * groupCount + wg]);
            }
        }
        // REQUIRE(false);
        // REQUIRE(false);
        // Swap input/output buffers for next pass
        // std::swap(inputBuffer, outputBuffer);
        // std::cerr << "Completed radix pass " << pass << std::endl;
        // Update histogram pipeline buffers to new input
        // sortPipeline->setBuffers(contextPtr, {inputBuffer, outputBuffer, workgroupPrefixSums, globalPrefixSum, sortUniform});
    }

    // Fetch final sorted output from inputBuffer (after last swap, inputBuffer holds sorted data)
    // std::vector<uint32_t> gpuSorted = mynydd::fetchData<uint32_t>(contextPtr, inputBuffer, n);

    // CPU reference sort
    // std::vector<uint32_t> cpuSorted = inputData;
    // std::sort(cpuSorted.begin(), cpuSorted.end());

    // // Validate GPU sort matches CPU sort
    // REQUIRE(gpuSorted.size() == cpuSorted.size());
    // for (size_t i = 0; i < n; ++i) {
    //     REQUIRE(gpuSorted[i] == cpuSorted[i]);
    // }
}