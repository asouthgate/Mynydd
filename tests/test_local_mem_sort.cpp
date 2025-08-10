#include <cstdint>
#include <sys/types.h>
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

template<typename T>
std::vector<size_t> prefix_sum(const std::vector<T>& input) {
    size_t n = input.size();
    std::vector<size_t> output(n, 0);

    output[0] = 0;
    for (size_t i = 1; i < n; ++i) {
        output[i] = input[i-1] + output[i - 1];
    }
    
    return output;
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


TEST_CASE("Full 32-bit radix sort pipeline with 8-bit passes", "[sort]") {
    std::cerr << "\nRunning full radix sort test..." << std::endl;
    const size_t n = 1 << 16; // 65536 elements for test
    const uint32_t bitsPerPass = 8;
    const uint32_t nPasses = 32 / bitsPerPass; // 4 passes for 32 bits
    const uint32_t numBins = 1 << bitsPerPass; // 256
    const uint32_t itemsPerGroup = 256;
    const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto inputBuffer = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);
    auto outputBuffer = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);

    auto perWorkgroupHistograms = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), false);
    auto globalHistogram = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * sizeof(uint32_t), false);
    auto globalPrefixSum = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * sizeof(uint32_t), false);
    auto transposedHistograms = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);
    auto workgroupPrefixSums = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);

    auto radixUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(RadixParams), true);
    auto sumUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(SumParams), true);
    auto workgroupPrefixUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(PrefixParams), true);
    auto globalPrefixUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(PrefixParams), true);
    auto transposeUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(PrefixParams), true);
    auto sortUniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(SortParams), true);

    // Load compute pipelines
    auto histPipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{inputBuffer, perWorkgroupHistograms, radixUniform},
        groupCount
    );

    auto histPipelinePong = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{outputBuffer, perWorkgroupHistograms, radixUniform},
        groupCount
    );

    auto sumPipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{perWorkgroupHistograms, globalHistogram, sumUniform},
        1
    );

    auto transposePipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/transpose.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{perWorkgroupHistograms, transposedHistograms, transposeUniform},
        (numBins + 15) / 16, (groupCount + 15) / 16, 1
    );

    auto workgroupPrefixPipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{transposedHistograms, workgroupPrefixSums, workgroupPrefixUniform},
        numBins
    );

    auto globalPrefixPipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{globalHistogram, globalPrefixSum, globalPrefixUniform},
        1
    );

    auto sortPipeline = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/radix_sort.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{
            inputBuffer,
            workgroupPrefixSums,
            globalPrefixSum,
            outputBuffer,
            sortUniform
        },
        groupCount
    );
    auto sortPipelinePong = std::make_shared<mynydd::ComputeEngine<uint32_t>>(
        contextPtr, "shaders/radix_sort.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{
            outputBuffer,
            workgroupPrefixSums,
            globalPrefixSum,
            inputBuffer,
            sortUniform
        },
        groupCount
    );

    // std::cerr << "Setup pipelines" << std::endl;

    // Generate random input
    std::vector<uint32_t> inputData(n);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (auto& v : inputData) v = dist(rng);
    mynydd::uploadData<uint32_t>(contextPtr, inputData, inputBuffer);

    // For each radix pass (4 passes, 8 bits each)
    // Execute tests for one pass
    for (size_t pass = 0; pass < nPasses; ++pass) {
        uint32_t bitOffset = pass * bitsPerPass;
        std::cerr << "Running radix pass " << pass << " with bit offset " << bitOffset << std::endl;

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

        PrefixParams workgroupPrefixParams = {
            .groupCount = numBins, // Yes, this is inverted, because it's using the output of transpose shader step
            .numBins = groupCount
        };

        PrefixParams transposeParams = workgroupPrefixParams;

        PrefixParams globalPrefixParams = {
            .groupCount = 1,
            .numBins = numBins
        };

        SortParams sortParams = {
            .bitOffset = bitOffset,
            .numBins = numBins,
            .totalSize = static_cast<uint32_t>(n),
            .workgroupSize=itemsPerGroup,
            .groupCount=groupCount
        };

        mynydd::uploadUniformData<RadixParams>(contextPtr, radixParams, radixUniform);
        mynydd::uploadUniformData<SumParams>(contextPtr, sumParams, sumUniform);
        mynydd::uploadUniformData<PrefixParams>(contextPtr, globalPrefixParams, globalPrefixUniform);
        mynydd::uploadUniformData<PrefixParams>(contextPtr, workgroupPrefixParams, workgroupPrefixUniform);
        mynydd::uploadUniformData<PrefixParams>(contextPtr, transposeParams, transposeUniform);
        mynydd::uploadUniformData<SortParams>(contextPtr, sortParams, sortUniform);

        // // 1) Histogram partial counts
        mynydd::executeBatch<uint32_t>(
            contextPtr, 
        {
                pass % 2 == 0 ? histPipeline : histPipelinePong,
                sumPipeline,
                globalPrefixPipeline, 
                transposePipeline,
                workgroupPrefixPipeline,
                pass % 2 == 0 ? sortPipeline : sortPipelinePong
            }
        );

        if (pass == 0) {
            inputData = inputData;
        } else {
            inputData = mynydd::fetchData<uint32_t>(contextPtr, pass % 2 == 0 ? inputBuffer : outputBuffer, n);
        }
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
        auto out_global_prefix_sum = mynydd::fetchData<uint32_t>(contextPtr, globalPrefixSum, numBins);
        auto expected_global_prefix_sum = prefix_sum(out_global_hist);

        for (uint32_t bin = 1; bin < numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] >= out_global_prefix_sum[bin - 1]);
        }
        // REQUIRE(out_global_prefix_sum[numBins - 1] == n); // not required for an exclusive scan
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] == expected_global_prefix_sum[bin]);
        }
        auto out_workgroup_prefix_sums = mynydd::fetchData<uint32_t>(contextPtr, workgroupPrefixSums, groupCount * numBins);
        
        // NOTE: THIS IS TRANSPOSED: ROWS ARE OF LENGTH groupCount
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            auto expected_wg_hist = prefix_sum(std::vector<uint32_t>(
                out_wg_hist_transposed.begin() + bin * groupCount, 
                out_wg_hist_transposed.begin() + (bin + 1) * groupCount
            ));
            for (uint32_t wg = 0; wg < groupCount; ++wg) {
                REQUIRE(out_workgroup_prefix_sums[bin * groupCount + wg] == expected_wg_hist[wg]);
            }
        }

        auto out_sorted = mynydd::fetchData<uint32_t>(
            contextPtr, pass % 2 ?  inputBuffer : outputBuffer, n
        );
        auto input_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, pass % 2 ? outputBuffer : inputBuffer, n
        );
        // Validate sorted output
        for (size_t i = 1; i < 100; ++i) {
            // print out input_retrieved, out_sorted, current_radix, input_radix
            std::cerr << "Input: " << input_retrieved[i] << ", Sorted: " << out_sorted[i] << std::endl;
            std::cerr << "Last radix: " << ((out_sorted[i - 1] >> bitOffset) & (numBins - 1)) 
                      << ", Current radix: " << ((out_sorted[i] >> bitOffset) & (numBins - 1)) 
                      << ", Input radix: " << ((input_retrieved[i] >> bitOffset) & (numBins - 1)) << std::endl;
        }
        for (size_t i = 1; i < out_sorted.size(); ++i) {
            uint32_t last_radix = (out_sorted[i - 1] >> bitOffset) & (numBins - 1);
            uint32_t current_radix = (out_sorted[i] >> bitOffset) & (numBins - 1);
            uint32_t input_radix = (input_retrieved[i] >> bitOffset) & (numBins - 1);
            REQUIRE(last_radix <= current_radix);
        }

    }
}