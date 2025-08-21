#include <cstdint>
#include <sys/types.h>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <chrono>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <vector>

#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/radix_sort.hpp>
#include "test_morton_helpers.hpp"


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

    auto input = std::make_shared<mynydd::Buffer>(contextPtr, n * sizeof(uint32_t), false);
    auto output = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), true);
    auto uniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(mynydd::RadixParams), true);

    auto pipeline = std::make_shared<mynydd::PipelineStep<float>>(
        contextPtr, "shaders/histogram.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{input, output, uniform},
        groupCount
    );

    std::vector<uint32_t> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = i % numBins;
    }

    mynydd::RadixParams params = {
        .bitOffset = 0,
        .numBins = numBins,
        .totalSize = static_cast<uint32_t>(n),
        .itemsPerGroup = itemsPerGroup
    };

    mynydd::uploadUniformData<mynydd::RadixParams>(contextPtr, params, uniform);
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
    auto inputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, partialHistograms.size() * sizeof(uint32_t), false);
    auto outputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, numBins * sizeof(uint32_t), true);

    // Uniform struct matching shader uniform block
    struct SumParams {
        uint32_t groupCount;
        uint32_t numBins;
    } sumParams{groupCount, numBins};

    auto uniformBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, sizeof(SumParams), true);

    // Upload partial histograms and uniform params
    mynydd::uploadData<uint32_t>(contextPtr, partialHistograms, inputBuffer);
    mynydd::uploadUniformData<SumParams>(contextPtr, sumParams, uniformBuffer);

    // Load the summation shader (compiled SPIR-V must match the shader code given)
    auto pipeline = std::make_shared<mynydd::PipelineStep<float>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{inputBuffer, outputBuffer, uniformBuffer},
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

void print_radixes(std::vector<uint32_t>& input_retrieved, uint32_t bitsPerPass, uint32_t nPasses, uint32_t numBins, uint32_t pass) {
    // Validate sorted output
    for (size_t i = 0; i < 512; ++i) {
        // print out input_retrieved, out_sorted, current_radix, input_radix
        std::cerr << "pass " << pass << " I/O: "; 
        for (size_t tmpBitOffset = 0; tmpBitOffset < bitsPerPass * nPasses; tmpBitOffset += bitsPerPass) {
            std::cerr << " " << ((input_retrieved[i] >> tmpBitOffset) & (numBins - 1));
        }
        std::cerr << std::endl;
    }
}


std::vector<uint32_t> runFullRadixSortTest(
    std::shared_ptr<mynydd::VulkanContext> contextPtr,
    std::vector<uint32_t>& inputData
) {

    const size_t n = inputData.size();
    const uint32_t itemsPerGroup = 256;

    mynydd::RadixSortPipeline radixSortPipeline(
        contextPtr, itemsPerGroup, static_cast<uint32_t>(n)
    );

    mynydd::uploadData<uint32_t>(contextPtr, inputData, radixSortPipeline.ioBufferA);

    auto inputBuffer = radixSortPipeline.ioBufferA;
    auto outputBuffer = radixSortPipeline.ioBufferB;

    // For each radix pass (4 passes, 8 bits each)
    // Execute tests for one pass
    for (size_t pass = 0; pass < 4; ++pass) {

        radixSortPipeline.execute_pass(pass);

        inputBuffer = pass % 2 == 0 ? radixSortPipeline.ioBufferA : radixSortPipeline.ioBufferB;
        outputBuffer = pass % 2 == 0 ? radixSortPipeline.ioBufferB : radixSortPipeline.ioBufferA;

        uint32_t bitOffset = pass * radixSortPipeline.bitsPerPass;
        // // std::cerr << "Running radix pass " << pass << " with bit offset " << bitOffset << std::endl;

        auto input_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, inputBuffer, n
        );

        inputData = mynydd::fetchData<uint32_t>(contextPtr, inputBuffer, n);
        
        // ---------------------- TEST GLOBAL HISTOGRAM ----------------------
        auto expected_histogram = compute_full_histogram(inputData, radixSortPipeline.numBins, bitOffset);
        std::vector<uint32_t> out_global_hist = mynydd::fetchData<uint32_t>(contextPtr, radixSortPipeline.globalHistogram, radixSortPipeline.numBins);
        auto out_wg_hist = mynydd::fetchData<uint32_t>(contextPtr, radixSortPipeline.perWorkgroupHistograms, radixSortPipeline.groupCount * radixSortPipeline.numBins);
        REQUIRE(out_global_hist.size() == radixSortPipeline.numBins);

        size_t hist_sum = 0;
        // for (uint32_t i = 0; i < 10; ++i) {
        //     std::cerr << "Global histogram: " << i << ": " << out_global_hist[i] << std::endl;
        // }
        for (uint32_t bin = 0; bin < radixSortPipeline.numBins; ++bin) {
            hist_sum += out_global_hist[bin];
            REQUIRE(out_global_hist[bin] == expected_histogram[bin]);
        }
        REQUIRE(hist_sum == n);

        // ---------------------- TEST WORKGROUP HISTOGRAM ----------------------
        auto expected_wg_histogram = compute_wg_histogram(inputData, radixSortPipeline.numBins, itemsPerGroup, bitOffset);
        // for (uint32_t wg = 0; wg < groupCount; ++wg) {
        //     std:: cerr << "wg: " << wg << " histogram: ";
        //     for (uint32_t bin = 0; bin < numBins; ++bin) {
        //         std::cerr << out_wg_hist[wg * numBins + bin] << " ";
        //     }
        //     std::cerr << std::endl;
        // }
        for (uint32_t bin = 0; bin < expected_wg_histogram.size(); ++bin) {
            REQUIRE(out_wg_hist[bin] == expected_wg_histogram[bin]);
        }
        auto out_wg_hist_transposed = mynydd::fetchData<uint32_t>(contextPtr, radixSortPipeline.transposedHistograms, radixSortPipeline.groupCount * radixSortPipeline.numBins);
        for (uint32_t wg = 0; wg < radixSortPipeline.groupCount; ++wg) {
            for (uint32_t bin = 0; bin < radixSortPipeline.numBins; ++bin) {
                REQUIRE(out_wg_hist[wg * radixSortPipeline.numBins + bin] == out_wg_hist_transposed[bin * radixSortPipeline.groupCount + wg]);
            }
        }
        // for (uint32_t bin = 0; bin < numBins; ++bin) {
        //     size_t wg_hist_sum = 0;
        //     for (uint32_t wg = 0; wg < groupCount; ++wg) {
        //         std::cerr << "Transposed wg hist bin " << bin << ": wg " << wg << ":" << out_wg_hist_transposed[bin * groupCount + wg] << std::endl;
        //     }
        // }

        // ---------------------- TEST GLOBAL PREFIX SUMS ----------------------
        auto out_global_prefix_sum = mynydd::fetchData<uint32_t>(contextPtr, radixSortPipeline.globalPrefixSum, radixSortPipeline.numBins);
        auto expected_global_prefix_sum = prefix_sum(out_global_hist);

        // for (uint32_t i = 0; i < numBins; ++i) {
        //     std::cerr << "Out global prefix sum: " << i << ": " << out_global_prefix_sum[i] << std::endl;
        // }
        for (uint32_t bin = 1; bin < radixSortPipeline.numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] >= out_global_prefix_sum[bin - 1]);
        }
        REQUIRE(out_global_prefix_sum[0] == 0);
        for (uint32_t bin = 0; bin < radixSortPipeline.numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] == expected_global_prefix_sum[bin]);
        }

        // ---------------------- TEST WORKGROUP PREFIX SUMS ----------------------
        auto out_workgroup_prefix_sums = mynydd::fetchData<uint32_t>(contextPtr, radixSortPipeline.workgroupPrefixSums, radixSortPipeline.groupCount * radixSortPipeline.numBins);
        
        for (uint32_t wg = 0; wg < radixSortPipeline.groupCount; ++wg) {
            for (uint32_t bin = 0; bin < radixSortPipeline.numBins; ++bin) {
            }
        }
        // NOTE: THIS IS TRANSPOSED: ROWS ARE OF LENGTH groupCount
        for (uint32_t bin = 0; bin < radixSortPipeline.numBins; ++bin) {
            auto expected_wg_prefix_sum = prefix_sum(std::vector<uint32_t>(
                out_wg_hist_transposed.begin() + bin * radixSortPipeline.groupCount, 
                out_wg_hist_transposed.begin() + (bin + 1) * radixSortPipeline.groupCount
            ));
            for (uint32_t wg = 0; wg < radixSortPipeline.groupCount; ++wg) {
                // std:: cerr << "Workgroup " << wg << " bin " << bin 
                //           << ": " << out_workgroup_prefix_sums[bin * groupCount + wg] 
                //           << ": (expected)" << expected_wg_prefix_sum[wg]
                //           << std::endl;

                REQUIRE(out_workgroup_prefix_sums[bin * radixSortPipeline.groupCount + wg] == expected_wg_prefix_sum[wg]);
            }
        }


        // ---------------------- TEST FINAL SORTING ----------------------
        auto out_sorted = mynydd::fetchData<uint32_t>(
            contextPtr, outputBuffer, n
        );
        input_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, inputBuffer, n
        );

        // Make sure it's not all zeroes
        size_t sum = 0;
        for (size_t i = 1; i < n; ++i) {
            sum += out_sorted[i];
        }
        REQUIRE(sum > 0);
        
        // Check that the output is sorted in this radix
        for (size_t i = 1; i < out_sorted.size(); ++i) {
            uint32_t last_radix = (out_sorted[i - 1] >> bitOffset) & (radixSortPipeline.numBins - 1);
            uint32_t current_radix = (out_sorted[i] >> bitOffset) & (radixSortPipeline.numBins - 1);
            REQUIRE(last_radix <= current_radix);
        }

        // print_radixes(out_sorted, bitsPerPass, nPasses, numBins, pass);
        // Finally, assess stability
        if (pass > 0) {
            // It must also be true that for any given radix position, the previous one must be sorted within that
            for (size_t i = 1; i < n; ++i) {
                uint32_t last_radix = (out_sorted[i] >> bitOffset) & (radixSortPipeline.numBins - 1);
                uint32_t prev_radix = (out_sorted[i-1] >> bitOffset) & (radixSortPipeline.numBins - 1);
                uint32_t prev_radix_prev_pass = (out_sorted[i - 1] >> (bitOffset - 8)) & (radixSortPipeline.numBins - 1);
                uint32_t last_radix_prev_pass = (out_sorted[i] >> (bitOffset - 8)) & (radixSortPipeline.numBins - 1);
                // std::cerr << "Pass " << pass << "Last radix: " << last_radix 
                //           << ", Last radix prev pass: " << last_radix_prev_pass 
                //           << std::endl;
                if (last_radix == prev_radix) {
                    REQUIRE(prev_radix_prev_pass <= last_radix_prev_pass);
                }
            }
        }
        
    }
    auto output_retrieved = mynydd::fetchData<uint32_t>(
        contextPtr, outputBuffer, n
    );
    // print_radixes(output_retrieved, bitsPerPass, nPasses, numBins);
    for (size_t i = 1; i < n; ++i) {
        REQUIRE(output_retrieved[i] >= output_retrieved[i - 1]);
    }
    return output_retrieved;
}

struct CellInfo {
    uint start;
    uint count;
};

std::vector<CellInfo> runSortedKeys2IndexTest(
    std::shared_ptr<mynydd::VulkanContext> contextPtr,
    std::vector<uint32_t>& sorted_keys,
    uint32_t nCells
) {

    uint32_t nKeys = sorted_keys.size();

    struct IndexParams {
        uint32_t nKeys;
    } params{nKeys};


    auto inputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, nKeys * sizeof(uint32_t), false);
    auto outputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, nKeys * sizeof(CellInfo), true);
    auto uniformBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, sizeof(IndexParams), true);

    mynydd::uploadData<uint32_t>(contextPtr, sorted_keys, inputBuffer);
    mynydd::uploadUniformData<IndexParams>(contextPtr, params, uniformBuffer);

    auto groupCount = (nKeys + 63) / 64;

    auto pipeline = std::make_shared<mynydd::PipelineStep<Particle>>(
        contextPtr, "shaders/build_index_from_sorted_keys.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            inputBuffer, outputBuffer, uniformBuffer
        },
        groupCount
    );

    mynydd::executeBatch<Particle>(contextPtr, {pipeline});

    std::vector<CellInfo> outIndex = mynydd::fetchData<CellInfo>(contextPtr, outputBuffer, nCells);

    for (uint32_t ak = 0; ak < nCells; ++ak) {
        auto& cell = outIndex[ak];
        REQUIRE(cell.start < nKeys);
        if (cell.count > 0) {
            for (uint i = cell.start; i < cell.start + cell.count; ++i) {
                REQUIRE(sorted_keys[i] == ak);
            }
        }
    }

    return outIndex;
}


TEST_CASE("Full 32-bit radix sort pipeline with 8-bit passes", "[sort]") {
    const size_t n = 1 << 16; // 65536 elements for test
    std::vector<uint32_t> inputData(n);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (auto& v : inputData) v = dist(rng);
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    auto output_retrieved = runFullRadixSortTest(contextPtr, inputData);
}

void run_full_pipeline_morton(uint32_t nBits) {
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    auto particles = getMortonTestGridRegularParticleData(nBits);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto morton_keys = runMortonTest(contextPtr, nBits, particles);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto sorted_keys = runFullRadixSortTest(contextPtr, morton_keys);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto final_index = runSortedKeys2IndexTest(contextPtr, sorted_keys, nBits);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto duration_morton = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto duration_sort = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto duration_index = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    std::cerr << "TEST: Running full pipeline Morton test with " << nBits << " bits..." << std::endl;
    std::cerr << "TEST: Morton keys generation took: " << duration_morton << " µs" << std::endl;
    std::cerr << "TEST: Sorting took: " << duration_sort << " µs" << std::endl;
    std::cerr << "TEST: Indexing took: " << duration_index << " µs" << std::endl;
}

TEST_CASE("Test Morton (2 bits) + sort + final index", "[morton]") {
    run_full_pipeline_morton(2);
}

TEST_CASE("Test Morton (3 bits) + sort + final index", "[morton]") {
    run_full_pipeline_morton(3);
}

TEST_CASE("Test Morton (4 bits) + sort + final index", "[morton]") {
    run_full_pipeline_morton(4);
}

TEST_CASE("Test Morton (5 bits) + sort + final index", "[morton]") {
    run_full_pipeline_morton(5);
}