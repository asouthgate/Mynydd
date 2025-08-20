std::vector<uint32_t> runFullRadixSortTest(
    std::shared_ptr<mynydd::VulkanContext> contextPtr,
    std::vector<uint32_t>& inputData
) {

    // for (size_t i = 0; i < 10; ++i) {
    //     std::cerr << "Input data: " << i << ": " << inputData[i] << std::endl;
    // }
    // for (size_t i = inputData.size()-10; i < inputData.size(); ++i) {
    //     std::cerr << "Input data: " << i << ": " << inputData[i] << std::endl;
    // }

    const size_t n = inputData.size();

    const uint32_t bitsPerPass = 8;
    const uint32_t nPasses = 32 / bitsPerPass; // 4 passes for 32 bits
    const uint32_t numBins = 1 << bitsPerPass; // 256
    const uint32_t itemsPerGroup = 256;
    const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

    auto ioBufferA = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);
    auto ioBufferB = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);

    auto perWorkgroupHistograms = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), false);
    auto globalHistogram = std::make_shared<mynydd::Buffer>(contextPtr, numBins * sizeof(uint32_t), false);
    auto globalPrefixSum = std::make_shared<mynydd::Buffer>(contextPtr, numBins * sizeof(uint32_t), false);
    auto transposedHistograms = std::make_shared<mynydd::Buffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);
    auto workgroupPrefixSums = std::make_shared<mynydd::Buffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);

    auto radixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(RadixParams), true);
    auto sumUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(SumParams), true);
    auto workgroupPrefixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
    auto globalPrefixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
    auto transposeUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
    auto sortUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(SortParams), true);

    // Load compute pipelines
    auto histPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferA, perWorkgroupHistograms, radixUniform},
        groupCount
    );

    auto histPipelinePong = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/histogram.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferB, perWorkgroupHistograms, radixUniform},
        groupCount
    );

    auto sumPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/histogram_sum.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, globalHistogram, sumUniform},
        1
    );
    // std::cerr << "n: " << n << std::endl;
    // std::cerr << "itemsPerGroup: " << itemsPerGroup << std::endl;
    // std::cerr << "groupCount: " << groupCount << std::endl;
    // std::cerr << "numBins: " << numBins << std::endl;

    auto transposePipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/transpose.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, transposedHistograms, transposeUniform},
        (numBins * groupCount + numBins - 1) / numBins
    );

    auto workgroupPrefixPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{transposedHistograms, workgroupPrefixSums, workgroupPrefixUniform},
        numBins
    );

    
    auto globalPrefixPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{globalHistogram, globalPrefixSum, globalPrefixUniform},
        1
    );

    auto sortPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/radix_sort.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            ioBufferA,
            workgroupPrefixSums,
            globalPrefixSum,
            ioBufferB,
            sortUniform
        },
        groupCount
    );
    auto sortPipelinePong = std::make_shared<mynydd::PipelineStep<uint32_t>>(
        contextPtr, "shaders/radix_sort.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            ioBufferB,
            workgroupPrefixSums,
            globalPrefixSum,
            ioBufferA,
            sortUniform
        },
        groupCount
    );

    mynydd::uploadData<uint32_t>(contextPtr, inputData, ioBufferA);

    auto inputBuffer = ioBufferA;
    auto outputBuffer = ioBufferB;

    // For each radix pass (4 passes, 8 bits each)
    // Execute tests for one pass
    for (size_t pass = 0; pass < nPasses; ++pass) {

        inputBuffer = pass % 2 == 0 ? ioBufferA : ioBufferB;
        outputBuffer = pass % 2 == 0 ? ioBufferB : ioBufferA;

        uint32_t bitOffset = pass * bitsPerPass;
        // std::cerr << "Running radix pass " << pass << " with bit offset " << bitOffset << std::endl;

        auto input_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, inputBuffer, n
        );
        
        // print_radixes(input_retrieved, bitsPerPass, nPasses, numBins, pass);

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

        PrefixParams transposeParams = {
            .groupCount = groupCount,
            .numBins = numBins
        };

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

        inputData = mynydd::fetchData<uint32_t>(contextPtr, inputBuffer, n);
        
        // ---------------------- TEST GLOBAL HISTOGRAM ----------------------
        auto expected_histogram = compute_full_histogram(inputData, numBins, bitOffset);
        std::vector<uint32_t> out_global_hist = mynydd::fetchData<uint32_t>(contextPtr, globalHistogram, numBins);
        auto out_wg_hist = mynydd::fetchData<uint32_t>(contextPtr, perWorkgroupHistograms, groupCount * numBins);
        REQUIRE(out_global_hist.size() == numBins);

        size_t hist_sum = 0;
        // for (uint32_t i = 0; i < 10; ++i) {
        //     std::cerr << "Global histogram: " << i << ": " << out_global_hist[i] << std::endl;
        // }
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            hist_sum += out_global_hist[bin];
            REQUIRE(out_global_hist[bin] == expected_histogram[bin]);
        }
        REQUIRE(hist_sum == n);

        // ---------------------- TEST WORKGROUP HISTOGRAM ----------------------
        auto expected_wg_histogram = compute_wg_histogram(inputData, numBins, itemsPerGroup, bitOffset);
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
        auto out_wg_hist_transposed = mynydd::fetchData<uint32_t>(contextPtr, transposedHistograms, groupCount * numBins);
        for (uint32_t wg = 0; wg < groupCount; ++wg) {
            for (uint32_t bin = 0; bin < numBins; ++bin) {
                REQUIRE(out_wg_hist[wg * numBins + bin] == out_wg_hist_transposed[bin * groupCount + wg]);
            }
        }
        // for (uint32_t bin = 0; bin < numBins; ++bin) {
        //     size_t wg_hist_sum = 0;
        //     for (uint32_t wg = 0; wg < groupCount; ++wg) {
        //         std::cerr << "Transposed wg hist bin " << bin << ": wg " << wg << ":" << out_wg_hist_transposed[bin * groupCount + wg] << std::endl;
        //     }
        // }

        // ---------------------- TEST GLOBAL PREFIX SUMS ----------------------
        auto out_global_prefix_sum = mynydd::fetchData<uint32_t>(contextPtr, globalPrefixSum, numBins);
        auto expected_global_prefix_sum = prefix_sum(out_global_hist);

        // for (uint32_t i = 0; i < numBins; ++i) {
        //     std::cerr << "Out global prefix sum: " << i << ": " << out_global_prefix_sum[i] << std::endl;
        // }
        for (uint32_t bin = 1; bin < numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] >= out_global_prefix_sum[bin - 1]);
        }
        REQUIRE(out_global_prefix_sum[0] == 0);
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            REQUIRE(out_global_prefix_sum[bin] == expected_global_prefix_sum[bin]);
        }

        // ---------------------- TEST WORKGROUP PREFIX SUMS ----------------------
        auto out_workgroup_prefix_sums = mynydd::fetchData<uint32_t>(contextPtr, workgroupPrefixSums, groupCount * numBins);
        
        for (uint32_t wg = 0; wg < groupCount; ++wg) {
            for (uint32_t bin = 0; bin < numBins; ++bin) {
            }
        }
        // NOTE: THIS IS TRANSPOSED: ROWS ARE OF LENGTH groupCount
        for (uint32_t bin = 0; bin < numBins; ++bin) {
            auto expected_wg_prefix_sum = prefix_sum(std::vector<uint32_t>(
                out_wg_hist_transposed.begin() + bin * groupCount, 
                out_wg_hist_transposed.begin() + (bin + 1) * groupCount
            ));
            for (uint32_t wg = 0; wg < groupCount; ++wg) {
                // std:: cerr << "Workgroup " << wg << " bin " << bin 
                //           << ": " << out_workgroup_prefix_sums[bin * groupCount + wg] 
                //           << ": (expected)" << expected_wg_prefix_sum[wg]
                //           << std::endl;

                REQUIRE(out_workgroup_prefix_sums[bin * groupCount + wg] == expected_wg_prefix_sum[wg]);
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
        for (size_t i = 1; i < 50; ++i) {
            sum += out_sorted[i];
        }
        REQUIRE(sum > 0);
        
        // Check that the output is sorted in this radix
        for (size_t i = 1; i < out_sorted.size(); ++i) {
            uint32_t last_radix = (out_sorted[i - 1] >> bitOffset) & (numBins - 1);
            uint32_t current_radix = (out_sorted[i] >> bitOffset) & (numBins - 1);
            REQUIRE(last_radix <= current_radix);
        }

        // print_radixes(out_sorted, bitsPerPass, nPasses, numBins, pass);
        // Finally, assess stability
        if (pass > 0) {
            // It must also be true that for any given radix position, the previous one must be sorted within that
            for (size_t i = 1; i < n; ++i) {
                uint32_t last_radix = (out_sorted[i] >> bitOffset) & (numBins - 1);
                uint32_t prev_radix = (out_sorted[i-1] >> bitOffset) & (numBins - 1);
                uint32_t prev_radix_prev_pass = (out_sorted[i - 1] >> (bitOffset - 8)) & (numBins - 1);
                uint32_t last_radix_prev_pass = (out_sorted[i] >> (bitOffset - 8)) & (numBins - 1);
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
