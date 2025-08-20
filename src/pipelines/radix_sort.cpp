#include <assert.h>
#include <cstddef>
#include <cstring>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "../include/mynydd/mynydd.hpp"
#include "../include/mynydd/pipelines/radix_sort.hpp"

using namespace mynydd;

namespace mynydd {

    RadixSortPipeline::RadixSortPipeline(
        std::shared_ptr<VulkanContext> contextPtr, 
        uint32_t itemsPerGroup, 
        uint32_t nInputElements
    ) : contextPtr(contextPtr),
        numBins(1 << bitsPerPass), 
        nPasses(32 / bitsPerPass),
        nInputElements(nInputElements),
        groupCount((nInputElements + itemsPerGroup - 1) / itemsPerGroup)
    {

        // const size_t n = inputData.size();

        // const uint32_t groupCount = (n + itemsPerGroup - 1) / itemsPerGroup;

        ioBufferA = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);
        ioBufferB = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * itemsPerGroup * sizeof(uint32_t), false);

        perWorkgroupHistograms = std::make_shared<mynydd::Buffer>(contextPtr, groupCount * numBins * sizeof(uint32_t), false);
        globalHistogram = std::make_shared<mynydd::Buffer>(contextPtr, numBins * sizeof(uint32_t), false);
        globalPrefixSum = std::make_shared<mynydd::Buffer>(contextPtr, numBins * sizeof(uint32_t), false);
        transposedHistograms = std::make_shared<mynydd::Buffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);
        workgroupPrefixSums = std::make_shared<mynydd::Buffer>(contextPtr, numBins * groupCount * sizeof(uint32_t), false);

        radixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(RadixParams), true);
        sumUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(SumParams), true);
        workgroupPrefixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
        globalPrefixUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
        transposeUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);
        sortUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(SortParams), true);

        // Load compute pipelines
        histPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/histogram.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferA, perWorkgroupHistograms, radixUniform},
            groupCount
        );

        histPipelinePong = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/histogram.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferB, perWorkgroupHistograms, radixUniform},
            groupCount
        );

        sumPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/histogram_sum.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, globalHistogram, sumUniform},
            1
        );

        transposePipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/transpose.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, transposedHistograms, transposeUniform},
            (numBins * groupCount + numBins - 1) / numBins
        );

        workgroupPrefixPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/workgroup_scan.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{transposedHistograms, workgroupPrefixSums, workgroupPrefixUniform},
            numBins
        );

        globalPrefixPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
            contextPtr, "shaders/workgroup_scan.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{globalHistogram, globalPrefixSum, globalPrefixUniform},
            1
        );

        sortPipeline = std::make_shared<mynydd::PipelineStep<uint32_t>>(
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
        sortPipelinePong = std::make_shared<mynydd::PipelineStep<uint32_t>>(
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
    }

    void RadixSortPipeline::execute() {
        for (size_t pass = 0; pass < nPasses; ++pass) {
            execute_pass(pass);
        }
    }

    void RadixSortPipeline::execute_pass(size_t pass) {
        auto inputBuffer = pass % 2 == 0 ? ioBufferA : ioBufferB;
        auto outputBuffer = pass % 2 == 0 ? ioBufferB : ioBufferA;

        uint32_t bitOffset = pass * bitsPerPass;
        // std::cerr << "Running radix pass " << pass << " with bit offset " << bitOffset << std::endl;

        auto input_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, inputBuffer, nInputElements
        );
        
        // print_radixes(input_retrieved, bitsPerPass, nPasses, numBins, pass);

        RadixParams radixParams = {
            .bitOffset = bitOffset,
            .numBins = numBins,
            .totalSize = nInputElements,
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
            .totalSize = nInputElements,
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
    }

}