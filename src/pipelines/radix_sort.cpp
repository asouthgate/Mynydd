#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

        this->itemsPerGroup = itemsPerGroup;

        if (groupCount * itemsPerGroup < nInputElements) {
            throw std::runtime_error("groupCount * itemsPerGroup cannot be less than nInputElements.");
        }

        ioBufferA = std::make_shared<mynydd::Buffer>(contextPtr, nInputElements * sizeof(uint32_t), false);
        ioBufferB = std::make_shared<mynydd::Buffer>(contextPtr, nInputElements * sizeof(uint32_t), false);

        ioSortedIndicesA = std::make_shared<mynydd::Buffer>(contextPtr, nInputElements * sizeof(uint32_t), false);
        ioSortedIndicesB = std::make_shared<mynydd::Buffer>(contextPtr, nInputElements * sizeof(uint32_t), false);

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

        initRangePipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/init_range_index.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{ioSortedIndicesB}, // B will be prev for the first pass
            groupCount,
            1,
            1,
            std::vector<uint32_t>{sizeof(uint32_t)}
        );
    
        // Load compute pipelines
        histPipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/histogram.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferA, perWorkgroupHistograms, radixUniform},
            groupCount
        );

        histPipelinePong = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/histogram.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{ioBufferB, perWorkgroupHistograms, radixUniform},
            groupCount
        );

        sumPipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/histogram_sum.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, globalHistogram, sumUniform},
            1
        );

        transposePipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/transpose.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{perWorkgroupHistograms, transposedHistograms, transposeUniform},
            (numBins * groupCount + numBins - 1) / numBins
        );

        workgroupPrefixPipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/workgroup_scan.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{transposedHistograms, workgroupPrefixSums, workgroupPrefixUniform},
            numBins
        );

        globalPrefixPipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/workgroup_scan.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{globalHistogram, globalPrefixSum, globalPrefixUniform},
            1
        );

        sortPipeline = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/radix_sort.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{
                ioBufferA,
                workgroupPrefixSums,
                globalPrefixSum,
                ioSortedIndicesB,
                ioBufferB,
                ioSortedIndicesA,
                sortUniform
            },
            groupCount
        );
        sortPipelinePong = std::make_shared<mynydd::PipelineStep>(
            contextPtr, "shaders/radix_sort.comp.spv",
            std::vector<std::shared_ptr<mynydd::Buffer>>{
                ioBufferB,
                workgroupPrefixSums,
                globalPrefixSum,
                ioSortedIndicesA,
                ioBufferA,
                ioSortedIndicesB,
                sortUniform
            },
            groupCount
        );
    }

    void RadixSortPipeline::execute_init() {
        // First, initialize the range index buffer
        std::cerr << nInputElements << " input elements, initializing range indices." << std::endl;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(contextPtr->commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer for batch execution.");
        }

        initRangePipeline->setPushConstantsData(nInputElements, 0);

        mynydd::executeBatch(
            contextPtr,
            {initRangePipeline},
            false
        );

        std::cerr << "Retrieving data from buffer " << ioSortedIndicesB->getBuffer() << std::endl;
        auto init_retrieved = mynydd::fetchData<uint32_t>(
            contextPtr, ioSortedIndicesB, nInputElements
        );

        for (size_t i = 0; i < nInputElements && i < 10; ++i) {
            assert(init_retrieved[i] == i);
        }

    }

    void RadixSortPipeline::execute() {
        
        execute_init();

        for (size_t pass = 0; pass < nPasses; ++pass) {
            execute_pass(pass);

            auto ioIndices_retrieved = mynydd::fetchData<uint32_t>(
                contextPtr, getSortedIndicesBufferAtPass(pass), nInputElements
            );

            int zeros = 0;
            for (size_t i = 0; i < nInputElements; ++i) {
                if (ioIndices_retrieved[i] == 0) {
                    zeros++;
                }
                if (zeros > itemsPerGroup) {
                    assert(zeros <= itemsPerGroup);
                }
            }
        }
    }

    void RadixSortPipeline::execute_pass(size_t pass) {
        uint32_t bitOffset = pass * bitsPerPass;

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
        mynydd::executeBatch(
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