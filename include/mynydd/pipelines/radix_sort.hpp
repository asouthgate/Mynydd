#pragma once

#include <assert.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <mynydd/mynydd.hpp>


namespace mynydd {

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

    struct VulkanContext;
    
    class RadixSortPipeline {
        public:
            RadixSortPipeline(
                std::shared_ptr<VulkanContext> contextPtr, 
                uint32_t itemsPerGroup, 
                uint32_t totalSize
            );

            void execute();
            void execute_pass(size_t pass);
            void execute_init();
            std::shared_ptr<mynydd::Buffer> getSortedMortonKeysBuffer() {
                return (nPasses % 2 == 0) ? m_ioBufferA : m_ioBufferB;
            }
            std::shared_ptr<mynydd::Buffer> getSortedIndicesBuffer() {
                return (nPasses % 2 == 0) ? m_ioSortedIndicesB : m_ioSortedIndicesA;
            }

            // TODO: getters
            uint32_t itemsPerGroup = 256; // Hardcoded temporarily
            uint32_t bitsPerPass = 8;
            uint32_t groupCount;
            uint32_t numBins;
            uint32_t nPasses;
            uint32_t nInputElements;

            // TODO: don't necessarily need this to be shared ptr
            std::shared_ptr<mynydd::Buffer> m_ioBufferA;
            std::shared_ptr<mynydd::Buffer> m_ioBufferB;
            std::shared_ptr<mynydd::Buffer> m_ioSortedIndicesA;
            std::shared_ptr<mynydd::Buffer> m_ioSortedIndicesB;
            std::shared_ptr<mynydd::Buffer> perWorkgroupHistograms;
            std::shared_ptr<mynydd::Buffer> globalHistogram;
            std::shared_ptr<mynydd::Buffer> globalPrefixSum;
            std::shared_ptr<mynydd::Buffer> transposedHistograms;
            std::shared_ptr<mynydd::Buffer> workgroupPrefixSums;


            ~RadixSortPipeline() {}; // member variables are RAII
            
        private:
            std::shared_ptr<VulkanContext> contextPtr;

            std::shared_ptr<mynydd::Buffer> radixUniform;
            std::shared_ptr<mynydd::Buffer> sumUniform;
            std::shared_ptr<mynydd::Buffer> workgroupPrefixUniform;
            std::shared_ptr<mynydd::Buffer> globalPrefixUniform;
            std::shared_ptr<mynydd::Buffer> transposeUniform;
            std::shared_ptr<mynydd::Buffer> sortUniform;

            std::shared_ptr<mynydd::PipelineStep> initRangePipeline;
            std::shared_ptr<mynydd::PipelineStep> histPipeline;
            std::shared_ptr<mynydd::PipelineStep> histPipelinePong;
            std::shared_ptr<mynydd::PipelineStep> sumPipeline;
            std::shared_ptr<mynydd::PipelineStep> transposePipeline;
            std::shared_ptr<mynydd::PipelineStep> workgroupPrefixPipeline;
            std::shared_ptr<mynydd::PipelineStep> sortPipeline;
            std::shared_ptr<mynydd::PipelineStep> sortPipelinePong;
            std::shared_ptr<mynydd::PipelineStep> globalPrefixPipeline;


    };
}