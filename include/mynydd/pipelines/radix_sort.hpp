#pragma once

#include <assert.h>
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

            // TODO: getters
            uint32_t itemsPerGroup = 256; // Hardcoded temporarily
            uint32_t bitsPerPass = 8;
            uint32_t groupCount;
            uint32_t numBins;
            uint32_t nPasses;
            uint32_t nInputElements;

            // TODO: don't necessarily need this to be shared ptr
            std::shared_ptr<mynydd::Buffer> ioBufferA;
            std::shared_ptr<mynydd::Buffer> ioBufferB;
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

            std::shared_ptr<mynydd::PipelineStep<uint32_t>> histPipeline;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> histPipelinePong;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> sumPipeline;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> transposePipeline;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> workgroupPrefixPipeline;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> sortPipeline;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> sortPipelinePong;
            std::shared_ptr<mynydd::PipelineStep<uint32_t>> globalPrefixPipeline;


    };
}