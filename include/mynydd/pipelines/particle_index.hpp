#ifndef PARTICLE_INDEX_HPP
#define PARTICLE_INDEX_HPP


#include <assert.h>
#include <cstring>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/radix_sort.hpp>

namespace mynydd {

    struct VulkanContext;

    struct MortonParams {
        uint32_t nBits;
        uint32_t nParticles;
        alignas(16) glm::vec3 domainMin; // alignas required for silly alignment issues
        alignas(16) glm::vec3 domainMax;
    };

    struct CellInfo {
        uint left;
        uint right;
    };

    struct IndexParams {
        uint32_t nKeys;
    };

    
    template<typename T>
    class ParticleIndexPipeline {
        public:
            ParticleIndexPipeline(
                std::shared_ptr<VulkanContext> contextPtr,
                std::shared_ptr<mynydd::Buffer> inputBuffer,
                uint32_t nBitsPerAxis,
                uint32_t itemsPerGroup, 
                uint32_t nDataPoints,
                glm::vec3 domainMin = glm::vec3(0.0f),
                glm::vec3 domainMax = glm::vec3(1.0f)
            ) : contextPtr(contextPtr),
                nBitsPerAxis(nBitsPerAxis),
                inputBuffer(inputBuffer),
                nDataPoints(nDataPoints),
                radixSortPipeline(contextPtr, itemsPerGroup, static_cast<uint32_t>(nDataPoints))
            {
                
                // assert (inputBuffer->getSize() == nDataPoints * sizeof(T) &&
                //     "Input buffer size must match number of data points times size of T");
                // TODO: FOR SOME REASON THE WORKGROUP SIZE IS 64 HERE; FIX

                mortonUniformBuffer = std::make_shared<mynydd::Buffer>(
                    contextPtr, sizeof(MortonParams), true);

                mortonStep = std::make_shared<mynydd::PipelineStep>(
                    contextPtr, "shaders/morton_u32_3d.comp.spv",
                    std::vector<std::shared_ptr<mynydd::Buffer>>{
                        inputBuffer, radixSortPipeline.ioBufferA, mortonUniformBuffer
                    },
                    (nDataPoints + 63) / 64
                );
                outputIndexBuffer = std::make_shared<mynydd::Buffer>(
                    contextPtr, getNCells() * sizeof(mynydd::CellInfo), false);

                indexUniformBuffer = std::make_shared<mynydd::Buffer>(
                        contextPtr, sizeof(IndexParams), true);

                sortedKeys2IndexStep = std::make_shared<mynydd::PipelineStep>(
                    contextPtr, "shaders/build_index_from_sorted_keys.comp.spv",
                    std::vector<std::shared_ptr<mynydd::Buffer>>{
                        (sizeof(uint32_t) / nBitsPerAxis) % 2 ? radixSortPipeline.ioBufferB : radixSortPipeline.ioBufferA, 
                        outputIndexBuffer, 
                        indexUniformBuffer
                    },
                    (nDataPoints + 63) / 64
                );

                std::cerr << "ParticleIndexPipeline created with " 
                          << nDataPoints << " data points." << std::endl;

            }
            ~ParticleIndexPipeline() {}; // member variables are RAII

            void execute() {
                MortonParams mortonParams{
                    nBitsPerAxis,
                    nDataPoints,
                    domainMin,
                    domainMax
                };
                IndexParams indexParams{
                    nDataPoints
                };

                mynydd::uploadUniformData<MortonParams>(contextPtr, mortonParams, mortonUniformBuffer);
                mynydd::uploadUniformData<IndexParams>(contextPtr, indexParams, indexUniformBuffer);

                mynydd::executeBatch(contextPtr, {mortonStep});

                radixSortPipeline.execute();
                // the index needs to be zeroed every time
                vkCmdFillBuffer(
                    contextPtr->commandBuffer,
                    outputIndexBuffer->getBuffer(),
                    0,
                    VK_WHOLE_SIZE,
                    0
                );
                mynydd::executeBatch(contextPtr, {sortedKeys2IndexStep});

            }

            uint32_t getNCells() const {
                return pow(2, 3 * nBitsPerAxis);
            }

            uint32_t itemsPerGroup = 256; // Hardcoded temporarily
            uint32_t nDataPoints;
            glm::vec3 domainMin = glm::vec3(0.0f);
            glm::vec3 domainMax = glm::vec3(1.0f);
            uint32_t nBitsPerAxis;

            // TODO: don't necessarily need this to be shared ptr
            std::shared_ptr<mynydd::Buffer> inputBuffer;
            std::shared_ptr<mynydd::Buffer> outputIndexBuffer;
            RadixSortPipeline radixSortPipeline;

        private:
            std::shared_ptr<mynydd::Buffer> mortonUniformBuffer;
            std::shared_ptr<mynydd::Buffer> indexUniformBuffer;
            std::shared_ptr<mynydd::Buffer> mortonOutputBuffer;
            std::shared_ptr<VulkanContext> contextPtr;
            std::shared_ptr<PipelineStep> mortonStep;
            std::shared_ptr<PipelineStep> sortedKeys2IndexStep;
            std::shared_ptr<mynydd::Buffer> radixUniform;
    };
}


#endif // PARTICLE_INDEX_HPP