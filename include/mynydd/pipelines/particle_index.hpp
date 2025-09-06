#ifndef PARTICLE_INDEX_HPP
#define PARTICLE_INDEX_HPP


#include <assert.h>
#include <cstdint>
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
                m_radixSortPipeline(contextPtr, itemsPerGroup, static_cast<uint32_t>(nDataPoints))
            {
                
                // assert (inputBuffer->getSize() == nDataPoints * sizeof(T) &&
                //     "Input buffer size must match number of data points times size of T");
                // TODO: FOR SOME REASON THE WORKGROUP SIZE IS 64 HERE; FIX

                mortonUniformBuffer = std::make_shared<mynydd::Buffer>(
                    contextPtr, sizeof(MortonParams), true);

                mortonStep = std::make_shared<mynydd::PipelineStep>(
                    contextPtr, "shaders/morton_u32_3d.comp.spv",
                    std::vector<std::shared_ptr<mynydd::Buffer>>{
                        inputBuffer, m_radixSortPipeline.m_ioBufferA, mortonUniformBuffer
                    },
                    (nDataPoints + 63) / 64
                );
                m_outputIndexCellRangeBuffer = std::make_shared<mynydd::Buffer>(
                    contextPtr, getNCells() * sizeof(mynydd::CellInfo), false);

                indexUniformBuffer = std::make_shared<mynydd::Buffer>(
                        contextPtr, sizeof(IndexParams), true);

                sortedKeys2IndexStep = std::make_shared<mynydd::PipelineStep>(
                    contextPtr, "shaders/build_index_from_sorted_keys.comp.spv",
                    std::vector<std::shared_ptr<mynydd::Buffer>>{
                        m_radixSortPipeline.getSortedMortonKeysBuffer(), 
                        m_outputIndexCellRangeBuffer, 
                        indexUniformBuffer
                    },
                    (nDataPoints + 63) / 64
                );

                std::cerr << "ParticleIndexPipeline created with " 
                          << nDataPoints << " data points." << std::endl;

            }
            ~ParticleIndexPipeline() {}; // member variables are RAII

            uint32_t pos2bin(float p, uint32_t nBits) {
                // repeat shader logic: uint(clamp(normPos, 0.0, 1.0) * float((1u << nbits) - 1u) + 0.5);
                float normPos = glm::clamp(p, 0.0f, 1.0f);
                float b = normPos * static_cast<float>((1u << nBits) - 1u) + 0.5f;
                return static_cast<uint32_t>(b);
            }

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

                m_radixSortPipeline.execute();
                // the index needs to be zeroed every time
                vkCmdFillBuffer(
                    contextPtr->commandBuffer,
                    m_outputIndexCellRangeBuffer->getBuffer(),
                    0,
                    VK_WHOLE_SIZE,
                    0
                );
                mynydd::executeBatch(contextPtr, {sortedKeys2IndexStep});

            }

            uint32_t getNCells() const {
                return pow(2, 3 * nBitsPerAxis);
            }

            void debug_assert_bin_consistency() {
                auto indexData = mynydd::fetchData<uint32_t>(
                    contextPtr, m_radixSortPipeline.getSortedIndicesBuffer(), nDataPoints
                );

                auto cellData = mynydd::fetchData<mynydd::CellInfo>(
                    contextPtr, m_outputIndexCellRangeBuffer, getNCells()
                );

                auto inputData = mynydd::fetchData<T>(
                    contextPtr, inputBuffer, nDataPoints);

                for (uint32_t ak = 0; ak < getNCells(); ++ak) {
                    uint32_t start = cellData[ak].left;
                    uint32_t end = cellData[ak].right;

                    if (start == end) {
                        continue; // Empty cell
                    }

                    std::vector<uint32_t> bini;
                    std::vector<uint32_t> binj;
                    std::vector<uint32_t> bink;
                    for (uint32_t pind = start; pind < end; ++pind) {
                        auto particle = inputData[indexData[pind]];
                        uint32_t pi = pos2bin(particle.position.x, nBitsPerAxis);
                        uint32_t pj = pos2bin(particle.position.y, nBitsPerAxis);
                        uint32_t pk = pos2bin(particle.position.z, nBitsPerAxis);

                        if (ak < 10) std::cerr << "Cell " << ak << " contains particle at " 
                                << particle.position.x << ", " 
                                << particle.position.y << ", " 
                                << particle.position.z << " mapped to bin " 
                                << pi << ", " << pj << ", " << pk << std::endl;

                        bini.push_back(pi);
                        binj.push_back(pj);
                        bink.push_back(pk);
                    }
                    // All in a bin should have the same pak value
                    for (auto pi : bini) assert(pi == bini[0]);
                    for (auto pj : binj) assert(pj == binj[0]); 
                    for (auto pk : bink) assert(pk == bink[0]);
                }
            }

            std::shared_ptr<mynydd::Buffer> getOutputIndexCellRangeBuffer() const {
                return m_outputIndexCellRangeBuffer;
            }

            std::shared_ptr<mynydd::Buffer> getSortedIndicesBuffer() {
                return m_radixSortPipeline.getSortedIndicesBuffer();
            }

            std::shared_ptr<mynydd::Buffer> getSortedMortonKeysBuffer() {
                return m_radixSortPipeline.getSortedMortonKeysBuffer();
            }

            uint32_t itemsPerGroup = 256; // Hardcoded temporarily
            uint32_t nDataPoints;
            glm::vec3 domainMin = glm::vec3(0.0f);
            glm::vec3 domainMax = glm::vec3(1.0f);
            uint32_t nBitsPerAxis;

            // TODO: don't necessarily need this to be shared ptr
            std::shared_ptr<mynydd::Buffer> inputBuffer;

        private:

            std::shared_ptr<mynydd::Buffer> m_outputIndexCellRangeBuffer;  // sized number of cells

            RadixSortPipeline m_radixSortPipeline;

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