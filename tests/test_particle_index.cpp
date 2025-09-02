#include "mynydd/mynydd.tpp"
#include <cstdint>
#include <sys/types.h>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <vector>

#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include "test_morton_helpers.hpp"
#include "test_utils.hpp"

uint32_t pos2bin(float p, uint32_t nBits) {
    // repeat shader logic: uint(clamp(normPos, 0.0, 1.0) * float((1u << nbits) - 1u) + 0.5);
    float normPos = glm::clamp(p, 0.0f, 1.0f);
    float b = normPos * static_cast<float>((1u << nBits) - 1u) + 0.5f;
    return static_cast<uint32_t>(b);
}

TEST_CASE("Particle index works correctly", "[index]") {

    // Create a Vulkan context
    uint32_t nParticles = 4096 * 4;
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    auto inputBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(Particle), false);

    auto outputBufferTest = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(uint32_t), false);

    // Upload data
    std::vector<Particle> inputData(nParticles);
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : inputData) {
        v.position = glm::vec3(dist(rng), dist(rng), dist(rng));
    }

    mynydd::uploadData<Particle>(contextPtr, inputData, inputBuffer);

    mynydd::ParticleIndexPipeline<Particle> particleIndexPipeline(
        contextPtr,
        inputBuffer,
        4, // nBitsPerAxis
        256, // itemsPerGroup
        nParticles, // nDataPoints
        glm::vec3(0.0f), // domainMin
        glm::vec3(1.0f)  // domainMax
    );

    // Execute the pipeline
    particleIndexPipeline.execute();

    auto cellData = mynydd::fetchData<mynydd::CellInfo>(
        contextPtr, particleIndexPipeline.outputIndexBuffer, particleIndexPipeline.getNCells()
    );

    auto indexData = mynydd::fetchData<uint32_t>(
        contextPtr, particleIndexPipeline.radixSortPipeline.ioSortedIndicesB, nParticles
    );

    // TODO: this test is trivial, fix

    REQUIRE(true);
}