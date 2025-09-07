#include <chrono>
#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

#include "sph.hpp"


SPHData simulate_inputs(uint32_t nParticles) {
    // Generate some input data to start with
    std::vector<ParticlePosition> inputPos(nParticles);
    std::vector<float> inputDensities(nParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t ak = 0; ak < nParticles; ++ak) {
        inputPos[ak].position = glm::vec3(dist(rng), dist(rng), dist(rng));
        inputDensities[ak] = dist(rng);
    }
    return {inputDensities, inputPos, {}, {}};
}


SPHData run_sph_example(const SPHData& inputData) {

    auto nParticles = static_cast<uint32_t>(inputData.positions.size());
    std::cerr << "Testing particle index with " << nParticles << " particles" << std::endl;

    auto inputPos = inputData.positions;
    auto inputDensities = inputData.densities;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto pingPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(ParticlePosition), false);

    auto pongPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(ParticlePosition), false);

    auto pingDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    auto pongDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    mynydd::ParticleIndexPipeline<ParticlePosition> particleIndexPipeline(
        contextPtr,
        pingPosBuffer,
        4, // nBitsPerAxis
        256, // itemsPerGroup
        nParticles, // nDataPoints
        glm::vec3(0.0f), // domainMin
        glm::vec3(1.0f)  // domainMax
    );

    uint32_t groupCount = (nParticles + 256 - 1) / 256;
    
    auto scatterParticleData = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/scatter_particle_data.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pingDensityBuffer,
            pingPosBuffer,
            particleIndexPipeline.getSortedIndicesBuffer(),
            pongDensityBuffer,
            pongPosBuffer
        },
        groupCount
    );

    auto computeDensities = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/compute_density.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pongDensityBuffer,
            particleIndexPipeline.getSortedMortonKeysBuffer(),
            particleIndexPipeline.getSortedIndicesBuffer(),
            particleIndexPipeline.getOutputIndexCellRangeBuffer(),
            pingDensityBuffer
        },
        groupCount
    );

    mynydd::uploadData<ParticlePosition>(contextPtr, inputPos, pingPosBuffer);
    mynydd::uploadData<float>(contextPtr, inputDensities, pingDensityBuffer);


    auto t0 = std::chrono::high_resolution_clock::now();
    particleIndexPipeline.execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    mynydd::executeBatch(contextPtr, {scatterParticleData, computeDensities});
    auto t2 = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> elapsed1 = t1 - t0;
    std::cerr << "Particle indexing computation took " << elapsed1.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> elapsed2 = t2 - t1;
    std::cerr << "Density computation took " << elapsed2.count() << " ms" << std::endl;

    particleIndexPipeline.debug_assert_bin_consistency();

    return {
        mynydd::fetchData<float>(contextPtr, pingDensityBuffer, nParticles),
        mynydd::fetchData<ParticlePosition>(contextPtr, pingPosBuffer, nParticles),
        mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedIndicesBuffer(), nParticles),
        mynydd::fetchData<mynydd::CellInfo>(contextPtr, particleIndexPipeline.getOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells())
    };

 }