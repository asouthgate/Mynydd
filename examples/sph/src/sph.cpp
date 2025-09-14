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
    std::vector<Vec3Aln16> inputPos(nParticles);
    std::vector<float> inputDensities(nParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t ak = 0; ak < nParticles; ++ak) {
        inputPos[ak].position = glm::vec3(dist(rng), dist(rng), dist(rng));
        inputDensities[ak] = dist(rng);
    }
    return {inputDensities, inputPos, {}, {}, {}};
}


SPHData run_sph_example(const SPHData& inputData, uint32_t nBitsPerAxis, int dist) {

    auto nParticles = static_cast<uint32_t>(inputData.positions.size());
    std::cerr << "Testing particle index with " << nParticles << " particles" << std::endl;

    auto inputPos = inputData.positions;
    auto inputDensities = inputData.densities;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto pingPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(Vec3Aln16), false);

    auto pongPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(Vec3Aln16), false);

    auto pingDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    auto pongDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    auto pressureBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    auto pressureForceBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(Vec3Aln16), false);


    mynydd::ParticleIndexPipeline<Vec3Aln16> particleIndexPipeline(
        contextPtr,
        pingPosBuffer,
        nBitsPerAxis, // nBitsPerAxis
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
        "examples/sph/compute_particle_state_1.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pongDensityBuffer,
            pongPosBuffer,
            particleIndexPipeline.getSortedMortonKeysBuffer(), // TODO: dont need anymore
            particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(),
            particleIndexPipeline.getOutputIndexCellRangeBuffer(), // DONT NEED ANYMORE
            pingDensityBuffer,
            pressureBuffer
        },
        groupCount,
        1,
        1,
        std::vector<uint32_t>{sizeof(DensityParams)}
    );

    auto computeForces = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/compute_particle_state_2.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pingDensityBuffer,
            pongPosBuffer,
            pressureBuffer,
            particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(),
            pressureForceBuffer
        },
        groupCount,
        1,
        1,
        std::vector<uint32_t>{sizeof(DensityParams)}
    );


    mynydd::uploadData<Vec3Aln16>(contextPtr, inputPos, pingPosBuffer);
    mynydd::uploadData<float>(contextPtr, inputDensities, pingDensityBuffer);

    DensityParams gridParams = {
        nBitsPerAxis,
        nParticles,
        glm::vec3(0.0f),
        glm::vec3(1.0f),
        dist
    };

    computeDensities->setPushConstantsData(gridParams, 0);

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
        mynydd::fetchData<Vec3Aln16>(contextPtr, pongPosBuffer, nParticles),
        mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedMortonKeysBuffer(), nParticles),
        mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedIndicesBuffer(), nParticles),
        mynydd::fetchData<mynydd::CellInfo>(contextPtr, particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()),
        gridParams
    };

 }