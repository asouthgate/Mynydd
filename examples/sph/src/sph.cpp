#include <chrono>
#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "sph.hpp"


SPHData simulate_inputs(uint32_t nParticles) {
    // Generate some input data to start with
    std::vector<dVec3Aln32> inputPos(nParticles);
    std::vector<dVec3Aln32> inputVel(nParticles);
    std::vector<double> inputDensities(nParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t ak = 0; ak < nParticles; ++ak) {
        inputPos[ak].data = glm::dvec3(dist(rng), dist(rng), dist(rng));
        inputDensities[ak] = dist(rng);
        inputVel[ak].data = glm::dvec3(0.0, 0.0, 0.0);
    }
    return {inputDensities, {}, {}, inputPos,  inputVel, {}};
}


SPHData run_sph_example(const SPHData& inputData, uint32_t nBitsPerAxis, int index_search_dist, double dt) {

    auto nParticles = static_cast<uint32_t>(inputData.positions.size());
    std::cerr << "Testing particle index with " << nParticles << " particles" << std::endl;

    auto inputPos = inputData.positions;
    auto inputVel = inputData.velocities;
    auto inputDensities = inputData.densities;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    // 2 Buffers are required: x_n and x_n+1
    // TODO: figure out whether vec3 or dvec3 for positions
    auto pingPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);
    auto pongPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);

    // 2 Buffers are required: v_n-1/2 and v_n+1/2
    auto pingVelocityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);
    auto pongVelocityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);

    // 2 buffers are only required only for memory safety, not for computing 1 step at a time.
    auto pingDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(double), false);
    auto pongDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(double), false);

    // These buffers are not required, other than for debugging.
    auto pressureBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(double), false);
    auto pressureForceBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);

    mynydd::ParticleIndexPipeline<dVec3Aln32> particleIndexPipeline(
        contextPtr,
        pingPosBuffer,
        nBitsPerAxis, // nBitsPerAxis
        256, // itemsPerGroup
        nParticles, // nDataPoints
        glm::dvec3(0.0), // domainMin
        glm::dvec3(1.0)  // domainMax
    );

    uint32_t groupCount = (nParticles + 256 - 1) / 256;
    
    auto scatterParticleData = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/scatter_particle_data.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pingDensityBuffer,
            pingPosBuffer,
            pingVelocityBuffer,
            particleIndexPipeline.getSortedIndicesBuffer(),
            pongDensityBuffer,
            pongPosBuffer,
            pongVelocityBuffer
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
        std::vector<uint32_t>{sizeof(Step2Params)}
    );

    auto leapFrogStep = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/compute_particle_state_2.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            pingDensityBuffer,
            pongPosBuffer,
            pongVelocityBuffer,
            pressureBuffer,
            particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(),
            pressureForceBuffer,
            pingPosBuffer,
            pingVelocityBuffer,
        },
        groupCount,
        1,
        1,
        std::vector<uint32_t>{sizeof(Step2Params)}
    );

    mynydd::uploadData<dVec3Aln32>(contextPtr, inputPos, pingPosBuffer);
    mynydd::uploadData<dVec3Aln32>(contextPtr, inputVel, pingVelocityBuffer);
    mynydd::uploadData<double>(contextPtr, inputDensities, pingDensityBuffer);

    double h;
    if (index_search_dist == 0) {
        std::cerr << "Using d = 0 (same cell only) for SPH search" << std::endl;
        h = 0.5 / double(1 << nBitsPerAxis);
    } else if (index_search_dist == 1) {
        std::cerr << "Using d = 1 (neighbouring cells) for SPH search" << std::endl;
        // h should be 1 because otherwise ball can fall outside of searched area (points are not always in middle of cells)
        h = 1.0 / double(1 << nBitsPerAxis);
    } else {
        throw std::runtime_error("Only index_search_dist of 0 or 1 supported");
    }

    // normalize mass so that density is approx 1.0
    double mass = 1.0 / nParticles;

    Step2Params step2Params = {
        nBitsPerAxis,
        nParticles,
        glm::dvec3(0.0),
        glm::dvec3(1.0),
        index_search_dist,
        dt,
        h,
        mass
    };

    computeDensities->setPushConstantsData(step2Params, 0);
    leapFrogStep->setPushConstantsData(step2Params, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    particleIndexPipeline.execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    mynydd::executeBatch(contextPtr, {scatterParticleData, computeDensities});
    auto t2 = std::chrono::high_resolution_clock::now();
    particleIndexPipeline.debug_assert_bin_consistency();
    auto t3 = std::chrono::high_resolution_clock::now();
    mynydd::executeBatch(contextPtr, {leapFrogStep});
    auto t4 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed1 = t1 - t0;
    std::cerr << "Particle indexing computation took " << elapsed1.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> elapsed2 = t2 - t1;
    std::cerr << "Density computation took " << elapsed2.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> elapsed3 = t4 - t3;
    std::cerr << "Leapfrog took " << elapsed3.count() << " ms" << std::endl;

    return {
        mynydd::fetchData<double>(contextPtr, pingDensityBuffer, nParticles),
        mynydd::fetchData<double>(contextPtr, pressureBuffer, nParticles),
        mynydd::fetchData<dVec3Aln32>(contextPtr, pressureForceBuffer, nParticles),
        mynydd::fetchData<dVec3Aln32>(contextPtr, pongPosBuffer, nParticles),
        mynydd::fetchData<dVec3Aln32>(contextPtr, pongVelocityBuffer, nParticles),
        mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedMortonKeysBuffer(), nParticles),
        mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedIndicesBuffer(), nParticles),
        mynydd::fetchData<mynydd::CellInfo>(contextPtr, particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()),
        mynydd::fetchData<dVec3Aln32>(contextPtr, pingPosBuffer, nParticles),
        mynydd::fetchData<dVec3Aln32>(contextPtr, pingVelocityBuffer, nParticles),
        step2Params
    };

 }