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
        inputDensities[ak] = 1.0;
        inputVel[ak].data = glm::dvec3(0.0, 0.0, 0.0);
    }
    return {inputDensities, {}, {}, inputPos,  inputVel, {}};
}

void _validate_positions_in_bounds(std::vector<dVec3Aln32> posData, const SPHParams& params) {
    for (const auto& p : posData) {
        if (p.data.x < params.domainMin.x || p.data.x > params.domainMax.x ||
            p.data.y < params.domainMin.y || p.data.y > params.domainMax.y ||
            p.data.z < params.domainMin.z || p.data.z > params.domainMax.z) {
            throw std::runtime_error("Particle position out of bounds");
        }
    }
}

void _debug_print_state(std::vector<dVec3Aln32> vel, std::vector<dVec3Aln32> pos, const SPHParams& params, uint iteration) {
    double kinetic_energy = 0.0;
    glm::dvec3 avg(0.0);
    for (size_t k = 0; k < vel.size(); ++k) {
        avg += vel[k].data;
        double vmag = glm::length(vel[k].data);
        kinetic_energy += 0.5 * params.mass * vmag * vmag;
    }
    avg /= (double) vel.size();

    glm::dvec3 avg_pos(0.0);
    for (size_t k = 0; k < pos.size(); ++k) {
        avg_pos += pos[k].data;
    }
    avg_pos /= (double) pos.size();

    std::cerr << "it=" << iteration << 
            ", v_avg=" << "(" << avg.x << " " << avg.y << " " << avg.z << ")" <<
            ", x_avg=" << "(" << avg_pos.x << " " << avg_pos.y << " " << avg_pos.z << ")" <<
            ", kinetic_energy=" << kinetic_energy << 
            ", v0=(" << vel[0].data.x << " " << vel[0].data.y << " " << vel[0].data.z << ")" <<
            ", x0=(" << pos[0].data.x << " " << pos[0].data.y << " " << pos[0].data.z << ")" << 
            std::endl;
}

void _validate_velocities_in_bounds(std::vector<dVec3Aln32> velData, const SPHParams& params) {
    double maxv = (1 << params.nBits) / params.dt;
    for (const auto& p : velData) {
        if (p.data.x > maxv ||
            p.data.y > maxv ||
            p.data.z > maxv ) {
            std::cerr << "Error: particle velocity OOB " << p.data.x << " " << p.data.y << " " << p.data.z << std::endl;
            throw std::runtime_error("Particle velocity out of bounds");
        }
    }
}

SPHData run_sph_example(const SPHData& inputData, SPHParams& params, uint iterations) {

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
        params.nBits, // nBitsPerAxis
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
        std::vector<uint32_t>{sizeof(SPHParams)}
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
        std::vector<uint32_t>{sizeof(SPHParams)}
    );

    mynydd::uploadData<dVec3Aln32>(contextPtr, inputPos, pingPosBuffer);
    mynydd::uploadData<dVec3Aln32>(contextPtr, inputVel, pingVelocityBuffer);
    mynydd::uploadData<double>(contextPtr, inputDensities, pingDensityBuffer);

    double h;
    if (params.dist == 0) {
        std::cerr << "Using d = 0 (same cell only) for SPH search" << std::endl;
        h = 0.5 / double(1 << params.nBits);
    } else if (params.dist == 1) {
        std::cerr << "Using d = 1 (neighbouring cells) for SPH search" << std::endl;
        // h should be 1 because otherwise ball can fall outside of searched area (points are not always in middle of cells)
        h = 1.0 / double(1 << params.nBits);
    } else {
        throw std::runtime_error("Only index_search_dist of 0 or 1 supported");
    }

    computeDensities->setPushConstantsData(params, 0);
    leapFrogStep->setPushConstantsData(params, 0);

    std::vector<double> index_step_times;
    std::vector<double> density_times;
    std::vector<double> leapfrog_times;

    bool debug_enabled = true;

    for (uint it = 0; it < iterations; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        particleIndexPipeline.execute();
        auto t1 = std::chrono::high_resolution_clock::now();
        mynydd::executeBatch(contextPtr, {scatterParticleData, computeDensities});
        auto t2 = std::chrono::high_resolution_clock::now();

        if (debug_enabled) {
            // std::cerr << "Validating after density, indexing iteration " << it << ":" << std::endl;
            particleIndexPipeline.debug_assert_bin_consistency();
            _validate_velocities_in_bounds(mynydd::fetchData<dVec3Aln32>(contextPtr, pongVelocityBuffer, nParticles), params);
            _validate_positions_in_bounds(mynydd::fetchData<dVec3Aln32>(contextPtr, pongPosBuffer, nParticles), params);
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        mynydd::executeBatch(contextPtr, {leapFrogStep});
        auto t4 = std::chrono::high_resolution_clock::now();

        // Check that no positions are outside of the domain
        if (debug_enabled) {
            // std::cerr << "Validating after leapfrog, indexing iteration " << it << ":" << std::endl;

            auto velocities = mynydd::fetchData<dVec3Aln32>(contextPtr, pingVelocityBuffer, nParticles);
            auto positions = mynydd::fetchData<dVec3Aln32>(contextPtr, pingPosBuffer, nParticles);
            _validate_velocities_in_bounds(velocities, params);
            _validate_positions_in_bounds(positions, params);

            // now report average positions and velocities
            _debug_print_state(velocities, positions, params, it);
        }

        std::chrono::duration<double, std::milli> elapsed1 = t1 - t0;
        std::chrono::duration<double, std::milli> elapsed2 = t2 - t1;
        std::chrono::duration<double, std::milli> elapsed3 = t4 - t3;

        index_step_times.push_back(elapsed1.count());
        density_times.push_back(elapsed2.count());
        leapfrog_times.push_back(elapsed3.count());
    }

    double index_time_avg = std::accumulate(index_step_times.begin(), index_step_times.end(), 0.0) / index_step_times.size();
    double density_time_avg = std::accumulate(density_times.begin(), density_times.end(), 0.0) / density_times.size();
    double leapfrog_time_avg = std::accumulate(leapfrog_times.begin(), leapfrog_times.end(), 0.0) / leapfrog_times.size();

    std:: cerr << "Average particle index time over " << iterations << " iterations: " << index_time_avg << " ms" << std::endl;
    std:: cerr << "Average density computation time over " << iterations << " iterations: " << density_time_avg << " ms" << std::endl;
    std:: cerr << "Average leapfrog time over " << iterations << " iterations: " << leapfrog_time_avg << " ms" << std::endl;

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
        mynydd::fetchData<dVec3Aln32>(contextPtr, pingVelocityBuffer, nParticles)
    };

 }