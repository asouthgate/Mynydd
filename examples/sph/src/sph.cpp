#include <chrono>
#include <cstdint>
#include <H5Cpp.h>
#include <iomanip>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <filesystem> // C++17
namespace fs = std::filesystem;
#include <hdf5.h>
#include <filesystem>
#include <stdexcept>


#include <iostream>



#include "sph.hpp"
#include <algorithm>

SPHData simulate_inputs_uniform(uint32_t nParticles, double jitter) {
    std::vector<dVec3Aln32> inputPos(nParticles);
    std::vector<dVec3Aln32> inputVel(nParticles);
    std::vector<double> inputDensities(nParticles, 1.0);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-jitter, jitter);

    // Compute number of points along each axis
    uint32_t nPerAxis = std::ceil(std::cbrt(nParticles));
    double spacing = 1.0 / nPerAxis;

    for (uint32_t idx = 0; idx < nParticles; ++idx) {
        uint32_t ix = idx % nPerAxis;
        uint32_t iy = (idx / nPerAxis) % nPerAxis;
        uint32_t iz = idx / (nPerAxis * nPerAxis);

        // Base position
        double x = ix * spacing;
        double y = iy * spacing;
        double z = iz * spacing;

        // Add jitter
        x += dist(rng);
        y += dist(rng);
        z += dist(rng);

        // Clamp to [0,1]
        x = std::clamp(x, 0.0, 1.0);
        y = std::clamp(y, 0.0, 1.0);
        z = std::clamp(z, 0.0, 1.0);

        inputPos[idx].data = glm::dvec3(x, y, z);
        inputVel[idx].data = glm::dvec3(0.0);
    }

    return {inputDensities, {}, {}, inputPos, inputVel, {}};
}


SPHData simulate_inputs(uint32_t nParticles, double min, double max) {
    // Generate some input data to start with
    std::vector<dVec3Aln32> inputPos(nParticles);
    std::vector<dVec3Aln32> inputVel(nParticles);
    std::vector<double> inputDensities(nParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(min, max);
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

void write_dvec3_to_hdf5(const std::vector<dVec3Aln32>& pos,
                         const std::vector<uint32_t>& morton_keys,
                          const std::string& basepath,
                          uint64_t iter)
{

    std::filesystem::path dir(basepath);
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
        std::cout << "Created directory: " << dir << "\n";
    }

    const hsize_t n = pos.size();
    const hsize_t dims[2] = { n, 3 };
    std::vector<double> buf(n*3);
    for (size_t i=0;i<n;++i){
        buf[i*3+0] = pos[i].data[0];
        buf[i*3+1] = pos[i].data[1];
        buf[i*3+2] = pos[i].data[2];
    }

    const hsize_t dims_keys[1] = { n };

    std::string tmp = basepath + "/" + basepath + ".tmp." + std::to_string(iter) + ".h5";
    std::string finalname = basepath + "/" + basepath + "." + std::to_string(iter) + ".h5";

    {
        H5::H5File file(tmp, H5F_ACC_TRUNC);
        H5::DataSpace space(2, dims);
        H5::DataSet ds = file.createDataSet("positions", H5::PredType::NATIVE_DOUBLE, space);
        ds.write(buf.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSpace space_keys(1, dims_keys);
        H5::DataSet ds_keys = file.createDataSet("morton_keys", H5::PredType::NATIVE_UINT32, space_keys);
        ds_keys.write(morton_keys.data(), H5::PredType::NATIVE_UINT32);

        
        file.flush(H5F_SCOPE_GLOBAL); // close/flush

        
    }

    std::filesystem::rename(tmp, finalname); // atomic on same FS
}


std::string _get_hd5_filename() {
    // Get current time as system_clock::time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t (calendar time)
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);  // thread-safe on Windows
#else
    localtime_r(&now_time, &local_tm);  // thread-safe on Linux/macOS
#endif

    // Format timestamp
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "sph_%Y%m%d_%H%M%S");

    // Generate 4-character random alphanumeric tag
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::mt19937 rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<> dist(0, sizeof(charset) - 2);

    std::string tag;
    for (int i = 0; i < 4; ++i) {
        tag += charset[dist(rng)];
    }

    oss << "_" << tag;

    return oss.str();
}

void _debug_print_state(std::vector<dVec3Aln32> vel, std::vector<dVec3Aln32> pos, std::vector<double> densities, const SPHParams& params, uint iteration) {
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

    double avg_density = 0.0;
    for (size_t k = 0; k < densities.size(); ++k) {
        avg_density += densities[k];
    }
    avg_density /= (double) densities.size();

    std::cerr << "it=" << iteration << 
            ", v_avg=" << "(" << avg.x << " " << avg.y << " " << avg.z << ")" <<
            ", x_avg=" << "(" << avg_pos.x << " " << avg_pos.y << " " << avg_pos.z << ")" <<
            ", rho_avg=" << avg_density <<
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

// Unit cube triangles (6 faces × 2 triangles per face)
std::vector<glm::dvec3> BOUNDARY_VERTICES = {
    // -Z face
    glm::dvec3(0,0,0), glm::dvec3(1,0,0), glm::dvec3(1,1,0),
    glm::dvec3(0,0,0), glm::dvec3(1,1,0), glm::dvec3(0,1,0),

    // +Z face
    glm::dvec3(0,0,1), glm::dvec3(1,0,1), glm::dvec3(1,1,1),
    glm::dvec3(0,0,1), glm::dvec3(1,1,1), glm::dvec3(0,1,1),

    // -X face
    glm::dvec3(0,0,0), glm::dvec3(0,0,1), glm::dvec3(0,1,1),
    glm::dvec3(0,0,0), glm::dvec3(0,1,1), glm::dvec3(0,1,0),

    // +X face
    glm::dvec3(1,0,0), glm::dvec3(1,0,1), glm::dvec3(1,1,1),
    glm::dvec3(1,0,0), glm::dvec3(1,1,1), glm::dvec3(1,1,0),

    // -Y face
    glm::dvec3(0,0,0), glm::dvec3(1,0,0), glm::dvec3(1,0,1),
    glm::dvec3(0,0,0), glm::dvec3(1,0,1), glm::dvec3(0,0,1),

    // +Y face
    glm::dvec3(0,1,0), glm::dvec3(1,1,0), glm::dvec3(1,1,1),
    glm::dvec3(0,1,0), glm::dvec3(1,1,1), glm::dvec3(0,1,1)
};


SPHData run_sph_example(const SPHData& inputData, SPHParams& params, uint iterations, std::string fname, bool debug_mode) {

    std::cerr << "Beginning simulation with params " <<
        " nBits=" << params.nBits <<
        " nParticles=" << params.nParticles <<
        " dist=" << params.dist <<
        " dt=" << params.dt <<
        " h=" << params.h <<
        " mass=" << params.mass <<
        " gravity=(" << params.gravity.x << "," << params.gravity.y << "," << params.gravity.z << ")" <<
        " rho0=" << params.rho0 <<
        " c2=" << params.c2 <<
        std::endl;

    std::cerr << "Expected nmber of nbrs is" << (4.0/3.0)*M_PI*params.h*params.h*params.h*double(params.nParticles) * params.rho0 / double(params.nParticles) << std::endl;

    auto nParticles = static_cast<uint32_t>(inputData.positions.size());
    std::cerr << "Testing particle index with " << nParticles << " particles" << std::endl;
    if (fname == "") {
        fname = _get_hd5_filename();
    }
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

    std::vector<dVec3Aln32> vertices(BOUNDARY_VERTICES.size());
    for (size_t i = 0; i < BOUNDARY_VERTICES.size(); ++i) {
        vertices[i].data = BOUNDARY_VERTICES[i];
    }
    auto meshVerticesBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, vertices.size() * sizeof(dVec3Aln32), false);


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
            meshVerticesBuffer
        },
        groupCount,
        1,
        1,
        std::vector<uint32_t>{sizeof(SPHParams)}
    );

    mynydd::uploadData<dVec3Aln32>(contextPtr, inputPos, pingPosBuffer);
    mynydd::uploadData<dVec3Aln32>(contextPtr, inputVel, pingVelocityBuffer);
    mynydd::uploadData<double>(contextPtr, inputDensities, pingDensityBuffer);
    
    mynydd::uploadData<dVec3Aln32>(contextPtr, vertices, meshVerticesBuffer);

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

    bool write_hdf5 = true;
    uint hdf5_cadence = 10; // write every n iterations

    for (uint it = 0; it < iterations; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        particleIndexPipeline.execute();
        auto t1 = std::chrono::high_resolution_clock::now();
        mynydd::executeBatch(contextPtr, {scatterParticleData, computeDensities});
        auto t2 = std::chrono::high_resolution_clock::now();

        if (debug_mode) {
            // std::cerr << "Validating after density, indexing iteration " << it << ":" << std::endl;
            particleIndexPipeline.debug_assert_bin_consistency();
            _validate_velocities_in_bounds(mynydd::fetchData<dVec3Aln32>(contextPtr, pongVelocityBuffer, nParticles), params);
            _validate_positions_in_bounds(mynydd::fetchData<dVec3Aln32>(contextPtr, pongPosBuffer, nParticles), params);
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        mynydd::executeBatch(contextPtr, {leapFrogStep});
        auto t4 = std::chrono::high_resolution_clock::now();

        // Check that no positions are outside of the domain
        if (debug_mode) {
            // std::cerr << "Validating after leapfrog, indexing iteration " << it << ":" << std::endl;

            auto velocities = mynydd::fetchData<dVec3Aln32>(contextPtr, pingVelocityBuffer, nParticles);
            auto positions = mynydd::fetchData<dVec3Aln32>(contextPtr, pingPosBuffer, nParticles);
            auto densities = mynydd::fetchData<double>(contextPtr, pingDensityBuffer, nParticles);
            
            // now report average positions and velocities
            _debug_print_state(velocities, positions, densities, params, it);

            _validate_velocities_in_bounds(velocities, params);
            _validate_positions_in_bounds(positions, params);

        }

        if (write_hdf5 && (it % hdf5_cadence == 0 || it == iterations - 1)) {
            write_dvec3_to_hdf5(
                mynydd::fetchData<dVec3Aln32>(contextPtr, pingPosBuffer, nParticles), 
                mynydd::fetchData<uint32_t>(contextPtr, particleIndexPipeline.getSortedMortonKeysBuffer(), nParticles),
                fname, it
            );
        }

        std::chrono::duration<double, std::milli> elapsed1 = t1 - t0;
        std::chrono::duration<double, std::milli> elapsed2 = t2 - t1;
        std::chrono::duration<double, std::milli> elapsed3 = t4 - t3;

        index_step_times.push_back(elapsed1.count());
        density_times.push_back(elapsed2.count());
        leapfrog_times.push_back(elapsed3.count());
        std::cout << "\r" << it << ": index=" << elapsed1.count() << "ms density=" << elapsed2.count() << "ms leapfrog=" << elapsed3.count() << "ms" << std::flush;
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