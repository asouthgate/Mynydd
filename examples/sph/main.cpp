#include <chrono>
#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct ParticlePosition {
    alignas(16) glm::vec3 position;
};

struct Params {
    float dt;
    float _pad[3];
};

int main(int argc, char** argv) {

    uint32_t nParticles = 4096 * 16;
    std::cerr << argc << std::endl;
    if (argc == 2) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
    } else if (argc > 2) {
        std::cerr << "Usage: nParticles" << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "Testing particle index with " << nParticles << " particles" << std::endl;

    // Generate some input data to start with
    std::vector<ParticlePosition> inputPos(nParticles);
    std::vector<float> inputDensities(nParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t ak = 0; ak < nParticles; ++ak) {
        inputPos[ak].position = glm::vec3(dist(rng), dist(rng), dist(rng));
        inputDensities[ak] = dist(rng);
    }

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto inputPosBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(ParticlePosition), false);

    auto inputDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    auto outputDensityBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(float), false);

    mynydd::ParticleIndexPipeline<ParticlePosition> particleIndexPipeline(
        contextPtr,
        inputPosBuffer,
        4, // nBitsPerAxis
        256, // itemsPerGroup
        nParticles, // nDataPoints
        glm::vec3(0.0f), // domainMin
        glm::vec3(1.0f)  // domainMax
    );

    uint32_t groupCount = (nParticles + 256 - 1) / 256;
    auto computeDensities = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/compute_density.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            inputDensityBuffer,
            particleIndexPipeline.getSortedMortonKeysBuffer(),
            particleIndexPipeline.getSortedIndicesBuffer(),
            particleIndexPipeline.getOutputIndexCellRangeBuffer(),
            outputDensityBuffer
        },
        groupCount
    );

    mynydd::uploadData<ParticlePosition>(contextPtr, inputPos, inputPosBuffer);
    mynydd::uploadData<float>(contextPtr, inputDensities, inputDensityBuffer);


    auto t0 = std::chrono::high_resolution_clock::now();
    particleIndexPipeline.execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    mynydd::executeBatch(contextPtr, {computeDensities});
    particleIndexPipeline.debug_assert_bin_consistency();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = t1 - t0;
    std::cerr << "Particle indexing computation took " << elapsed1.count() << " ms" << std::endl;
    std::chrono::duration<double, std::milli> elapsed2 = t2 - t1;
    std::cerr << "Density computation took " << elapsed2.count() << " ms" << std::endl;

    auto densities = mynydd::fetchData<float>(contextPtr, outputDensityBuffer, nParticles);
    // Check that densities are valid by iterating over the cell index, retrieving all particles in a bin, and manually computing density
    auto cellData = mynydd::fetchData<mynydd::CellInfo>(
        contextPtr, particleIndexPipeline.getOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()
    );
    auto indexData = mynydd::fetchData<uint32_t>(
        contextPtr, particleIndexPipeline.getSortedIndicesBuffer(), nParticles
    );

    size_t printed = 0;

    for (uint32_t morton_key = 0; morton_key < particleIndexPipeline.getNCells(); ++morton_key) {
        uint32_t start = cellData[morton_key].left;
        uint32_t end = cellData[morton_key].right;

        if (start == end) {
            continue; // Empty cell
        }

        float avg_dens = 0.0f;
        for (uint32_t pind = start; pind < end; ++pind) {
            uint32_t unsorted_ind = indexData[pind];
            auto particle = inputPos[unsorted_ind];

            if (printed < 10) {
                std::cerr << "Cell " << morton_key 
                    << " with start" << start
                    << " and end" << end
                    << " contains particle at original_index"
                    << pind << " -> " << unsorted_ind << " at position "
                    << particle.position.x << ", " 
                    << particle.position.y << ", " 
                    << particle.position.z << " with density: " 
                    << inputDensities[unsorted_ind] << std::endl;
                printed++;
            }

            avg_dens += inputDensities[unsorted_ind];
        }
        avg_dens /= float(end - start);
        if (fabs(avg_dens - densities[start]) < 1e-5) {
        } else {
            std::cerr << "Cell " << morton_key << " density check FAILED: computed " << avg_dens << " vs shader " << densities[start] << std::endl;
            throw std::runtime_error("Density check failed");
        }

    }


}