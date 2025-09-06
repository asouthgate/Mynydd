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


    auto computeDensities = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "examples/sph/compute_density.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            inputDensityBuffer,
            particleIndexPipeline.radixSortPipeline.getSortedMortonKeysBuffer(),
            particleIndexPipeline.radixSortPipeline.getSortedIndicesBuffer(),
            particleIndexPipeline.getOutputIndexCellRangeBuffer(),
            outputDensityBuffer
        },
        256
    );

    mynydd::uploadData<ParticlePosition>(contextPtr, inputPos, inputPosBuffer);
    mynydd::uploadData<float>(contextPtr, inputDensities, inputDensityBuffer);

    particleIndexPipeline.execute();

    mynydd::executeBatch(contextPtr, {computeDensities});
    particleIndexPipeline.debug_assert_bin_consistency();

    auto densities = mynydd::fetchData<float>(contextPtr, outputDensityBuffer, nParticles);
    // Check that densities are valid by iterating over the cell index, retrieving all particles in a bin, and manually computing density
    auto cellData = mynydd::fetchData<mynydd::CellInfo>(
        contextPtr, particleIndexPipeline.getOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()
    );
    auto indexData = mynydd::fetchData<uint32_t>(
        contextPtr, particleIndexPipeline.radixSortPipeline.getSortedIndicesBuffer(), nParticles
    );

    for (uint32_t morton_key = 0; morton_key < particleIndexPipeline.getNCells(); ++morton_key) {
        uint32_t start = cellData[morton_key].left;
        uint32_t end = cellData[morton_key].right;

        if (start == end) {
            continue; // Empty cell
        }

        float avg_dens = 0.0f;
        std::cerr << "Cell ak: " << morton_key << " raw range: " << start << " " << end << std::endl;
        for (uint32_t pind = start; pind < end; ++pind) {
            uint32_t unsorted_ind = indexData[pind];
            auto particle = inputPos[unsorted_ind];

            if (morton_key < 20) std::cerr << "Cell " << morton_key 
                    << " with start" << start
                    << " and end" << end
                    << " contains particle at original_index"
                    << pind << " -> " << unsorted_ind << " at position "
                    << particle.position.x << ", " 
                    << particle.position.y << ", " 
                    << particle.position.z << " with density: " 
                    << inputDensities[unsorted_ind] << std::endl;

            avg_dens += inputDensities[unsorted_ind];
        }
        avg_dens /= float(end - start);
        std::cerr << "Cell " << morton_key << " average density: " << avg_dens <<  "computed vs " 
            << densities[start] << " from compute shader" << std::endl;
        assert(avg_dens == densities[start]);

    }


}