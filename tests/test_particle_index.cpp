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

// #include "../src/pipelines/shaders/morton_kernels.comp.kern"
// #include <mynydd/shader_interop.hpp>

uint32_t pos2bin(double p, uint32_t nBits) {
    // repeat shader logic: uint(clamp(normPos, 0.0, 1.0) * double((1u << nbits) - 1u) + 0.5);
    double normPos = glm::clamp(p, 0.0, 1.0);
    double b = normPos * static_cast<double>((1u << nBits) - 1u) + 0.5;
    return static_cast<uint32_t>(b);
}

void particle_index_test(uint32_t nParticles) {
        std::cerr << "Running particle index test with " << nParticles << " particles." << std::endl;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    auto inputBuffer = 
        std::make_shared<mynydd::Buffer>(contextPtr, nParticles * sizeof(dVec3Aln32), false);

    // Upload data
    std::vector<dVec3Aln32> inputData(nParticles);
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : inputData) {
        v.data = glm::dvec3(dist(rng), dist(rng), dist(rng));
    }

    mynydd::uploadData<dVec3Aln32>(contextPtr, inputData, inputBuffer);

    uint nBits = 4;
    mynydd::ParticleIndexPipeline<dVec3Aln32> particleIndexPipeline(
        contextPtr,
        inputBuffer,
        nBits, // nBitsPerAxis
        256, // itemsPerGroup
        nParticles, // nDataPoints
        glm::dvec3(0.0), // domainMin
        glm::dvec3(1.0)  // domainMax
    );

    // Execute the pipeline
    particleIndexPipeline.execute();

    auto cellData = mynydd::fetchData<mynydd::CellInfo>(
        contextPtr, particleIndexPipeline.getOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()
    );
    auto flatCellData = mynydd::fetchData<mynydd::CellInfo>(
        contextPtr, particleIndexPipeline.getFlatOutputIndexCellRangeBuffer(), particleIndexPipeline.getNCells()
    );


    auto indexData = mynydd::fetchData<uint32_t>(
        contextPtr, particleIndexPipeline.getSortedIndicesBuffer(), nParticles
    );

    requireNotJustZeroes(indexData);

    REQUIRE(particleIndexPipeline.getNCells() == 16 * 16 * 16);

    uint32_t binsum = 0;
    for (uint32_t ak = 0; ak < particleIndexPipeline.getNCells(); ++ak) {

        uint nCellsPerAxis = (1 << particleIndexPipeline.nBitsPerAxis);

        uint i = ak / (nCellsPerAxis * nCellsPerAxis);        // z / depth index
        uint j = (ak / nCellsPerAxis) % nCellsPerAxis;        // y index
        uint k = ak % nCellsPerAxis;              // x index

        uint32_t ak_morton = morton3D(i, j, k);

        uvec3 demorton = decodeMorton3D(ak_morton, nBits);       
        REQUIRE(demorton.x == i);
        REQUIRE(demorton.y == j);
        REQUIRE(demorton.z == k); 

        uint32_t ak_flat = ijk2ak(demorton, nBits);

        //REQUIRE(ak_flat == ak);

        uint32_t start = cellData[ak_morton].left;
        uint32_t end = cellData[ak_morton].right;

        uint32_t flat_start = flatCellData[ak_flat].left;
        uint32_t flat_end = flatCellData[ak_flat].right;

        REQUIRE(start == flat_start);
        REQUIRE(end == flat_end);

        binsum += (end - start);

        if (start == end) {
            continue; // Empty cell
        }

        // All in a bin should have the same pak value
        int pak = -1;
        for (uint32_t pind = start; pind < end; ++pind) {
            auto particle = inputData[indexData[pind]];
;
            double pi = pos2bin(particle.data.x, particleIndexPipeline.nBitsPerAxis);
            double pj = pos2bin(particle.data.y, particleIndexPipeline.nBitsPerAxis);
            double pk = pos2bin(particle.data.z, particleIndexPipeline.nBitsPerAxis);

            REQUIRE(pi == i);
            REQUIRE(pj == j);
            REQUIRE(pk == k);
        }

    }

    particleIndexPipeline.debug_assert_bin_consistency();

    std::cerr << "Particle index test: total particles in bins: " << binsum << std::endl;
    REQUIRE(binsum == nParticles);
}

TEST_CASE("Particle index works correctly", "[index]") {
    uint32_t nParticles = 1 << 20;
    particle_index_test(nParticles);
}

TEST_CASE("Particle index works correctly with non pow 2 nParticles", "[index]") {
    uint32_t nParticles = 1000;
    particle_index_test(nParticles);
}