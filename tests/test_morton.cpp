#include <cstddef>
#include <cstdint>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <mynydd/mynydd.hpp>

#include <mynydd/shader_interop.hpp>
#include "shaders/morton_kernels.comp.kern"



struct Particle {
    glm::vec3 position;
    uint32_t key;
};

struct KeyRange {
    uint32_t keyMin;
    uint32_t keyMax;
};

// Naive function to compute key ranges for particles within dmax
std::vector<KeyRange> computeKeyRanges(const std::vector<Particle>& particles, float dmax) {
    size_t n = particles.size();
    std::vector<KeyRange> ranges(n);

    for (size_t i = 0; i < n; ++i) {
        uint32_t keyMin = std::numeric_limits<uint32_t>::max();
        uint32_t keyMax = 0;

        const glm::vec3& posI = particles[i].position;

        for (size_t j = 0; j < n; ++j) {
            const glm::vec3& posJ = particles[j].position;
            float dist = glm::distance(posI, posJ);
            if (dist <= dmax) {
                keyMin = std::min(keyMin, particles[j].key);
                keyMax = std::max(keyMax, particles[j].key);
            }
        }

        ranges[i] = { keyMin, keyMax };
    }

    return ranges;
}

TEST_CASE("Morton kernels produce expected results in 2D", "[morton]") {
    // Test some known values (plotted these out in a lesser programming language)
    REQUIRE(morton2D(0, 14) == 168);
    REQUIRE(morton2D(14, 15) == 254);
    REQUIRE(morton2D(8, 6) == 104);
    REQUIRE(morton2D(5, 3) == 27);

    REQUIRE(morton2D(0u, 14u) == 168u);
    REQUIRE(morton2D(14u, 15u) == 254u);
    REQUIRE(morton2D(8u, 6u) == 104u);
    REQUIRE(morton2D(5u, 3u) == 27u);

}


TEST_CASE("Morton kernels produce expected results in 3D", "[morton]") {
    // Test some known values again
    REQUIRE(morton3D(0, 6, 7) == 436);
    REQUIRE(morton3D(6, 6, 7) == 508);
    REQUIRE(morton3D(7, 0, 6) == 361);
    REQUIRE(morton3D(0, 1, 1) == 6);

    REQUIRE(morton3D(0u, 6u, 7u) == 436u);
    REQUIRE(morton3D(6u, 6u, 7u) == 508u);
    REQUIRE(morton3D(7u, 0u, 6u) == 361u);
    REQUIRE(morton3D(0u, 1u, 1u) == 6u);

    REQUIRE(morton3D_loop(0u, 6u, 7u, 3u) == 436u);
    REQUIRE(morton3D_loop(6u, 6u, 7u, 3u) == 508u);
    REQUIRE(morton3D_loop(7u, 0u, 6u, 3u) == 361u);
    REQUIRE(morton3D_loop(0u, 1u, 1u, 3u) == 6u);

}


TEST_CASE("Binning works as expected for Morton curves", "[morton]") {
    // Test some known values again
    uint32_t nbits = 3;
    float binsize = 1.0 / pow(2, nbits);
    float eps = 0.000001;
    REQUIRE(binPosition(0.0, nbits) == 0);
    REQUIRE(binPosition(eps, nbits) == 0);
    REQUIRE(binPosition(binsize + eps, nbits) == 1);
    REQUIRE(binPosition(0.49, nbits) == 3);
    REQUIRE(binPosition(0.51, nbits) == 4);
    REQUIRE(binPosition(1.0, nbits) == 7);

}


TEST_CASE("3D Morton shader produces unique, monotone keys", "[morton]") {
    const uint32_t nBits = 4; // up to 1024 per axis
    const uint32_t nPerDim = pow(2, nBits); // 32 particles per axis
    const uint32_t nParticles = pow(nPerDim, 3);

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();


    auto inputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, nParticles * sizeof(Particle), false);
    auto outputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, nParticles * sizeof(Particle), true);

    struct Params {
        uint32_t nBits;
        uint32_t nParticles;
        alignas(16) glm::vec3 domainMin; // alignas required for silly alignment issues
        alignas(16) glm::vec3 domainMax;
    } params{nBits, nParticles, glm::vec3(0.0f), glm::vec3(float(nPerDim - 1))};

    auto uniformBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, sizeof(Params), true);

    // generate deterministic particle positions
    std::vector<Particle> particles(nParticles);
    uint32_t idx = 0;
    for (uint32_t z = 0; z < nPerDim; ++z) {
        for (uint32_t y = 0; y < nPerDim; ++y) {
            for (uint32_t x = 0; x < nPerDim; ++x) {
                particles[idx].position = glm::vec3(float(x), float(y), float(z));
                particles[idx].key = 0;
                ++idx;
            }
        }
    }

    mynydd::uploadData<Particle>(contextPtr, particles, inputBuffer);
    mynydd::uploadUniformData<Params>(contextPtr, params, uniformBuffer);

    auto groupCount = (nParticles + 63) / 64;

    auto pipeline = std::make_shared<mynydd::ComputeEngine<Particle>>(
        contextPtr, "shaders/morton_u32_3d.comp.spv",
        std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{
            inputBuffer, outputBuffer, uniformBuffer
        },
        groupCount
    );

    mynydd::executeBatch<Particle>(contextPtr, {pipeline});

    std::vector<Particle> outParticles = mynydd::fetchData<Particle>(contextPtr, outputBuffer, nParticles);


    // Sort particles in-place by key
    std::sort(outParticles.begin(), outParticles.end(),
            [](const Particle &a, const Particle &b) {
                return a.key < b.key;
            });


    std::vector<KeyRange> keyRanges = computeKeyRanges(outParticles, 1.0f);

    // Extract keys
    std::vector<uint32_t> keys(nParticles);
    for (uint32_t i = 0; i < nParticles; ++i) {
        std::cerr << outParticles[i].key << " " << outParticles[i].position.x << " "
                  << outParticles[i].position.y << " " << outParticles[i].position.z  
                 << " Range: " << keyRanges[i].keyMin << " - " << keyRanges[i].keyMax 
                 << " dRange " << keyRanges[i].keyMax - keyRanges[i].keyMin << std::endl;
        keys[i] = outParticles[i].key;
    }

    // Check uniqueness directly on sorted particles
    for (size_t i = 1; i < outParticles.size()-1; ++i) {
        REQUIRE(outParticles[i].key != outParticles[i-1].key);
    }
}