#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <mynydd/mynydd.hpp>

#include "test_morton_helpers.hpp"


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

std::vector<uint32_t> runMortonTest(std::shared_ptr<mynydd::VulkanContext> contextPtr, const uint32_t nBits) {
    const uint32_t nPerDim = pow(2, nBits);
    const uint32_t nParticles = pow(nPerDim, 3);
    struct Params {
        uint32_t nBits;
        uint32_t nParticles;
        alignas(16) glm::vec3 domainMin; // alignas required for silly alignment issues
        alignas(16) glm::vec3 domainMax;
    } params{nBits, nParticles, glm::vec3(0.0f), glm::vec3(float(nPerDim - 1))};

    auto inputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, nParticles * sizeof(Particle), false);
    auto outputBuffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr, nParticles * sizeof(Particle), true);
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

    // Check uniqueness directly on sorted particles
    for (size_t i = 1; i < outParticles.size()-1; ++i) {
        REQUIRE(outParticles[i].key != outParticles[i-1].key);
    }

    std::vector<uint32_t> keys(nParticles);
    for (size_t i = 0; i < outParticles.size(); ++i) {
        keys[i] = outParticles[i].key;
    }
    return keys;
}