#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <chrono>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <vector>
#include <mynydd/mynydd.hpp>

#include "test_morton_helpers.hpp"
#include "../src/pipelines/shaders/morton_kernels.comp.kern"


std::vector<Particle> getMortonTestGridRegularParticleData(uint32_t nBits) {
    const uint32_t nPerDim = pow(2, nBits);
    const uint32_t nParticles = pow(nPerDim, 3);
    // generate deterministic particle positions
    std::vector<Particle> testParticles(nParticles);
    uint32_t idx = 0;
    for (uint32_t z = 0; z < nPerDim; ++z) {
        for (uint32_t y = 0; y < nPerDim; ++y) {
            for (uint32_t x = 0; x < nPerDim; ++x) {
                testParticles[idx].position = glm::vec3(float(x), float(y), float(z));
                // testParticles[idx].key = 0;
                ++idx;
            }
        }
    }
    std::mt19937 gen(12345);
    std::shuffle(testParticles.begin(), testParticles.end(), gen);
    return testParticles;
}

std::vector<uint32_t> runMortonTest(
    std::shared_ptr<mynydd::VulkanContext> contextPtr,
    const uint32_t nBits,
    std::vector<Particle>& particles
) {
    std::cerr << "TEST: Running Morton test with " << nBits << " bits..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    const uint32_t nPerDim = pow(2, nBits);
    const uint32_t nParticles = pow(nPerDim, 3);
    struct Params {
        uint32_t nBits;
        uint32_t nParticles;
        alignas(16) glm::vec3 domainMin; // alignas required for silly alignment issues
        alignas(16) glm::vec3 domainMax;
    } params{nBits, nParticles, glm::vec3(0.0f), glm::vec3(float(nPerDim - 1))};

    auto inputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, nParticles * sizeof(Particle), false);
    auto outputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, nParticles * sizeof(uint32_t), false);
    auto uniformBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, sizeof(Params), true);

    auto t1 = std::chrono::high_resolution_clock::now();

    mynydd::uploadData<Particle>(contextPtr, particles, inputBuffer);
    mynydd::uploadUniformData<Params>(contextPtr, params, uniformBuffer);

    auto groupCount = (nParticles + 63) / 64;

    auto pipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr, "shaders/morton_u32_3d.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{
            inputBuffer, outputBuffer, uniformBuffer
        },
        groupCount
    );
    auto t2 = std::chrono::high_resolution_clock::now();
    mynydd::executeBatch(contextPtr, {pipeline});

    std::vector<uint32_t> outKeys = mynydd::fetchData<uint32_t>(contextPtr, outputBuffer, nParticles);


    size_t sum = 0;
    for (size_t i = 0; i < outKeys.size(); ++i) {
        sum += outKeys[i];
    }
    REQUIRE((sum > 0 && "Keys came back all zero. This is likely a bug in the Morton shader."));

    // std::vector<uint32_t> outKeys(nParticles);
    // for (size_t i = 0; i < outParticles.size(); ++i) {
    //     outKeys[i] = outParticles[i].key;
    // }

    // Sort particlesqin-place by key
    // std::sort(outParticles.begin(), outParticles.end(),
    //         [](const Particle &a, const Particle &b) {
    //             return a.key < b.key;
    //         });

    auto t3 = std::chrono::high_resolution_clock::now();

    auto durationInput = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto durationCompute = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto durationFetch = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::cerr << "Morton test completed in: "
              << "Input: " << durationInput << "µs, "
              << "Compute: " << durationCompute << "µs, "
              << "Fetch: " << durationFetch << "µs, "
              << std::endl;

    return outKeys;
}