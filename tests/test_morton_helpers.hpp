#ifndef TEST_MORTON_HELPERS_HPP
#define TEST_MORTON_HELPERS_HPP
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <mynydd/mynydd.hpp>


struct Particle {
    glm::vec3 position;
    // uint32_t key;
};

struct KeyRange {
    uint32_t keyMin;
    uint32_t keyMax;
};

std::vector<Particle> getMortonTestGridRegularParticleData(uint32_t nBits);
std::vector<KeyRange> computeKeyRanges(const std::vector<Particle>& particles, float dmax);
std::vector<uint32_t> runMortonTest(std::shared_ptr<mynydd::VulkanContext> contextPtr, const uint32_t nBits, std::vector<Particle>& particles);

#endif // TEST_MORTON_HELPERS_HPP