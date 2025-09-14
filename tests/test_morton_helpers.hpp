#ifndef TEST_MORTON_HELPERS_HPP
#define TEST_MORTON_HELPERS_HPP
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <mynydd/mynydd.hpp>
#include <mynydd/shader_interop.hpp>


struct Particle {
    alignas(16) glm::vec3 position;
    // uint32_t key;
};

struct Vec3Aln16 {
    alignas(16) glm::vec3 data;
};


struct KeyRange {
    uint32_t keyMin;
    uint32_t keyMax;
};

std::vector<Particle> getMortonTestGridRegularParticleData(uint32_t nBits);
std::vector<KeyRange> computeKeyRanges(const std::vector<Particle>& particles, float dmax);
std::vector<uint32_t> runMortonTest(std::shared_ptr<mynydd::VulkanContext> contextPtr, const uint32_t nBits, std::vector<Particle>& particles);
int morton3D(int x, int y, int z);
int morton2D(int x, int y);
int morton3D_loop(int x, int y, int z, int a);
uint binPosition(float normPos, uint nbits);
uint ijk2ak(uvec3 ijk, uint nBits);
// Decode a 3D Morton code into (x,y,z) with up to nbits bits per axis
uvec3 decodeMorton3D(uint code, uint nbits);

#endif // TEST_MORTON_HELPERS_HPP