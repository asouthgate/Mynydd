#ifndef TEST_MORTON_HELPERS_HPP
#define TEST_MORTON_HELPERS_HPP
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <mynydd/mynydd.hpp>
#include <mynydd/shader_interop.hpp>


struct dVec3Aln32 {
    alignas(32) glm::dvec3 data;
};


struct KeyRange {
    uint32_t keyMin;
    uint32_t keyMax;
};

std::vector<dVec3Aln32> getMortonTestGridRegularParticleData(uint32_t nBits);
std::vector<KeyRange> computeKeyRanges(const std::vector<dVec3Aln32>& particles, double dmax);
std::vector<uint32_t> runMortonTest(std::shared_ptr<mynydd::VulkanContext> contextPtr, const uint32_t nBits, std::vector<dVec3Aln32>& particles);
int morton3D(int x, int y, int z);
int morton2D(int x, int y);
int morton3D_loop(int x, int y, int z, int a);
uint binPosition(double normPos, uint nbits);
uint ijk2ak(uvec3 ijk, uint nBits);
// Decode a 3D Morton code into (x,y,z) with up to nbits bits per axis
uvec3 decodeMorton3D(uint code, uint nbits);

#endif // TEST_MORTON_HELPERS_HPP