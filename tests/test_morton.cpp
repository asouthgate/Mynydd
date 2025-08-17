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
#include "test_morton_helpers.hpp"


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
    const uint32_t nBits = 4;
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    std::vector<Particle> particles = getMortonTestGridRegularParticleData(nBits);
    runMortonTest(contextPtr, nBits, particles);
    // TODO:
}