#include <algorithm>
#include <glm/fwd.hpp>
#include <iostream>
#include <cmath>  // For M_PI
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>

#include <mynydd/shader_interop.hpp>
#include "../src/mesh.hpp"
#include "../src/kernels.comp.kern"

const double EPS_CHECK = 1e-6;


TEST_CASE("computeIntersectionParams: center hit", "[mesh]") {
    glm::dvec3 v0(0.0, 0.0, 0.0);
    glm::dvec3 v1(1.0, 0.0, 0.0);
    glm::dvec3 v2(0.0, 1.0, 0.0);

    glm::dvec3 p0(0.25, 0.25, -1.0);
    glm::dvec3 p1(0.25, 0.25,  1.0);

    IntersectParams params = computeIntersectionParams(v0, v1, v2, p0, p1);

    REQUIRE_FALSE(isParallel(params));
    REQUIRE_FALSE(isOutsideUV(params));
    REQUIRE(doesIntersect(params));

    // expected (from Möller–Trumbore): u=0.25, v=0.25, t=0.5 (and determinant a = -2.0)
    CHECK(params.u == Catch::Approx(0.25).margin(EPS_CHECK)); // u
    CHECK(params.v == Catch::Approx(0.25).margin(EPS_CHECK)); // v
    CHECK(params.t == Catch::Approx(0.5).margin(EPS_CHECK));  // t
    CHECK(params.a == Catch::Approx(-2.0).margin(EPS_CHECK)); // det / a
}

TEST_CASE("computeIntersectionParams: miss outside UV", "[mesh]") {
    glm::dvec3 v0(0.0, 0.0, 0.0);
    glm::dvec3 v1(1.0, 0.0, 0.0);
    glm::dvec3 v2(0.0, 1.0, 0.0);

    glm::dvec3 p0(1.5, 1.5, -1.0);
    glm::dvec3 p1(1.5, 1.5,  1.0);

    IntersectParams params = computeIntersectionParams(v0, v1, v2, p0, p1);

    REQUIRE_FALSE(isParallel(params));
    REQUIRE(isOutsideUV(params));

    // expected barycentrics u=v=1.5, t = 0.5, det = -2.0
    CHECK(params.u == Catch::Approx(1.5).margin(EPS_CHECK));
    CHECK(params.v == Catch::Approx(1.5).margin(EPS_CHECK));
    CHECK(params.t == Catch::Approx(0.5).margin(EPS_CHECK));
    CHECK(params.a == Catch::Approx(-2.0).margin(EPS_CHECK));
}

TEST_CASE("computeIntersectionParams: parallel segment", "[mesh]") {
    glm::dvec3 v0(0.0, 0.0, 0.0);
    glm::dvec3 v1(1.0, 0.0, 0.0);
    glm::dvec3 v2(0.0, 1.0, 0.0);

    // segment lies in triangle plane (z == 0), so parallel -> no proper intersection
    glm::dvec3 p0(-1.0, 0.5, 0.0);
    glm::dvec3 p1( 2.0, 0.5, 0.0);

    IntersectParams params = computeIntersectionParams(v0, v1, v2, p0, p1);

    REQUIRE(isParallel(params));
    // For parallel case we return sentinel u,v,t = -1, and w = determinant (near 0)
    CHECK(params.u == Catch::Approx(-1.0).margin(EPS_CHECK));
    CHECK(params.v == Catch::Approx(-1.0).margin(EPS_CHECK));
    CHECK(params.t == Catch::Approx(-1.0).margin(EPS_CHECK));
    CHECK(std::abs(params.a) <= 1e-8); // determinant near zero
}

TEST_CASE("bounce_against_triangle: simple front collision", "[mesh]") {
    glm::dvec3 v0(0.0, 0.0, 0.0);
    glm::dvec3 v1(1.0, 0.0, 0.0);
    glm::dvec3 v2(0.0, 1.0, 0.0);

    // Particle moving towards the triangle from -Z
    glm::dvec3 p0(0.25, 0.25, -1.0);
    glm::dvec3 vel(0.0, 0.0, 2.0); // will reach the plane z=0 in dt=0.5
    double dt = 1.0;
    double restitution = 1.0; // perfectly elastic

    BoundaryResult br = bounce_against_triangle(p0, vel, dt, restitution, v0, v1, v2);

    // The particle should collide
    REQUIRE(br.collision);

    CHECK(br.pos.x == Catch::Approx(0.25).margin(1e-12));
    CHECK(br.pos.y == Catch::Approx(0.25).margin(1e-12));
    CHECK(br.pos.z < 0); // should still be negative because it bounced back

    // The velocity should be reflected along Z
    CHECK(br.vel.x == Catch::Approx(0.0).margin(1e-12));
    CHECK(br.vel.y == Catch::Approx(0.0).margin(1e-12));
    CHECK(br.vel.z == Catch::Approx(-2.0).margin(1e-12)); // reversed
}

TEST_CASE("bounce_against_triangle: misses triangle", "[mesh]") {
    glm::dvec3 v0(0.0, 0.0, 0.0);
    glm::dvec3 v1(1.0, 0.0, 0.0);
    glm::dvec3 v2(0.0, 1.0, 0.0);

    // Particle moving past triangle but not hitting
    glm::dvec3 p0(1.5, 1.5, -1.0);
    glm::dvec3 vel(0.0, 0.0, 2.0);
    double dt = 1.0;
    double restitution = 1.0;

    BoundaryResult br = bounce_against_triangle(p0, vel, dt, restitution, v0, v1, v2);

    REQUIRE_FALSE(br.collision);

    // Position should be just p0 + vel*dt
    CHECK(br.pos.x == Catch::Approx(1.5).margin(1e-12));
    CHECK(br.pos.y == Catch::Approx(1.5).margin(1e-12));
    CHECK(br.pos.z == Catch::Approx(1.0).margin(1e-12));

    // Velocity should be unchanged
    CHECK(br.vel.x == Catch::Approx(0.0).margin(1e-12));
    CHECK(br.vel.y == Catch::Approx(0.0).margin(1e-12));
    CHECK(br.vel.z == Catch::Approx(2.0).margin(1e-12));
}

// Helpers
static inline uint32_t flatIndex(glm::ivec3 ijk, glm::ivec3 dims) {
    return (ijk.z * dims.y + ijk.y) * dims.x + ijk.x;
}

TEST_CASE("Triangle indexing & accumulation", "[mesh]") {
    // Make one triangle fully inside multiple cells
    std::vector<glm::dvec3> verts = {
        {0.1, 0.1, 0.1},
        {1.9, 0.1, 0.1},
        {0.1, 1.9, 0.1},

        {0.1, 0.1, 6.9},
        {1.9, 0.1, 7.9},
        {0.1, 1.9, 7.9},
    };

    double h = 1.0;
    glm::dvec3 domainMin(0.0);
    glm::ivec3 dims(8, 8, 8); // 8 cells

    auto cellToTris = buildCellToTriangles(verts, h, domainMin, dims);

    uint32_t idx = flatIndex({0,0,0}, dims);
    REQUIRE(cellToTris[flatIndex({0,0,0}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({1,0,0}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({0,1,0}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({2,0,0}, dims)].size() == 0);
    REQUIRE(cellToTris[flatIndex({0,2,0}, dims)].size() == 0);

    REQUIRE(cellToTris[flatIndex({0,0,7}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({1,0,7}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({0,1,7}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({2,0,7}, dims)].size() == 0);
    REQUIRE(cellToTris[flatIndex({0,2,7}, dims)].size() == 0);
    REQUIRE(cellToTris[flatIndex({0,1,6}, dims)].size() == 1);

    accumulateNeighbors(cellToTris, dims);
    REQUIRE(cellToTris[flatIndex({2,0,0}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({0,2,0}, dims)].size() == 1);
    REQUIRE(cellToTris[flatIndex({3,0,0}, dims)].size() == 0);
    REQUIRE(cellToTris[flatIndex({0,3,0}, dims)].size() == 0);

}

TEST_CASE("GPU packing consistency", "[mesh]") {
    // Two triangles, two cells
    std::vector<glm::dvec3> verts = {
        {0.1,0.1,0.1}, {0.2,0.1,0.1}, {0.1,0.2,0.1},
        {1.1,0.1,0.1}, {1.2,0.1,0.1}, {1.1,0.2,0.1}
    };

    double h = 1.0;
    glm::dvec3 domainMin(0.0);
    glm::ivec3 dims(3,1,1);

    auto cellToTris = buildCellToTriangles(verts, h, domainMin, dims);
    std::vector<uint32_t> flatTriIndices;
    std::vector<CellTriangles> cellMeta;

    packForGPU(cellToTris, flatTriIndices, cellMeta);

    // Each cell's [left,right) slice should map back correctly
    for (size_t i = 0; i < cellMeta.size(); ++i) {
        auto meta = cellMeta[i];
        for (uint32_t j = meta.left; j < meta.right; ++j) {
            REQUIRE(j < flatTriIndices.size());
        }
    }
}