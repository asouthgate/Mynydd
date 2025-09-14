#include <cstdint>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>

#include <mynydd/shader_interop.hpp>
#include "../src/kernels.comp.kern"
#include "../src/sph.hpp"
#include "../src/pipelines/shaders/morton_kernels.comp.kern"


TEST_CASE("test_spiky_kernel_coeff_3D", "[sph]") {
    float h = 0.789f;
    CHECK(get_debrun_coeff_3D(h) == Catch::Approx(19.791529914316335).margin(1e-7f));
}

TEST_CASE("test_spiky_kernel", "[sph]") {
    float h = 1.329f;
    float r = 0.39881f;
    REQUIRE(debrun_spiky_kernel(-0.000001f, h) == 0.0f);
    REQUIRE(debrun_spiky_kernel(1.33f, h) == 0.0f);

    float sum = 0;
    int d = 25;
    float maxd = (float) d;
    float scale = 1.5f;
    // Test that it approximately integrates to 1 by computing a grid
    for (int i = -d; i < d; ++i) {
        for (int j = -d; j < d; ++j) {
            for (int k =  -d; k < d; ++k) {
                vec3 pos = vec3((float) i * scale / maxd, (float) j * scale / maxd, (float) k * scale / maxd);
                REQUIRE(pos.x <= scale);
                REQUIRE(pos.x >= -scale);
                float dist = glm::length(pos);
                REQUIRE(dist <= std::sqrt(3.0f) * scale);
                float dx = scale / maxd;
                float vol = dx * dx * dx;
                sum += debrun_spiky_kernel(dist, h) * vol;
            }
        }
    }
    REQUIRE(sum == Catch::Approx(1.0).margin(1e-2f));
}

// TEST_CASE("test_kernel_dwdr", "[sph]") {
//     float h = 1.329f;
//     CHECK(std::fabs(debrun_spiky_kernel_dwdr(0.0f, h)) > 1.0f); // doesn't disappear at origin
// }

// TEST_CASE("test_kernel_grad", "[sph]") {
//     float dx = 0.1361f;
//     float dy = 0.9981f;
//     float h = 1.8f;
//     float r = cal_r(dx, dy);

//     auto grad = debrun_spiky_kernel_grad(dx, dy, h);

//     CHECK(std::fabs(grad.x - dx * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
//     CHECK(std::fabs(grad.y - dy * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
//     CHECK(grad.x < 0.0f); // we expect the gradient to be pointing downwards
//     CHECK(grad.y < 0.0f);
// }

// TEST_CASE("test_debrun_spiky_kernel_lap", "[sph]") {
//     float r = 4.601086828130937f;
//     float h = 1.2f;
//     float l = debrun_spiky_kernel_lap(r, h);

//     CHECK(std::fabs(-27.948690054856538f * get_debrun_coeff(h) - l) < 1e-5f);
// }

TEST_CASE("cal_pressure_wcsph behaves correctly", "[sph]") {
    float rho = 1100.0f;
    float rho0 = 1000.0f;
    float c2 = 1500.0f;
    float gamma = 7.0f;

    float result = cal_pressure_wcsph(rho, rho0, c2, gamma);
    float expected_bweak = c2 * rho0 / gamma;
    float expected = expected_bweak * (std::pow(rho / rho0, gamma) - 1.0f);

    REQUIRE(std::abs(result - expected) < 1e-6f);
}

TEST_CASE("cal_rho_ij returns zero outside support radius", "[sph]") {
    float mass_j = 2.0f;
    float h = 1.0f;

    // distance outside kernel support
    float dist_far = 1.1f;
    REQUIRE(cal_rho_ij(mass_j, dist_far, h) == 0.0f);

    // distance inside kernel support
    float dist_near = 0.5f;
    REQUIRE(cal_rho_ij(mass_j, dist_near, h) > 0.0f);
}

TEST_CASE("cal_pressure_force_coefficient computes correctly", "[sph]") {
    float pi = 2000.0f;
    float pj = 1500.0f;
    float rhoi = 1000.0f;
    float rhoj = 900.0f;
    float mj = 1.5f;

    float result = cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    float expected = ((pi / (rhoi*rhoi)) + (pj / (rhoj*rhoj))) * mj;

    REQUIRE(std::abs(result - expected) < 1e-6f);
}


TEST_CASE("Test that pipeline produces correct density values for d = 0 for first particle in each cell", "[sph]") {

    auto simulated = simulate_inputs(4096 * 16);
    SPHData out = run_sph_example(simulated, 4, 0);

    auto inputPos = simulated.positions;
    auto inputDensities = simulated.densities;

    auto outputPos = out.positions;
    auto cellData = out.cellInfos;
    auto densities = out.densities;
    auto indexData = out.sortedIndices;
    
    size_t nCells = static_cast<uint32_t>(cellData.size());
    size_t printed = 0;

    std:: cerr << "Checking SPH output with" << nCells << " cells" << std::endl;

    for (uint32_t key = 0; key < nCells; ++key) {


        uint32_t start = cellData[key].left;
        uint32_t end = cellData[key].right;

        if (start == end) {
            continue; // Empty cell
        }

        float dens = 0.0f;
        for (uint32_t pind = start; pind < end; ++pind) {
            uint32_t unsorted_ind = indexData[pind];
            auto particle = inputPos[unsorted_ind];
            
            REQUIRE(particle.position.x == outputPos[pind].position.x);
            REQUIRE(particle.position.y == outputPos[pind].position.y);
            REQUIRE(particle.position.z == outputPos[pind].position.z);
            dens += cal_rho_ij(1.0, length(particle.position - outputPos[start].position), 1.0);
        }
        std:: cerr << "Cell " << key << " has " << (end - start) << " particles, computed density " << dens 
            << " vs gpu density " << densities[start] << std::endl;
        REQUIRE (fabs(dens - densities[start]) < 1e-4);

    }
}


TEST_CASE("Test that pipeline produces correct density values for random cell with d = 1, calculated directly in validation", "[sph]") {
    // This test manually scans through in a loop to find particles in or near a cell
    // And then manually calculate the densities
    // The validation computation does therefore not use the index, so it is a more independent check

    auto simulated = simulate_inputs(512);
    SPHData out = run_sph_example(simulated, 4, 1);

    auto inputPos = simulated.positions;
    auto inputDensities = simulated.densities;

    auto outputPos = out.positions;
    auto outputDensities = out.densities;


    for (uint32_t p0idx : {0, 27, 35, 109, 111}) {

        // choose from output pos so we can match to validation output density
        glm::vec3 p0 = outputPos[p0idx].position;
        uvec3 ijk = uvec3(
            binPosition(p0.x, out.params.nBits),
            binPosition(p0.y, out.params.nBits),
            binPosition(p0.z, out.params.nBits)
        );
        std::cerr << "Checking density at index " << p0idx <<  " position " << p0.x << ", " << p0.y << ", " << p0.z << std::endl;
        float densitySum = 0;
        uint count = 0;

        // Now must iterate over inputPos which is matched to inputDensities
        for (size_t pind = 0; pind < inputPos.size(); ++pind) {
            auto p = inputPos[pind].position;
            uvec3 b = uvec3(
                binPosition(p.x, out.params.nBits),
                binPosition(p.y, out.params.nBits),
                binPosition(p.z, out.params.nBits)
            );

            // check if b is within a distance of 1 of ijk on an all axes
            if ( (b.x + 1 >= ijk.x) && (b.x <= ijk.x + 1) &&
                (b.y + 1 >= ijk.y) && (b.y <= ijk.y + 1) &&
                (b.z + 1 >= ijk.z) && (b.z <= ijk.z + 1) ) 
            {

                densitySum += cal_rho_ij(1.0, length(p - p0), 1.0);
                count++;
            }
        }

        // Store average density (or 0 if no neighbors)
        float gpu_dens = outputDensities[p0idx];
        std::cerr << "Cell " << ijk.x << ", " << ijk.y << ", " << ijk.z 
            << " has " << count << " particles in or near it, average density " << densitySum 
            << " and gpu density " << gpu_dens 
            << std::endl;
        REQUIRE (fabs(gpu_dens - densitySum) < 1e-5);
    }

}