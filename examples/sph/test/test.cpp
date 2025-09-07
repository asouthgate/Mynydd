#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>

#include <mynydd/shader_interop.hpp>
#include "../src/kernels.comp.kern"
#include "../src/sph.hpp"


TEST_CASE("test_spiky_kernel", "[sph]") {
    float h = 1.329f;
    float r = 0.39881f;

    // CHECK(std::fabs(debrun_spiky_kernel(r, h) - 0.6179313391538699f) < 1e-7f);
    CHECK(debrun_spiky_kernel(r, h) == Catch::Approx(0.6179313391538699f).margin(1e-7f));
    REQUIRE(debrun_spiky_kernel(-0.000001f, h) == 0.0f);
    REQUIRE(debrun_spiky_kernel(1.33f, h) == 0.0f);
}

TEST_CASE("test_kernel_dwdr", "[sph]") {
    float h = 1.329f;
    CHECK(std::fabs(debrun_spiky_kernel_dwdr(0.0f, h)) > 1.0f); // doesn't disappear at origin
}

TEST_CASE("test_kernel_grad", "[sph]") {
    float dx = 0.1361f;
    float dy = 0.9981f;
    float h = 1.8f;
    float r = cal_r(dx, dy);

    auto grad = debrun_spiky_kernel_grad(dx, dy, h);

    CHECK(std::fabs(grad.x - dx * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
    CHECK(std::fabs(grad.y - dy * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
    CHECK(grad.x < 0.0f); // we expect the gradient to be pointing downwards
    CHECK(grad.y < 0.0f);
}

TEST_CASE("test_debrun_spiky_kernel_lap", "[sph]") {
    float r = 4.601086828130937f;
    float h = 1.2f;
    float l = debrun_spiky_kernel_lap(r, h);

    CHECK(std::fabs(-27.948690054856538f * get_debrun_coeff(h) - l) < 1e-5f);
}

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


TEST_CASE("Test that pipeline produces correct density values", "[sph]") {


    auto simulated = simulate_inputs(4096 * 16);
    SPHData out = run_sph_example(simulated);

    auto inputPos = simulated.positions;
    auto inputDensities = simulated.densities;

    auto outputPos = out.positions;
    auto cellData = out.cellInfos;
    auto densities = out.densities;
    auto indexData = out.sortedIndices;

    size_t nCells = static_cast<uint32_t>(cellData.size());
    size_t printed = 0;

    std:: cerr << "Checking SPH output with" << nCells << " cells" << std::endl;

    for (uint32_t morton_key = 0; morton_key < nCells; ++morton_key) {
        uint32_t start = cellData[morton_key].left;
        uint32_t end = cellData[morton_key].right;

        if (start == end) {
            continue; // Empty cell
        }

        float avg_dens = 0.0f;
        for (uint32_t pind = start; pind < end; ++pind) {
            uint32_t unsorted_ind = indexData[pind];
            auto particle = inputPos[unsorted_ind];
            
            REQUIRE(particle.position.x == outputPos[pind].position.x);
            REQUIRE(particle.position.y == outputPos[pind].position.y);
            REQUIRE(particle.position.z == outputPos[pind].position.z);

            if (morton_key < 5 || morton_key > nCells - 5) {
                std::cerr << "Cell " << morton_key 
                    << " with start" << start
                    << " and end" << end
                    << " contains particle at original_index"
                    << pind << " -> " << unsorted_ind << " at position "
                    << particle.position.x << ", " 
                    << particle.position.y << ", " 
                    << particle.position.z << " with density: " 
                    << inputDensities[unsorted_ind] << std::endl;
                printed++;
            }

            avg_dens += inputDensities[unsorted_ind];
        }
        avg_dens /= float(end - start);
        REQUIRE (fabs(avg_dens - densities[start]) < 1e-5);

    }
}