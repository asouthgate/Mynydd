#include <cstdint>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <catch2/catch_approx.hpp>
#include <iomanip> 

#include <mynydd/shader_interop.hpp>
#include "../src/kernels.comp.kern"
#include "../src/sph.hpp"
#include "../src/pipelines/shaders/morton_kernels.comp.kern"


TEST_CASE("test_spiky_kernel_coeff_3D", "[sph]") {
    double h = 0.789;
    CHECK(get_debrun_coeff_3D(h) == Catch::Approx(19.791529914316335).margin(1e-7));
}

TEST_CASE("test_spiky_kernel", "[sph]") {
    double h = 1.329;
    double r = 0.39881;
    REQUIRE(debrun_spiky_kernel(-0.000001, h) == 0.0);
    REQUIRE(debrun_spiky_kernel(1.33, h) == 0.0);
    REQUIRE(debrun_spiky_kernel(1.30, h) >= 0.0);
    REQUIRE(debrun_spiky_kernel(0.1, h) >= 0.0);

    double sum = 0;
    int d = 25;
    double maxd = (double) d;
    double scale = 1.5;
    // Test that it approximately integrates to 1 by computing a grid
    for (int i = -d; i < d; ++i) {
        for (int j = -d; j < d; ++j) {
            for (int k =  -d; k < d; ++k) {
                dvec3 pos = dvec3((double) i * scale / maxd, (double) j * scale / maxd, (double) k * scale / maxd);
                REQUIRE(pos.x <= scale);
                REQUIRE(pos.x >= -scale);
                double dist = glm::length(pos);
                REQUIRE(dist <= std::sqrt(3.0) * scale);
                double dx = scale / maxd;
                double vol = dx * dx * dx;
                sum += debrun_spiky_kernel(dist, h) * vol;
            }
        }
    }
    REQUIRE(sum == Catch::Approx(1.0).margin(1e-2));
}

TEST_CASE("test_kernel_dwdr", "[sph]") {
     double h = 1.329;
     CHECK(std::fabs(debrun_spiky_kernel_dwdr(0.0, h)) > 1.0); // doesn't disappear at origin
}

TEST_CASE("test_kernel_grad", "[sph]") {
    double dx = 0.1361;
    double dy = 0.9981;
    double dz = 0.5012;
    dvec3 pos = dvec3(dx, dy, dz);
    double h = 1.8;
    double r = length(pos);

    auto grad = debrun_spiky_kernel_grad(pos, h);

    CHECK(std::fabs(grad.x - dx * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6);
    CHECK(std::fabs(grad.y - dy * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6);
    CHECK(std::fabs(grad.z - dz * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6);
    CHECK(grad.x < 0.0); // we expect the gradient to be pointing downwards
    CHECK(grad.y < 0.0);
}

TEST_CASE("test_debrun_spiky_kernel_lap", "[sph]") {
    double h = 1.2;
    double dr = 1e-5;  // small step for finite differences
    dvec3 v1 = dvec3(0.9, 0.9, 0.9);
    double r = length(v1);

    // Compute gradient at v1
    auto grad_center = debrun_spiky_kernel_grad(v1, h);

    // Finite difference approximation of Laplacian
    double lap_fd = 0.0f;
    for (int i = 0; i < 3; ++i) {
        dvec3 dv = dvec3(0.0);
        if (i == 0) dv.x = dr;
        if (i == 1) dv.y = dr;
        if (i == 2) dv.z = dr;

        auto grad_forward = debrun_spiky_kernel_grad(v1 + dv, h);
        auto grad_backward = debrun_spiky_kernel_grad(v1 - dv, h);
        double derivative = (grad_forward[i] - grad_backward[i]) / (2.0 * dr);
        // derivative of i-th component along its coordinate
        // double derivative = (grad_forwawrd[i] - grad_center[i]) / dr;
        lap_fd += derivative;    
    }
    double lap_analytic = debrun_spiky_kernel_lap(r, h);

    std:: cout << "Finite difference Laplacian: " << lap_fd << ", Analytic Laplacian: " << lap_analytic << std::endl;
    // Compare
    REQUIRE(Catch::Approx(lap_fd).epsilon(1e-3) == lap_analytic);
}

TEST_CASE("cal_pressure_wcsph behaves correctly", "[sph]") {
    double rho = 1100.0;
    double rho0 = 1000.0;
    double c2 = 1500.0;
    double gamma = 7.0;

    double result = cal_pressure_wcsph(rho, rho0, c2, gamma);
    double expected_bweak = c2 * rho0 / gamma;
    double expected = expected_bweak * (std::pow(rho / rho0, gamma) - 1.0);

    REQUIRE(std::abs(result - expected) < 1e-6f);
}

TEST_CASE("cal_rho_ij returns zero outside support radius", "[sph]") {
    double mass_j = 2.0;
    double h = 1.0;

    // distance outside kernel support
    double dist_far = 1.1;
    REQUIRE(cal_rho_ij(mass_j, dist_far, h) == 0.0);

    // distance inside kernel support
    double dist_near = 0.5;
    REQUIRE(cal_rho_ij(mass_j, dist_near, h) > 0.0);
}

TEST_CASE("cal_pressure_force_coefficient computes correctly", "[sph]") {
    double pi = 2000.0;
    double pj = 1500.0;
    double rhoi = 1000.0;
    double rhoj = 900.0;
    double mj = 1.5;

    double result = cal_pressure_force_coefficient(pi, pj, rhoi, rhoj, mj);
    double expected = ((pi / (rhoi*rhoi)) + (pj / (rhoj*rhoj))) * mj;

    REQUIRE(std::abs(result - expected) < 1e-6);
}


TEST_CASE("Test that pipeline produces correct density values for d = 0 for first particle in each cell", "[sph]") {

    auto simulated = simulate_inputs(4096 * 16);
    auto nBitsPerAxis = 4;
    SPHData out = run_sph_example(simulated, nBitsPerAxis, 0);

    auto inputPos = simulated.positions;
    auto inputDensities = simulated.densities;

    auto outputPos = out.positions;
    auto cellData = out.cellInfos;
    auto densities = out.densities;
    auto indexData = out.sortedIndices;
    
    size_t nCells = static_cast<uint32_t>(cellData.size());
    size_t printed = 0;
    double h = 1.5 / (1 << nBitsPerAxis);
    std:: cerr << "Checking SPH output with" << nCells << " cells" << std::endl;

    for (uint32_t key = 0; key < nCells; ++key) {


        uint32_t start = cellData[key].left;
        uint32_t end = cellData[key].right;

        if (start == end) {
            continue; // Empty cell
        }

        double dens = 0.0;
        for (uint32_t pind = start; pind < end; ++pind) {
            uint32_t unsorted_ind = indexData[pind];
            auto particle = inputPos[unsorted_ind];
            
            REQUIRE(particle.data.x == outputPos[pind].data.x);
            REQUIRE(particle.data.y == outputPos[pind].data.y);
            REQUIRE(particle.data.z == outputPos[pind].data.z);
            dens += cal_rho_ij(1.0, length(particle.data - outputPos[start].data), h);
        }
        // std:: cerr << "Cell " << key << " has " << (end - start) << " particles, computed density " << std::fixed << std::setprecision(6) << dens  
            // << " vs gpu density " << std::fixed << std::setprecision(6) << densities[start] << std::endl;
        REQUIRE (fabs(dens - densities[start]) < 1e-2);

    }
}


TEST_CASE("Test that pipeline produces correct density values for random cell with d = 1, calculated directly in validation", "[sph]") {
    // This test manually scans through in a loop to find particles in or near a cell
    // And then manually calculate the densities
    // The validation computation does therefore not use the index, so it is a more independent check

    auto simulated = simulate_inputs(512);
    auto nBitsPerAxis = 4;
    SPHData out = run_sph_example(simulated, nBitsPerAxis, 1);

    auto inputPos = simulated.positions;
    auto inputDensities = simulated.densities;

    auto outputPos = out.positions; // these are sorted
    auto outputDensities = out.densities;

    auto outputNewPos = out.newPositions;
    auto outputNewVel = out.newVelocities;

    for (uint32_t p0idx : {0, 27, 35, 109, 111}) {

        // choose from output pos so we can match to validation output density
        glm::dvec3 p0 = outputPos[p0idx].data;
        uvec3 ijk = uvec3(
            binPosition(p0.x, out.params.nBits),
            binPosition(p0.y, out.params.nBits),
            binPosition(p0.z, out.params.nBits)
        );
        std::cerr << "Checking density at index " << p0idx <<  " position " << p0.x << ", " << p0.y << ", " << p0.z << std::endl;
        double densitySum = 0;
        uint count = 0;

        // Now must iterate over inputPos which is matched to inputDensities
        for (size_t pind = 0; pind < inputPos.size(); ++pind) {
            auto p = inputPos[pind].data;
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

                densitySum += cal_rho_ij(1.0, length(p - p0), 1.5 / (1 << nBitsPerAxis));
                count++;
            }
        }

        // Store average density (or 0 if no neighbors)
        double gpu_dens = outputDensities[p0idx];
        REQUIRE (fabs(gpu_dens - densitySum) < 1e-5);
    }

}


TEST_CASE("Test that pipeline produces correct position, velocity values for random cell with d = 1, calculated directly in validation", "[sph]") {
    // This test manually scans through in a loop to find particles in or near a cell
    // And then manually calculate the densities
    // The validation computation does therefore not use the index, so it is a more independent check

    auto simulated = simulate_inputs(1024);
    auto nBitsPerAxis = 4;
    SPHData out = run_sph_example(simulated, nBitsPerAxis, 1);

    auto outputPos = out.positions; // these are sorted
    auto outputVel = out.velocities; // these are sorted
    auto outputPressureForces = out.pressureForces;

    auto outputNewPos = out.newPositions;
    auto outputNewVel = out.newVelocities;

    for (uint32_t p0idx : {0, 27, 35, 109, 111}) {
        // Just check that leapfrog is done correctly using the output pressure forces

        glm::dvec3 p0 = outputPos[p0idx].data;
        glm::dvec3 v0 = outputVel[p0idx].data;
        glm::dvec3 f0 = outputPressureForces[p0idx].data;
        REQUIRE(glm::length(f0) > 0);
        double dt = 0.001;
        glm::dvec3 expected_v1 = v0 + f0 * dt;
        glm::dvec3 expected_x1 = p0 + expected_v1 * dt;
        glm::dvec3 gpu_v1 = outputNewVel[p0idx].data;
        glm::dvec3 gpu_x1 = outputNewPos[p0idx].data;
        std::cerr << "Checking leapfrog at index " << p0idx <<  " position " << p0.x << ", " << p0.y << ", " << p0.z << std::endl;
        REQUIRE (glm::length(gpu_v1 - expected_v1) < 1e-6);
        REQUIRE (glm::length(gpu_x1 - expected_x1) < 1e-6);
        REQUIRE (glm::length(gpu_x1 - p0) > 1e-6);
        REQUIRE (glm::length(gpu_v1 - v0) > 1e-6);

    }
}