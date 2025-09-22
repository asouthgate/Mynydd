#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>

#include "sph.hpp"

void printSPHDataCSV(const SPHData& data) {
    std::cout << "density,pressure,fpx,fpy,fpz,x,y,z,morton_key\n";
    for (size_t i = 0; i < data.mortonKeys.size(); ++i) {
        const auto& pos = data.positions[i];
        double density = data.densities[i];
        uint32_t key = data.mortonKeys[i];
        std::cout << density << "," << data.pressures[i] << "," << data.pressureForces[i].data.x << ","
                    << data.pressureForces[i].data.y << "," << data.pressureForces[i].data.z << ","
                  << pos.data.x << "," << pos.data.y << "," << pos.data.z << ","
                  << key << "\n";
    }
}

int main(int argc, char** argv) {

    uint32_t nParticles = 4096 * 16;
    uint32_t nBitsPerAxis = 4;
    uint32_t niterations = 100;
    double dt = 0.001;
    double rho0_mod = 1.0;
    double c2 = 0.02;
    double mu = 0.01;
    if (argc > 2) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
        nBitsPerAxis = static_cast<uint32_t>(std::atoi(argv[2]));
        niterations = static_cast<uint32_t>(std::atoi(argv[3]));
        dt = std::atof(argv[4]);
        rho0_mod = std::atof(argv[5]);
        c2 = std::atof(argv[6]);
        mu = std::atof(argv[7]);
    } else if (argc > 7) {
        std::cerr << "Usage: nParticles nBits niterations dt rho0_mod c2 mu" << std::endl;
        return EXIT_FAILURE;
    }

    auto simulated = simulate_inputs(nParticles);
    double h = 1.0 / (1 << nBitsPerAxis);

    double nbr_vol_prop = (4.0 / 3.0) * M_PI * h * h * h;
    double rho0 = nParticles * nbr_vol_prop * rho0_mod;
    
    std::cerr << "Running SPH with " << nParticles << " particles, " << nBitsPerAxis << " bits per axis, "
                << niterations << " iterations, dt=" << dt << ", rho0=" << rho0 << ", c2=" << c2 << std::endl;

    SPHParams params {
        nBitsPerAxis,
        nParticles,
        glm::dvec3(0.0),
        glm::dvec3(1.0),
        1,
        dt,
        h,
        1.0,
        glm::dvec3(0.0, 0.0, -9.0),
        rho0,
        c2,
        mu
    };


    auto outputs = run_sph_example(simulated, params, niterations);

    printSPHDataCSV(outputs);

}