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
    if (argc == 2) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
    } else if (argc == 3) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
        nBitsPerAxis = static_cast<uint32_t>(std::atoi(argv[2]));
    }
    else if (argc == 4) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
        nBitsPerAxis = static_cast<uint32_t>(std::atoi(argv[2]));
        niterations = static_cast<uint32_t>(std::atoi(argv[3]));
    }
    else if (argc == 5) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
        nBitsPerAxis = static_cast<uint32_t>(std::atoi(argv[2]));
        niterations = static_cast<uint32_t>(std::atoi(argv[3]));
        dt = std::atof(argv[4]);
    }
    else if (argc > 5) {
        std::cerr << "Usage: nParticles" << std::endl;
        return EXIT_FAILURE;
    }

    auto simulated = simulate_inputs(nParticles);
    double h = 1.0 / (1 << nBitsPerAxis);
    // SPHParams params {
    //     nBitsPerAxis,
    //     nParticles,
    //     glm::dvec3(0.0),
    //     glm::dvec3(1.0),
    //     1,
    //     dt,
    //     h,
    //     1.0,
    //     glm::dvec3(0.0, 0.0, -9.0)
    // };

    double nbr_vol_prop = (4.0 / 3.0) * M_PI * h * h * h;
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
        nParticles * nbr_vol_prop
    };


    auto outputs = run_sph_example(simulated, params, niterations);

    printSPHDataCSV(outputs);

}