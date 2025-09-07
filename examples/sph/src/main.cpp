#include <chrono>
#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

#include "sph.hpp"

int main(int argc, char** argv) {

    uint32_t nParticles = 4096 * 16;
    std::cerr << argc << std::endl;
    if (argc == 2) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
    } else if (argc > 2) {
        std::cerr << "Usage: nParticles" << std::endl;
        return EXIT_FAILURE;
    }

    run_sph_example(nParticles);

}