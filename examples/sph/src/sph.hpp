#include <chrono>
#include <cstdint>
#include <random>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct ParticlePosition {
    alignas(16) glm::vec3 position;
};

void run_sph_example(uint32_t nParticles);