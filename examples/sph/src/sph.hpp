#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct ParticlePosition {
    alignas(16) glm::vec3 position;
};


struct SPHData {
    std::vector<float> densities;
    std::vector<ParticlePosition> positions;
    std::vector<uint32_t> sortedIndices;
    std::vector<mynydd::CellInfo> cellInfos;
};

SPHData simulate_inputs(uint32_t nParticles);

SPHData run_sph_example(const SPHData& inputData);