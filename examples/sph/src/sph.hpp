#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct Vec3Aln16 {
    alignas(16) glm::vec3 position;
};

struct DensityParams {
    uint32_t nBits;
    uint32_t nParticles;
    alignas(16) glm::vec3 domainMin; // alignas required for silly alignment issues
    alignas(16) glm::vec3 domainMax;
    int dist;
};


struct SPHData {
    std::vector<float> densities;
    std::vector<Vec3Aln16> positions;
    std::vector<uint32_t> mortonKeys;
    std::vector<uint32_t> sortedIndices;
    std::vector<mynydd::CellInfo> cellInfos;
    DensityParams params;
};

SPHData simulate_inputs(uint32_t nParticles);

SPHData run_sph_example(const SPHData& inputData, uint32_t nBitsPerAxis = 4, int dist = 0);