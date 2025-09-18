#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct dVec3Aln32 {
    alignas(32) glm::dvec3 data;
};


struct DensityParams {
    uint32_t nBits;
    uint32_t nParticles;
    alignas(32) glm::dvec3 domainMin; // alignas required for silly alignment issues
    alignas(32) glm::dvec3 domainMax;
    int dist;
};

struct Step2Params {
    uint32_t nBits;
    uint32_t nParticles;
    alignas(32) glm::dvec3 domainMin; // alignas required for silly alignment issues
    alignas(32) glm::dvec3 domainMax;
    int dist;
    double dt;
};


struct SPHData {
    std::vector<double> densities;
    std::vector<double> pressures;
    std::vector<dVec3Aln32> pressureForces;
    std::vector<dVec3Aln32> positions;
    std::vector<uint32_t> mortonKeys;
    std::vector<uint32_t> sortedIndices;
    std::vector<mynydd::CellInfo> cellInfos;
    DensityParams params;
};

SPHData simulate_inputs(uint32_t nParticles);

SPHData run_sph_example(const SPHData& inputData, uint32_t nBitsPerAxis = 4, int dist = 0);