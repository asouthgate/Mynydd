#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct dVec3Aln32 {
    alignas(32) glm::dvec3 data;
};

struct SPHParams {
    uint32_t nBits;
    uint32_t nParticles;
    alignas(32) glm::dvec3 domainMin; // alignas required for silly alignment issues
    alignas(32) glm::dvec3 domainMax;
    int dist;
    double dt;
    double h;
    double mass;
    alignas(32) glm::dvec3 gravity;
    double rho0;
    double c2;
    double mu;
};


struct SPHData {
    std::vector<double> densities;
    std::vector<double> pressures;
    std::vector<dVec3Aln32> pressureForces;
    std::vector<dVec3Aln32> positions;
    std::vector<dVec3Aln32> velocities;
    std::vector<uint32_t> mortonKeys;
    std::vector<uint32_t> sortedIndices;
    std::vector<mynydd::CellInfo> cellInfos;
    std::vector<dVec3Aln32> newPositions;
    std::vector<dVec3Aln32> newVelocities;
};

SPHData simulate_inputs(uint32_t nParticles, double min = 0.0, double max = 1.0);
SPHData simulate_inputs_uniform(uint32_t nParticles, double jitter = 0.01);

SPHData run_sph_example(const SPHData& inputData, SPHParams& inputParams, uint iterations=1, std::string fname="", bool debug_mode=false);