#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <vector>

struct dVec3Aln32 {
    alignas(32) glm::dvec3 data;
};

template <typename T>
struct RasterResult {
    std::vector<T> values;
};

template <typename T>
RasterResult<T> computeGridMean(
    const std::vector<dVec3Aln32>& pos,
    const std::vector<T>& data,
    double gridMin = 0.0,
    double gridMax = 1.0,
    std::size_t resolution = 10);

// template <typename T>
// void write_raster_to_hdf5(const std::vector<T>& raster,
//                           double gridMin,
//                           double gridMax,
//                           std::size_t resolution,
//                           const std::string& basepath,
//                           uint64_t iter);

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
    uint32_t n_boundary_tris;
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


SPHData simulate_inputs_box(uint32_t nParticles, double x0, double x1, double y0, double y1, double z0, double z1);
SPHData run_sph_example(const SPHData& inputData, SPHParams& inputParams, std::vector<glm::dvec3> boundary_vertices, uint iterations=1, std::string fname="", 
    bool debug_mode=false);