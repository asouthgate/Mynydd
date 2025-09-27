#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

struct CellTriangles {
    uint32_t left;
    uint32_t right; // exclusive
};

// build a mapping from cell to list of triangles that intersect
std::vector<std::vector<uint32_t>> buildCellToTriangles(
    const std::vector<glm::dvec3>& vertices, // triangles packed in 3s
    double h,
    const glm::dvec3& domainMin,
    glm::ivec3 gridDims
);

// We won't just want to query triangles intersecting our cell, but all those adjacent
void accumulateNeighbors(
    std::vector<std::vector<uint32_t>>& cellToTris,
    glm::ivec3 gridDims
);

// Take vector of vector indexes, flatten it, and create indexes
void packForGPU(
    const std::vector<std::vector<uint32_t>>& cellToTris,
    std::vector<uint32_t>& flatTriIndices,
    std::vector<CellTriangles>& cellMeta
);

std::vector<glm::dvec3> get_test_boundary_mesh();