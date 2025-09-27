#include <vector>
#include <cstdint>
#include <algorithm>
#include <glm/glm.hpp>
#include "mesh.hpp"

std::vector<std::vector<uint32_t>> buildCellToTriangles(
    const std::vector<glm::dvec3>& vertices, // triangles packed in 3s
    double h,
    const glm::dvec3& domainMin,
    glm::ivec3 gridDims
) {
    size_t nCells = static_cast<size_t>(gridDims.x) * gridDims.y * gridDims.z;
    std::vector<std::vector<uint32_t>> cellToTris(nCells);

    uint32_t nTris = static_cast<uint32_t>(vertices.size() / 3);

    for (uint32_t t = 0; t < nTris; ++t) {
        glm::dvec3 v0 = vertices[t * 3 + 0];
        glm::dvec3 v1 = vertices[t * 3 + 1];
        glm::dvec3 v2 = vertices[t * 3 + 2];

        glm::dvec3 triMin = glm::min(glm::min(v0, v1), v2);
        glm::dvec3 triMax = glm::max(glm::max(v0, v1), v2);

        glm::ivec3 minCell = glm::clamp(glm::floor((triMin - domainMin) / h),
                                        glm::dvec3(0), glm::dvec3(gridDims - 1));
        glm::ivec3 maxCell = glm::clamp(glm::floor((triMax - domainMin) / h),
                                        glm::dvec3(0), glm::dvec3(gridDims - 1));

        for (int x = minCell.x; x <= maxCell.x; ++x) {
            for (int y = minCell.y; y <= maxCell.y; ++y) {
                for (int z = minCell.z; z <= maxCell.z; ++z) {
                    uint32_t ak = (z * gridDims.y + y) * gridDims.x + x;
                    cellToTris[ak].push_back(t);
                }
            }
        }
    }

    return cellToTris;
}

void accumulateNeighbors(
    std::vector<std::vector<uint32_t>>& cellToTris,
    glm::ivec3 gridDims
) {
    auto copy = cellToTris;

    for (int x = 0; x < gridDims.x; ++x) {
        for (int y = 0; y < gridDims.y; ++y) {
            for (int z = 0; z < gridDims.z; ++z) {
                uint32_t ak = (z * gridDims.y + y) * gridDims.x + x;

                std::vector<uint32_t> expanded = copy[ak];

                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            int nx = x + dx;
                            int ny = y + dy;
                            int nz = z + dz;

                            if (nx < 0 || ny < 0 || nz < 0 ||
                                nx >= gridDims.x || ny >= gridDims.y || nz >= gridDims.z) {
                                continue;
                            }

                            uint32_t neighborAk = (nz * gridDims.y + ny) * gridDims.x + nx;
                            expanded.insert(expanded.end(),
                                            copy[neighborAk].begin(),
                                            copy[neighborAk].end());
                        }
                    }
                }

                std::sort(expanded.begin(), expanded.end());
                expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());

                cellToTris[ak] = std::move(expanded);
            }
        }
    }
}

void packForGPU(
    const std::vector<std::vector<uint32_t>>& cellToTris,
    std::vector<uint32_t>& flatTriIndices,
    std::vector<CellTriangles>& cellMeta
) {
    flatTriIndices.clear();
    cellMeta.resize(cellToTris.size());

    uint32_t cursor = 0;

    for (size_t ak = 0; ak < cellToTris.size(); ++ak) {
        const auto& tris = cellToTris[ak];
        cellMeta[ak].left  = cursor;
        cellMeta[ak].right = cursor + static_cast<uint32_t>(tris.size());

        flatTriIndices.insert(flatTriIndices.end(), tris.begin(), tris.end());
        cursor += static_cast<uint32_t>(tris.size());
    }
}
