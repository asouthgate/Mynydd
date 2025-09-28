#include <vector>
#include <cstdint>
#include <algorithm>
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include "mesh.hpp"


void normalizeToUnitCube(std::vector<glm::dvec3>& vertices) {
    if (vertices.empty()) return;

    glm::dvec3 minV = vertices[0];
    glm::dvec3 maxV = vertices[0];

    // Find bounding box
    for (const auto& v : vertices) {
        minV = glm::min(minV, v);
        maxV = glm::max(maxV, v);
    }

    glm::dvec3 size = maxV - minV;
    double maxExtent = std::max({size.x, size.y, size.z});

    if (maxExtent <= 0.0) return; // Degenerate mesh

    // Normalize: shift to origin, scale to fit [0,1]
    for (auto& v : vertices) {
        v = (v - minV) / maxExtent;
    }
}


std::vector<glm::dvec3> loadObjAsTriangles(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file: " + filename);

    std::vector<glm::dvec3> positions;
    std::vector<glm::dvec3> triangles;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue; // skip comments

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            positions.emplace_back(x, y, z);
        }
        else if (type == "f") {
            std::vector<int> indices;
            std::string vert;
            while (iss >> vert) {
                std::istringstream viss(vert);
                std::string idxStr;
                std::getline(viss, idxStr, '/'); // only care about vertex index
                indices.push_back(std::stoi(idxStr) - 1); // OBJ is 1-based
            }

            if (indices.size() < 3) continue;

            // Triangulate polygon (fan method)
            for (size_t i = 1; i + 1 < indices.size(); i++) {
                triangles.push_back(positions[indices[0]]);
                triangles.push_back(positions[indices[i]]);
                triangles.push_back(positions[indices[i+1]]);
            }
        }
    }

    return triangles;
}


std::vector<std::vector<uint32_t>> buildCellToTriangles(
    const std::vector<glm::dvec3>& vertices, // triangles packed in 3s
    const glm::dvec3& domainMin,
    const glm::dvec3& domainMax,
    glm::ivec3 gridDims
) {
    size_t nCells = static_cast<size_t>(gridDims.x) * gridDims.y * gridDims.z;
    std::vector<std::vector<uint32_t>> cellToTris(nCells);

    // Compute per-dimension cell size (non-uniform possible)
    glm::dvec3 h = (domainMax - domainMin) / glm::dvec3(gridDims);

    uint32_t nTris = static_cast<uint32_t>(vertices.size() / 3);

    for (uint32_t t = 0; t < nTris; ++t) {
        glm::dvec3 v0 = vertices[t * 3 + 0];
        glm::dvec3 v1 = vertices[t * 3 + 1];
        glm::dvec3 v2 = vertices[t * 3 + 2];

        glm::dvec3 triMin = glm::min(glm::min(v0, v1), v2);
        glm::dvec3 triMax = glm::max(glm::max(v0, v1), v2);

        glm::ivec3 minCell = glm::clamp(
            glm::floor((triMin - domainMin) / h),
            glm::dvec3(0), glm::dvec3(gridDims - 1)
        );
        glm::ivec3 maxCell = glm::clamp(
            glm::floor((triMax - domainMin) / h),
            glm::dvec3(0), glm::dvec3(gridDims - 1)
        );

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


std::vector<glm::dvec3> get_test_boundary_mesh() {

    // Unit cube triangles (6 faces × 2 triangles per face)
    std::vector<glm::dvec3> boundary_vertices = {
        // -Z face
        glm::dvec3(0,0,0), glm::dvec3(1,0,0), glm::dvec3(1,1,0),
        glm::dvec3(0,0,0), glm::dvec3(1,1,0), glm::dvec3(0,1,0),

        // +Z face
        glm::dvec3(0,0,1), glm::dvec3(1,0,1), glm::dvec3(1,1,1),
        glm::dvec3(0,0,1), glm::dvec3(1,1,1), glm::dvec3(0,1,1),

        // -X face
        glm::dvec3(0,0,0), glm::dvec3(0,0,1), glm::dvec3(0,1,1),
        glm::dvec3(0,0,0), glm::dvec3(0,1,1), glm::dvec3(0,1,0),

        // +X face
        glm::dvec3(1,0,0), glm::dvec3(1,0,1), glm::dvec3(1,1,1),
        glm::dvec3(1,0,0), glm::dvec3(1,1,1), glm::dvec3(1,1,0),

        // -Y face
        glm::dvec3(0,0,0), glm::dvec3(1,0,0), glm::dvec3(1,0,1),
        glm::dvec3(0,0,0), glm::dvec3(1,0,1), glm::dvec3(0,0,1),

        // +Y face
        glm::dvec3(0,1,0), glm::dvec3(1,1,0), glm::dvec3(1,1,1),
        glm::dvec3(0,1,0), glm::dvec3(1,1,1), glm::dvec3(0,1,1)
    };

    // Small cube parameters
    double cube_min_x = 0.9; // touching +X face of unit cube
    double cube_max_x = 1.0;
    double cube_min_y = 0.45;
    double cube_max_y = 0.55; // edge length 0.1
    double cube_min_z = 0.0;
    double cube_max_z = 0.1; // edge length 0.1

    boundary_vertices.insert(
        boundary_vertices.end(),
        {
            // -X face
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_min_x, cube_max_y, cube_min_z), glm::dvec3(cube_min_x, cube_max_y, cube_max_z),
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_min_x, cube_max_y, cube_max_z), glm::dvec3(cube_min_x, cube_min_y, cube_max_z),

            // +X face
            glm::dvec3(cube_max_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z),
            glm::dvec3(cube_max_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z), glm::dvec3(cube_max_x, cube_min_y, cube_max_z),

            // -Y face
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_min_y, cube_max_z),
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_min_y, cube_max_z), glm::dvec3(cube_min_x, cube_min_y, cube_max_z),

            // +Y face
            glm::dvec3(cube_min_x, cube_max_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z),
            glm::dvec3(cube_min_x, cube_max_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z), glm::dvec3(cube_min_x, cube_max_y, cube_max_z),

            // -Z face
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_min_z),
            glm::dvec3(cube_min_x, cube_min_y, cube_min_z), glm::dvec3(cube_max_x, cube_max_y, cube_min_z), glm::dvec3(cube_min_x, cube_max_y, cube_min_z),

            // +Z face
            glm::dvec3(cube_min_x, cube_min_y, cube_max_z), glm::dvec3(cube_max_x, cube_min_y, cube_max_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z),
            glm::dvec3(cube_min_x, cube_min_y, cube_max_z), glm::dvec3(cube_max_x, cube_max_y, cube_max_z), glm::dvec3(cube_min_x, cube_max_y, cube_max_z)
        }
    );

    return boundary_vertices;
}