#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>
#include <yaml-cpp/yaml.h>

#include "mesh.hpp"
#include "sph.hpp"

void printSPHDataCSV(const SPHData& data) {
    std::cout << "density,pressure,fpx,fpy,fpz,x,y,z,morton_key\n";
    for (size_t i = 0; i < data.mortonKeys.size(); ++i) {
        const auto& pos = data.positions[i];
        double density = data.densities[i];
        uint32_t key = data.mortonKeys[i];
        std::cout << density << "," << data.pressures[i] << "," << data.pressureForces[i].data.x << ","
                    << data.pressureForces[i].data.y << "," << data.pressureForces[i].data.z << ","
                  << pos.data.x << "," << pos.data.y << "," << pos.data.z << ","
                  << key << "\n";
    }
}

int main(int argc, char** argv) {


    auto config_fname = argv[1];
    YAML::Node config = YAML::LoadFile(config_fname);

    uint32_t nParticles = config["nParticles"].as<uint32_t>();
    uint32_t nBitsPerAxis = config["nBitsPerAxis"].as<uint32_t>();
    uint32_t niterations = config["niterations"].as<uint32_t>();
    uint32_t rasterres = config["raster_resolution"].as<uint32_t>();
    double dt = config["dt"].as<double>();
    double c2 = config["c2"].as<double>();
    double mu = config["mu"].as<double>();
    double fgrav = config["fgrav"].as<double>();

    double x0 = config["x0"].as<double>();
    double x1 = config["x1"].as<double>();
    double y0 = config["y0"].as<double>();
    double y1 = config["y1"].as<double>();
    double z0 = config["z0"].as<double>();
    double z1 = config["z1"].as<double>();

    std::string mesh_path = config["mesh_path"].as<std::string>();

    double vol_prop = (fabs(x1 - x0) * fabs(y1 - y0) * fabs(z1 - z0));
    double rho0_mod = 1.0 / vol_prop;

    auto simulated = simulate_inputs_box(nParticles, x0, x1, y0, y1, z0, z1);
    double h = 1.0 / (1 << nBitsPerAxis);

    double rho0 = nParticles * rho0_mod;

    std::cerr << "Running SPH with " << nParticles << " particles, " << nBitsPerAxis << " bits per axis, "
                << niterations << " iterations, dt=" << dt << ", rho0=" << rho0 << ", c2=" << c2 << std::endl;

    std::cerr << "./sph_example " << nParticles << " " << nBitsPerAxis << " " << niterations << " "
                << dt << " " << c2 << " " << mu << " " << fgrav 
                << " " << x0 << " " << x1
                << " " << y0 << " " << y1
                << " " << z0 << " " << z1
                << "|" << std::endl;

    std:: cerr << "Raster resolution is " << rasterres << std::endl;

    SPHParams params {
        nBitsPerAxis,
        nParticles,
        glm::dvec3(0.0),
        glm::dvec3(1.0),
        1,
        dt,
        h,
        1.0,
        glm::dvec3(0.0, 0.0, fgrav),
        rho0,
        c2,
        mu
    };

    auto mesh = loadObjAsTriangles(mesh_path);
    normalizeToUnitCube(mesh);
    auto outputs = run_sph_example(
        simulated, params, mesh, niterations, "main_example_output", false
    );

    // printSPHDataCSV(outputs);

}