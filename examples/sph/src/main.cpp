#include <cstdint>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>
#include <mynydd/pipelines/particle_index.hpp>

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

    uint32_t nParticles = 5000;
    uint32_t nBitsPerAxis = 4;
    uint32_t niterations = 10000;
    double dt = 0.003;
    double c2 = 0.01;
    double mu = 0.001;
    double fgrav = -1.0;

    double x0 = 0.0;
    double x1 = 1.0;
    double y0 = 0.0;
    double y1 = 1.0;
    double z0 = 0.8;
    double z1 = 1.0;


    if (argc > 2) {
        nParticles = static_cast<uint32_t>(std::atoi(argv[1]));
        nBitsPerAxis = static_cast<uint32_t>(std::atoi(argv[2]));
        niterations = static_cast<uint32_t>(std::atoi(argv[3]));
        dt = std::atof(argv[4]);
        c2 = std::atof(argv[5]);
        mu = std::atof(argv[6]);
        fgrav = std::atof(argv[7]);
        x0 = std::atof(argv[8]);
        x1 = std::atof(argv[9]);
        y0 = std::atof(argv[10]);
        y1 = std::atof(argv[11]);
        z0 = std::atof(argv[12]);
        z1 = std::atof(argv[13]);

    }

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

    auto mesh = loadObjAsTriangles("../examples/sph/assets/example_mesh.obj");
    normalizeToUnitCube(mesh);
    auto outputs = run_sph_example(simulated, params, mesh, niterations, "main_example_output");

    // printSPHDataCSV(outputs);

}