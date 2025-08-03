#include <iostream>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    float density;
    float pressure;
};

struct Params {
    float dt;
    float _pad[3];
};

int main(int argc, char** argv) {

    Params params = {
        0.187777f
    };

    std::cout << "Running SPH example..." << std::endl;

    size_t n = 1024;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    

    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(Particle), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(Particle), false);
    auto uniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(Params), true);

    auto dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(
        contextPtr,
        input,
        output,
        uniform
    );
    mynydd::ComputeEngine<Particle> compeng(contextPtr, dynamicResourcesPtr, "examples/sph/shader.comp.spv");

    std::vector<Particle> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = Particle{
            glm::vec2(static_cast<float>(i % 32), static_cast<float>(i / 32)),
            glm::vec2(0.0f, 0.0f),
            2.0f,
            0.0f
        };
    }

    compeng.uploadUniformData(params);
    compeng.uploadData(inputData);
    compeng.execute();
    std::vector<Particle> out = compeng.fetchData();

    for (size_t i = 1; i < std::min<size_t>(out.size(), 10); ++i) {
        std::cout << "output[" << i << "] = (" << out[i].position.x << "," << out[i].position.y << ")" << std::endl;
    }

}