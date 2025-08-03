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

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto dynamicResourcesPtr = mynydd::createDataResources<float>(contextPtr, 1024);
    mynydd::ComputeEngine<Particle> compeng(contextPtr, dynamicResourcesPtr, "examples/sph/shader.comp.spv");

    std::vector<Particle> inputData(1024);
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
    std::vector<Particle> output = compeng.fetchData();

    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = (" << output[i].position.x << "," << output[i].position.y << ")" << std::endl;
    }

}