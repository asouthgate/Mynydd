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

    auto buffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr->device,
        contextPtr->physicalDevice,
        n * sizeof(Particle),
        false
    );
    
    auto dynamicResourcesPtr = mynydd::createDataResources<float>(contextPtr, buffer, n);
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
    std::vector<Particle> output = compeng.fetchData();

    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = (" << output[i].position.x << "," << output[i].position.y << ")" << std::endl;
    }

}