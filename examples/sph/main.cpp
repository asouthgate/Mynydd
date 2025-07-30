#include <iostream>
#include <glm/glm.hpp>
#include <mynydd/mynydd.hpp>

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    float density;
    float pressure;
};

int main(int argc, char** argv) {
    std::cout << "Running SPH example..." << std::endl;

    mynydd::VulkanContext context = mynydd::createVulkanContext();
    std::shared_ptr<mynydd::VulkanContext> contextPtr = std::make_shared<mynydd::VulkanContext>(context);
    mynydd::VulkanDynamicResources dynamicResources = mynydd::createDataResources<Particle>(contextPtr, 1024);
    std::shared_ptr<mynydd::VulkanDynamicResources> dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(dynamicResources);
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

    compeng.uploadData(inputData);

    std::vector<Particle> output = compeng.execute();
    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = (" << output[i].position.x << "," << output[i].position.y << ")" << std::endl;
    }

}