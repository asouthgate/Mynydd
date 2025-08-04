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
    mynydd::ComputeEngine<Particle> compeng(contextPtr, "examples/sph/shader.comp.spv", {input, output, uniform});

    std::vector<Particle> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = Particle{
            glm::vec2(static_cast<float>(i % 32), static_cast<float>(i / 32)),
            glm::vec2(0.0f, 0.0f),
            2.0f,
            0.0f
        };
    }
    mynydd::uploadUniformData<Params>(contextPtr, params, uniform);
    mynydd::uploadData<Particle>(contextPtr, inputData, input);
    compeng.execute(n);
    std::vector<Particle> out = mynydd::fetchData<Particle>(contextPtr, output, n);

    for (size_t i = 1; i < std::min<size_t>(out.size(), 10); ++i) {
        std::cout << "output[" << i << "] = (" << out[i].position.x << "," << out[i].position.y << ")" << std::endl;
    }

}