#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <mynydd/mynydd.hpp>

TEST_CASE("Compute pipeline processes data for float", "[vulkan]") {
    mynydd::VulkanContext context = mynydd::createVulkanContext();
    std::shared_ptr<mynydd::VulkanContext> contextPtr = std::make_shared<mynydd::VulkanContext>(context);
    mynydd::VulkanDynamicResources dynamicResources = mynydd::createDataResources<float>(contextPtr, 1024);
    
    std::shared_ptr<mynydd::VulkanDynamicResources> dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(dynamicResources);

    mynydd::ComputeEngine<float> pipeline(contextPtr, dynamicResourcesPtr, "shaders/shader.comp.spv");

    std::vector<float> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }

    pipeline.uploadData(inputData);
    pipeline.execute();
    std::vector<float> output = pipeline.fetchData();
    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        REQUIRE(output[i] == Catch::Approx(1.0 / static_cast<float>(i)));
    }

    SUCCEED("Compute shader executed for 1.0/floats.");
}

TEST_CASE("Compute pipeline processes data for double", "[vulkan]") {
    mynydd::VulkanContext context = mynydd::createVulkanContext();
    std::shared_ptr<mynydd::VulkanContext> contextPtr = std::make_shared<mynydd::VulkanContext>(context);
    mynydd::VulkanDynamicResources dynamicResources = mynydd::createDataResources<double>(contextPtr, 1024);
    std::shared_ptr<mynydd::VulkanDynamicResources> dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(dynamicResources);

    mynydd::ComputeEngine<double> pipeline(contextPtr, dynamicResourcesPtr, "shaders/shader_double.comp.spv");

    std::vector<double> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<double>(i);
    }

    pipeline.uploadData(inputData);
    pipeline.execute();
    std::vector<double> output = pipeline.fetchData();
    for (size_t i = 0; i < std::min<size_t>(output.size(), 10); ++i) {
        REQUIRE(output[i] == static_cast<double>(i) * 2.0);
    }
    SUCCEED("Compute shader produced expected results for doubles * 2.");
}

struct TestData {
    glm::vec2 position;
};

struct TestParams {
    float val;
    float _pad[3];
};

TEST_CASE("Shader uniforms are correctly uploaded with test data", "[vulkan]") {

    TestParams params = {
        0.187777f
    };

    mynydd::VulkanContext context = mynydd::createVulkanContext();
    std::shared_ptr<mynydd::VulkanContext> contextPtr = std::make_shared<mynydd::VulkanContext>(context);
    mynydd::VulkanDynamicResources dynamicResources = mynydd::createDataResources<TestData, TestParams>(contextPtr, 1024);
    std::shared_ptr<mynydd::VulkanDynamicResources> dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(dynamicResources);
    mynydd::ComputeEngine<TestData> compeng(contextPtr, dynamicResourcesPtr, "shaders/shader_uniform.comp.spv");

    std::vector<TestData> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = TestData{
            glm::vec2(static_cast<float>(i % 32), static_cast<float>(i / 32)),
        };
    }

    compeng.uploadUniformData(params);
    compeng.uploadData(inputData);
    compeng.execute();
    std::vector<TestData> output = compeng.fetchData();

    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        REQUIRE(output[i].position.x == params.val);
        REQUIRE(output[i].position.y == params.val + i);
    }

}