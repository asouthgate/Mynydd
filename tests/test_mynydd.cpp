#include <iostream>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <mynydd/mynydd.hpp>

TEST_CASE("Compute pipeline processes data for float", "[vulkan]") {
    std::cerr << "Starting compute pipeline test for float..." << std::endl;
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    std::cerr << "Created context ptr" << std::endl;

    size_t n = 1024;

    auto buffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr->device,
        contextPtr->physicalDevice,
        n * sizeof(float),
        false
    );

    auto dynamicResourcesPtr = mynydd::createDataResources<float>(contextPtr, buffer, n);
    std::cerr << "Created dynamicResources Ptr" << std::endl;

    mynydd::ComputeEngine<float> pipeline(contextPtr, dynamicResourcesPtr, "shaders/shader.comp.spv");
    std::cerr << "Created compute engine pipeline" << std::endl;
    std::vector<float> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }

    std::cerr << "Input data prepared with size: " << inputData.size() << std::endl;
    pipeline.uploadData(inputData);
    std::cerr << "Data uploaded to pipeline." << std::endl;
    pipeline.execute();
    std::cerr << "Pipeline executed." << std::endl;
    std::vector<float> output = pipeline.fetchData();
    std::cerr << "Data fetched from pipeline." << std::endl;
    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        REQUIRE(output[i] == Catch::Approx(1.0 / static_cast<float>(i)));
    }
    std::cerr << "Compute shader executed for 1.0/floats." << std::endl;
    SUCCEED("Compute shader executed for 1.0/floats.");
}

TEST_CASE("Compute pipeline processes data for double", "[vulkan]") {

    size_t n = 1024;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto buffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr->device,
        contextPtr->physicalDevice,
        n * sizeof(double),
        false
    );
    auto dynamicResourcesPtr = mynydd::createDataResources<double>(contextPtr, buffer, n);
    mynydd::ComputeEngine<double> pipeline(contextPtr, dynamicResourcesPtr, "shaders/shader_double.comp.spv");

    std::vector<double> inputData(n);
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

    size_t n = 1024;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();
    auto buffer = std::make_shared<mynydd::AllocatedBuffer>(
        contextPtr->device,
        contextPtr->physicalDevice,
        n * sizeof(TestData),
        false
    );
    auto dynamicResourcesPtr = mynydd::createDataResources<TestData, TestParams>(contextPtr, buffer, n);
    mynydd::ComputeEngine<TestData> compeng(contextPtr, dynamicResourcesPtr, "shaders/shader_uniform.comp.spv");

    std::vector<TestData> inputData(n);
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