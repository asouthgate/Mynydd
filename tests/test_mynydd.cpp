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

    size_t n = 1024;

    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(float), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(float), false);
    auto uniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(float), true);

    mynydd::ComputeEngine<float> pipeline(contextPtr, "shaders/shader.comp.spv", input, output, uniform);
    std::cerr << "Initialized ComputeEngine" << std::endl;
    std::vector<float> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }


    pipeline.uploadData(inputData);
    std::cerr << "Uploaded data" << std::endl;
    pipeline.execute();
    std::cerr << "Executed" << std::endl;
    std::vector<float> out = pipeline.fetchData();
    std::cerr << "Fetched" << std::endl;
    for (size_t i = 1; i < std::min<size_t>(out.size(), 10); ++i) {
        REQUIRE(out[i] == Catch::Approx(1.0 / static_cast<float>(i)));
    }
    SUCCEED("Compute shader executed for 1.0/floats.");
}

TEST_CASE("Compute pipeline processes data for double", "[vulkan]") {

    size_t n = 512;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(double), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(double), false);
    auto uniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(double), true);

    mynydd::ComputeEngine<double> pipeline(contextPtr, "shaders/shader_double.comp.spv", input, output, uniform);

    std::vector<double> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<double>(i);
    }

    pipeline.uploadData(inputData);
    pipeline.execute();
    std::vector<double> out = pipeline.fetchData();
    for (size_t i = 0; i < std::min<size_t>(out.size(), 10); ++i) {
        REQUIRE(out[i] == static_cast<double>(i) * 2.0);
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
    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(TestData), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(TestData), false);
    auto uniform = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, sizeof(TestParams), true);

    mynydd::ComputeEngine<TestData> compeng(contextPtr, "shaders/shader_uniform.comp.spv", input, output, uniform);

    std::vector<TestData> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = TestData{
            glm::vec2(static_cast<float>(i % 32), static_cast<float>(i / 32)),
        };
    }

    compeng.uploadUniformData(params);
    compeng.uploadData(inputData);
    compeng.execute();
    std::vector<TestData> outv = compeng.fetchData();

    for (size_t i = 1; i < std::min<size_t>(outv.size(), 10); ++i) {
        REQUIRE(outv[i].position.x == params.val);
        REQUIRE(outv[i].position.y == params.val + i);
    }

}