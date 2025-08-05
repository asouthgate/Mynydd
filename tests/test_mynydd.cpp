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
    auto pipeline = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/shader.comp.spv", std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{input}

    );
    std::cerr << "Initialized ComputeEngine" << std::endl;
    std::vector<float> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }


    mynydd::uploadData<float>(contextPtr, inputData, input);
    std::cerr << "Uploaded data" << std::endl;
    // pipeline.execute(n);
    mynydd::executeBatch<float>(contextPtr, {pipeline}, n);
    std::cerr << "Executed" << std::endl;
    std::vector<float> out = mynydd::fetchData<float>(contextPtr, input, n);
    std::cerr << "Fetched" << std::endl;
    for (size_t i = 1; i < std::min<size_t>(out.size(), 10); ++i) {
        std::cerr << "Checking output for index " << i << ": " << out[i] << std::endl;
        REQUIRE(out[i] == Catch::Approx(1.0 / static_cast<float>(i)));
    }
    std::cerr << "Checked output" << std::endl;
    SUCCEED("Compute shader executed for 1.0/floats.");
}

TEST_CASE("Compute pipeline processes data for double", "[vulkan]") {

    size_t n = 512;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto input = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(double), false);
    auto output = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(double), false);

    mynydd::ComputeEngine<double> pipeline(contextPtr, "shaders/shader_double.comp.spv", {input, output});

    std::vector<double> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<double>(i);
    }

    mynydd::uploadData<double>(contextPtr, inputData, input);
    pipeline.execute(n);
    std::vector<double> out = mynydd::fetchData<double>(contextPtr, output, n);
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

    mynydd::ComputeEngine<TestData> compeng(contextPtr, "shaders/shader_uniform.comp.spv", {input, output, uniform});

    std::vector<TestData> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = TestData{
            glm::vec2(static_cast<float>(i % 32), static_cast<float>(i / 32)),
        };
    }

    mynydd::uploadUniformData<TestParams>(contextPtr, params, uniform);
    mynydd::uploadData<TestData>(contextPtr, inputData, input);
    compeng.execute(n);
    std::vector<TestData> outv = mynydd::fetchData<TestData>(contextPtr, output, n);

    for (size_t i = 1; i < std::min<size_t>(outv.size(), 10); ++i) {
        REQUIRE(outv[i].position.x == params.val);
        REQUIRE(outv[i].position.y == params.val + i);
    }

}