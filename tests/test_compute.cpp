#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>
#include <memory>
#include "../src/compute_context.hpp"

TEST_CASE("Compute pipeline processes data for float", "[vulkan]") {
    mylib::VulkanContext context = mylib::createVulkanContext();
    std::shared_ptr<mylib::VulkanContext> contextPtr = std::make_shared<mylib::VulkanContext>(context);
    mylib::ComputePipeline<float> pipeline(contextPtr, "shaders/shader.comp.spv");

    std::vector<float> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }

    pipeline.createDynamicResources(1024);
    pipeline.uploadData(inputData);
    std::vector<float> output = pipeline.execute();
    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
        REQUIRE(output[i] == Catch::Approx(1.0 / static_cast<float>(i)));
    }

    SUCCEED("Compute shader executed for 1.0/floats.");
}

TEST_CASE("Compute pipeline processes data for double", "[vulkan]") {
    mylib::VulkanContext context = mylib::createVulkanContext();
    std::shared_ptr<mylib::VulkanContext> contextPtr = std::make_shared<mylib::VulkanContext>(context);
    mylib::ComputePipeline<double> pipeline(contextPtr, "shaders/shader_double.comp.spv");

    std::vector<double> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<double>(i);
    }

    pipeline.createDynamicResources(1024);
    pipeline.uploadData(inputData);
    std::vector<double> output = pipeline.execute();
    for (size_t i = 0; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
        REQUIRE(output[i] == static_cast<double>(i) * 2.0);
    }
    SUCCEED("Compute shader produced expected results for doubles * 2.");
}

