#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

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
    pipeline.execute();

    SUCCEED("Compute shader executed without crashing.");
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
    pipeline.execute();

    SUCCEED("Compute shader executed without crashing.");
}

