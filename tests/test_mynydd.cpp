#include <cstdint>
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

TEST_CASE("Push constants are passed to shader correctly", "[vulkan]") {
    size_t n = 512;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto outBuffer = std::make_shared<mynydd::Buffer>(contextPtr, n * sizeof(float), false);

    auto pipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "shaders/push_constants.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{outBuffer},
        256,
        1,
        1,
        std::vector<uint32_t>{sizeof(uint32_t)}
    );

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(contextPtr->commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer for batch execution.");
    }

    uint32_t x = 976;
    pipeline->setPushConstantsData(x, 0);

    mynydd::executeBatch(contextPtr, {pipeline}, false);

    std::vector<uint32_t> out = mynydd::fetchData<uint32_t>(contextPtr, outBuffer, n);
    for (size_t i = 0; i < n; ++i) {
        std::cerr << "Checking output for index " << i << ": " << out[i] << std::endl; 
        REQUIRE(out[i] == x);
    }
    SUCCEED("Compute shader push constants work as expected");
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

TEST_CASE("Compute pipeline processes multi-stage shader for doubles", "[vulkan]") {

    size_t n = 512;

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    
    auto input = std::make_shared<mynydd::Buffer>(contextPtr, n * sizeof(double), false);
    auto output = std::make_shared<mynydd::Buffer>(contextPtr, n * sizeof(double), false);

    auto pipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr,
        "shaders/shader_barrier.comp.spv", 
        std::vector<std::shared_ptr<mynydd::Buffer>>{input, output},
        256
    );

    std::vector<double> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<double>(i);
    }

    mynydd::uploadData<double>(contextPtr, inputData, input);
    mynydd::executeBatch(contextPtr, {pipeline});
    std::vector<double> out = mynydd::fetchData<double>(contextPtr, output, n);
    for (size_t i = 0; i < n - 1; ++i) {
        // this is expected to be correct except at workgroup boundaries
        // because the shader uses a barrier which only synchronises within a workgroup
        // std::cerr << "Checking output for index " << i << ": " << out[i] << std::endl;
        if (i % 64 == 63) {
            // at workgroup boundaries, the value is not computed correctly
            REQUIRE(out[i] == static_cast<double>(i) * 2.0);
        } else {
            // otherwise, it should be the sum of the two previous values
            REQUIRE(out[i] == static_cast<double>(i) * 2.0 + 
                     static_cast<double>(i + 1) * 2.0);
        }
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

TEST_CASE("A three-step sequence of pipelines produce expected outputs, using 3 buffers", "[vulkan]") {
    auto contextPtr = std::make_shared<mynydd::VulkanContext>();    

    size_t n = 1024;

    auto b1 = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(float), false);
    auto b2 = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(float), false);
    auto b3 = std::make_shared<mynydd::AllocatedBuffer>(contextPtr, n * sizeof(float), false);

    // Pipeline 1 takes b1, writes output to b2
    auto pipeline1 = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/multistep_1.comp.spv", std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{b1, b2}

    );

    // Pipeline 2 takes b2, writes output to b3
    auto pipeline2 = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/multistep_2.comp.spv", std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{b2, b3}
    );

    // Pipeline 3 takes b2 and b3, and writes to b1
    // Why reuse b1? For simplicity, but also to test that fences are working correctly
    // b1 should be free for use after pipeline2 has run
    auto pipeline3 = std::make_shared<mynydd::ComputeEngine<float>>(
        contextPtr, "shaders/multistep_3.comp.spv", std::vector<std::shared_ptr<mynydd::AllocatedBuffer>>{b2, b3, b1}
    );

    // Initialize some input data to upload to b1
    std::vector<float> inputData(n);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }
    mynydd::uploadData<float>(contextPtr, inputData, b1);

    // Execute the pipelines in a batch
    mynydd::executeBatch<float>(contextPtr, {pipeline1, pipeline2, pipeline3}, n);

    // Now fetch the outputs and check they are expected
    std::vector<float> out = mynydd::fetchData<float>(contextPtr, b3, n);
    for (size_t i = 1; i < std::min<size_t>(out.size(), n); ++i) {
        REQUIRE(out[i] == Catch::Approx(1.0 + 2.0 * static_cast<float>(i)));
    }

    std::vector<float> out2 = mynydd::fetchData<float>(contextPtr, b2, n);
    for (size_t i = 1; i < std::min<size_t>(out.size(), n); ++i) {
        REQUIRE(out2[i] == Catch::Approx(2.0 * static_cast<float>(i)));
    }


    std::vector<float> out3 = mynydd::fetchData<float>(contextPtr, b1, n);
    for (size_t i = 1; i < std::min<size_t>(out3.size(), n); ++i) {
        REQUIRE(out3[i] == Catch::Approx(
            0.6 * (out[i]) + 0.3 * out2[i])
        );
    }

    SUCCEED("Compute shader executed for 1.0/floats.");
}