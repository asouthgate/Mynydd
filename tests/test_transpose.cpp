#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <mynydd/mynydd.hpp>

TEST_CASE("Transpose shader correctly transposes arbitrary matrix", "[transpose]") {
    const uint32_t m = 35;  // rows of input
    const uint32_t n = 27;  // cols of input

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    // Create input and output buffers (uint32_t)
    auto inputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, m * n * sizeof(uint32_t), false);
    auto outputBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, n * m * sizeof(uint32_t), false);

    // Create uniform buffer for Params
    struct Params {
        uint32_t m;
        uint32_t n;
    } params{m, n};

    auto uniformBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, sizeof(Params), true);

    // Generate test data: input matrix with known pattern
    std::vector<uint32_t> inputData(m * n);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            inputData[row * n + col] = row * 1000 + col;  // unique per element
        }
    }

    // Upload data and params
    mynydd::uploadData<uint32_t>(contextPtr, inputData, inputBuffer);
    mynydd::uploadUniformData<Params>(contextPtr, params, uniformBuffer);

    // Load transpose compute shader pipeline
    auto pipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr, "shaders/transpose.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{inputBuffer, outputBuffer, uniformBuffer},
        (m * n + 256) / 256
    );


    // Run the shader
    mynydd::executeBatch(contextPtr, {pipeline});

    // Fetch output
    std::vector<uint32_t> outData = mynydd::fetchData<uint32_t>(contextPtr, outputBuffer, n * m);

    // Verify output is transpose of input
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t expected = inputData[row * n + col];
            uint32_t actual = outData[col * m + row];
            REQUIRE(actual == expected);
        }
    }
}