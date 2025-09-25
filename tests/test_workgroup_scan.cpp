#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <memory>
#include <vector>

#include <mynydd/mynydd.hpp>

TEST_CASE("Test that workgroup scan works on the single work group case", "[vulkan]") {
    const uint32_t groupCount = 1; // number of workgroups in original histogram (rows)
    const uint32_t numBins = 8;    // number of bins (cols)

    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto histBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, groupCount * numBins * sizeof(uint32_t), false);
    auto prefixBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, numBins * groupCount * sizeof(uint32_t), false);

    struct PrefixParams { uint32_t groupCount; uint32_t numBins; };
    PrefixParams pparams{ groupCount, numBins };
    auto pUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);

    std::vector<uint32_t> perGroupHist(groupCount * numBins);
    for (uint32_t g = 0; g < groupCount; ++g) {
        for (uint32_t b = 0; b < numBins; ++b) {
            perGroupHist[g * numBins + b] = (g + 1) * (b + 1); // arbitrary non-uniform pattern
        }
    }

    std::vector<uint32_t> cpuPrefix(numBins * groupCount);
    for (uint32_t g = 0; g < groupCount; ++g) {
        uint32_t sum = 0;
        for (uint32_t b = 0; b < numBins; ++b) {
            cpuPrefix[b * groupCount + g] = sum;
            sum += perGroupHist[g * numBins + b];
        }
    }

    // Upload hist to GPU (we will transpose it with the transpose shader)
    mynydd::uploadData<uint32_t>(contextPtr, perGroupHist, histBuffer);
    mynydd::uploadUniformData<PrefixParams>(contextPtr, pparams, pUniform);

    auto prefixPipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{histBuffer, prefixBuffer, pUniform},
        groupCount
    );

    mynydd::executeBatch(contextPtr, {prefixPipeline});

    std::vector<uint32_t> gpuPrefix = mynydd::fetchData<uint32_t>(contextPtr, prefixBuffer, numBins * groupCount);

    for (size_t i = 0; i < cpuPrefix.size(); ++i) {
        REQUIRE(gpuPrefix[i] == cpuPrefix[i]);
    }
}

TEST_CASE("Transpose + per-row prefix compute correct per-workgroup prefix sums", "[vulkan]") {
    // NOTE: IMPORTANT: THIS IS PRE-TRANSPOSE
    // In this case, since we transpose with groupCount of 4, we do a prefix sum over 4 values
    // Not over 8 values; we are doing it over a transposed matrix
    const uint32_t groupCount = 4; // number of workgroups in original histogram (rows)
    const uint32_t numBins = 8;    // number of bins (cols)
    const uint32_t groupCount_postinv = numBins; // number of workgroups in original histogram (rows)
    const uint32_t numBins_postinv = groupCount;    // number of bins (cols)


    auto contextPtr = std::make_shared<mynydd::VulkanContext>();

    auto histBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, groupCount * numBins * sizeof(uint32_t), false);
    auto transposedBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, numBins * groupCount * sizeof(uint32_t), false);
    auto prefixBuffer = std::make_shared<mynydd::Buffer>(
        contextPtr, numBins * groupCount * sizeof(uint32_t), false);

    // Uniform buffers for transpose and prefix shaders
    struct TransposeParams { uint32_t height; uint32_t width; }; // width=numBins, height=groupCount
    TransposeParams tparams{ groupCount, numBins };
    auto tUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(TransposeParams), true);

    struct PrefixParams { uint32_t groupCount; uint32_t numBins; };
    
    PrefixParams pparams{ groupCount_postinv, numBins_postinv };
    auto pUniform = std::make_shared<mynydd::Buffer>(contextPtr, sizeof(PrefixParams), true);

    // Prepare a deterministic, arbitrary per-workgroup histogram (group-major)
    // Example layout (group0 row, group1 row, ...):
    // We'll choose values so we can easily compute expected prefixes.
    std::vector<uint32_t> perGroupHist(groupCount * numBins);
    // fill with a pattern that's easy to reason about, e.g. histogram[g][b] = (g+1) * (b+1)
    for (uint32_t g = 0; g < groupCount; ++g) {
        for (uint32_t b = 0; b < numBins; ++b) {
            perGroupHist[g * numBins + b] = (g + 1) * (b + 1); // arbitrary non-uniform pattern
        }
    }

    // CPU reference: compute transposed and exclusive prefix per row (row = bin)
    std::vector<uint32_t> cpuTransposed(numBins * groupCount);
    for (uint32_t b = 0; b < numBins; ++b) {
        for (uint32_t g = 0; g < groupCount; ++g) {
            cpuTransposed[b * groupCount + g] = perGroupHist[g * numBins + b];
        }
    }

    std::vector<uint32_t> cpuPrefix(numBins * groupCount);
    for (uint32_t b = 0; b < numBins; ++b) {
        uint32_t sum = 0;
        for (uint32_t g = 0; g < groupCount; ++g) {
            cpuPrefix[b * groupCount + g] = sum;
            sum += cpuTransposed[b * groupCount + g];
        }
    }

    // Upload hist to GPU (we will transpose it with the transpose shader)
    mynydd::uploadData<uint32_t>(contextPtr, perGroupHist, histBuffer);
    mynydd::uploadUniformData<TransposeParams>(contextPtr, tparams, tUniform);
    mynydd::uploadUniformData<PrefixParams>(contextPtr, pparams, pUniform);

    // Dispatch transpose: using 2D workgroups (tileSize=16)
    const uint32_t tile = 16;
    uint32_t groupCountX = (tparams.width + tile - 1) / tile;  // ceil(numBins/16)
    uint32_t groupCountY = (tparams.height + tile - 1) / tile; // ceil(groupCount/16)

    // Create pipelines:
    //  - transpose: in = histBuffer (groupCount x numBins), out = transposedBuffer (numBins x groupCount), params tUniform
    //  - prefix: in = transposedBuffer, out = prefixBuffer, params pUniform
    auto transposePipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr, "shaders/transpose.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{histBuffer, transposedBuffer, tUniform},
        (tparams.width * tparams.height + 256) / 256
    );

    auto prefixPipeline = std::make_shared<mynydd::PipelineStep>(
        contextPtr, "shaders/workgroup_scan.comp.spv",
        std::vector<std::shared_ptr<mynydd::Buffer>>{transposedBuffer, prefixBuffer, pUniform},
        groupCount_postinv
    );

    // NOTE: if your executeBatch currently only supports a single groupCount argument,
    // you will need to extend it to allow a 2D dispatch or flatten the transpose to 1D.
    // Here we assume an overload executeBatch that accepts (pipelineList, gx, gy, gz).
    mynydd::executeBatch(contextPtr, {transposePipeline, prefixPipeline});

    // Dispatch prefix: one workgroup per bin (1D). local_size_x = 256; only thread 0 does the scan.
    // We dispatch numBins workgroups in X
    // Fetch GPU outputs
    std::vector<uint32_t> gpuTransposed = mynydd::fetchData<uint32_t>(contextPtr, transposedBuffer, numBins * groupCount);
    std::vector<uint32_t> gpuPrefix = mynydd::fetchData<uint32_t>(contextPtr, prefixBuffer, numBins * groupCount);

    // Verify transposed matches CPU transposed
    for (size_t i = 0; i < cpuTransposed.size(); ++i) {
        REQUIRE(gpuTransposed[i] == cpuTransposed[i]);
    }
    // Verify prefix matches CPU prefix
    for (size_t i = 0; i < cpuPrefix.size(); ++i) {
        REQUIRE(gpuPrefix[i] == cpuPrefix[i]);
    }
}