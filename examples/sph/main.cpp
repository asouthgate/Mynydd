#include <iostream>
#include <mynydd/mynydd.hpp>

int main(int argc, char** argv) {
    std::cout << "Running SPH example..." << std::endl;

    mynydd::VulkanContext context = mynydd::createVulkanContext();
    std::shared_ptr<mynydd::VulkanContext> contextPtr = std::make_shared<mynydd::VulkanContext>(context);
    mynydd::VulkanDynamicResources dynamicResources = mynydd::createDataResources<float>(contextPtr, 1024);
    std::shared_ptr<mynydd::VulkanDynamicResources> dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(dynamicResources);
    mynydd::ComputeEngine<float> compeng(contextPtr, dynamicResourcesPtr, "shader.comp.spv");

    std::vector<float> inputData(1024);
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i);
    }

    compeng.uploadData(inputData);


    std::vector<float> output = compeng.execute();
    for (size_t i = 1; i < std::min<size_t>(output.size(), 10); ++i) {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
    }

}