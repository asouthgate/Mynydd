#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

#include "compute_context.hpp"

int main() {
    mylib::VulkanContext context = mylib::createVulkanContext();
    std::shared_ptr<mylib::VulkanContext> contextPtr = std::make_shared<mylib::VulkanContext>(context);
    mylib::ComputePipeline pipeline(contextPtr);
    std::vector<float> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    pipeline.createDynamicResources(1024);
    pipeline.uploadData(data);
    pipeline.execute();
    return 0;
}