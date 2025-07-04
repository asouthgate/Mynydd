#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace mylib {

    /**
    * Context variables required for Vulkan compute.
    */
    struct VulkanContext {
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device; // logical device used for interface
        VkQueue computeQueue; // compute queue used for commands
        uint32_t computeQueueFamilyIndex;
    };

    struct VulkanPipelineResources {
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        VkShaderModule computeShaderModule;
        VkDescriptorSetLayout descriptorSetLayout;
    };

    struct VulkanDynamicResources {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
        size_t dataSize;
    };

    VulkanContext createVulkanContext();

    class ComputePipeline {
        public:
            ComputePipeline(std::shared_ptr<VulkanContext> contextPtr); 
            ~ComputePipeline();

            void createPipelineResources();
            void createDynamicResources(size_t n_data_elements);

            void uploadData(const std::vector<float> &data);
            void execute();            

        private:
            std::shared_ptr<VulkanContext> contextPtr; // shared because we can have multiple pipelines per context
            VulkanDynamicResources dynamicResources;
            VulkanPipelineResources pipelineResources;
            uint32_t numElements; // number of elements in the data buffer
            VkDeviceSize dataSize;
    };
};