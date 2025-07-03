#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

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
        VkDescriptorSet descriptorSet;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
    };

    /**
    */
    class ComputeContext {
        public:
            ComputeContext();
            ~ComputeContext();
        
        private:
            VulkanContext context;
    };

    class ComputePipeline {
        public:
            ComputePipeline(const VulkanContext &context);
            ~ComputePipeline();

            void createPipelineResources();
            void createDynamicResources(size_t n_data_elements);

            void uploadData(const std::vector<float> &data);
            void execute();

        private:
            std::shared_ptr<VulkanContext> context;
            VulkanDynamicResources dynamicResources;
            VulkanPipelineResources pipelineResources;
    }
};