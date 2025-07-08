#pragma once

#include <assert.h>
#include <cstring>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>


namespace mynydd {

    /**
    * Context variables required for Vulkan compute.
    */
    struct VulkanContext {
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device; // logical device used for interface
        VkQueue computeQueue; // compute queue used for commands
        uint32_t computeQueueFamilyIndex;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
    };

    struct VulkanPipelineResources {
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        VkShaderModule computeShaderModule;
    };

    struct VulkanDynamicResources {
        VkBuffer buffer;
        VkDeviceMemory memory;
        // A single descriptor set layout set/binding is used per buffer
        // This is for convenience; in future may use multiple bindings
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        size_t dataSize;
    };    

    VulkanContext createVulkanContext();

    template<typename T>
    class ComputeEngine {
        public:
            ComputeEngine(
                std::shared_ptr<VulkanContext> contextPtr,
                std::shared_ptr<VulkanDynamicResources> dynamicResources,
                const char* shaderPath
            ); 
            ~ComputeEngine();

            void uploadData(const std::vector<T> &data);
            std::vector<T> execute();            

        private:
            std::shared_ptr<VulkanContext> contextPtr; // shared because we can have multiple pipelines per context
            std::shared_ptr<VulkanDynamicResources> dynamicResourcesPtr; // shared because we can have multiple pipelines per data
            // VulkanDynamicResources dynamicResources;
            VulkanPipelineResources pipelineResources;
            uint32_t numElements; // number of elements in the data buffer
            VkDeviceSize dataSize;
    };
};

// Template implementation must be in the header file or a separate file included at the end of the header file.
// Included here beause it looks messy otherwise
#include "compute_context.tpp"