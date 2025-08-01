#pragma once

#include <assert.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <mynydd/memory.hpp>


namespace mynydd {

    // TODO: much of this should be private, in a class
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

        VulkanContext();

        ~VulkanContext() {
            std::cerr << "Destroying Vulkan context..." << std::endl;
            if (commandBuffer != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
            }
            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
            if (device != VK_NULL_HANDLE) {
                vkDestroyDevice(device, nullptr);
            }
            if (instance != VK_NULL_HANDLE) {
                vkDestroyInstance(instance, nullptr);
            }
            std::cerr << "Vulkan context destroyed." << std::endl;
        }

        VulkanContext(const VulkanContext&) = delete;            // No copy
        VulkanContext& operator=(const VulkanContext&) = delete; // No copy
        VulkanContext(VulkanContext&&) = default;                // Allow move
        VulkanContext& operator=(VulkanContext&&) = default;     // Allow move
    };

    struct VulkanPipelineResources {
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        VkShaderModule computeShaderModule;
    };

    VulkanContext createVulkanContext();


    struct VulkanDynamicResources {
        VkBuffer buffer;
        VkDeviceMemory memory;
        VkBuffer uniformBuffer;
        VkDeviceMemory uniformMemory;
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        size_t dataSize;
        size_t uniformSize;
    };

    VulkanContext createVulkanContext();

    VulkanDynamicResources create_dynamic_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        size_t dataSize,
        size_t uniformSize
    );

    // template<typename T>
    // class DataResources {
    //     public:
    //         DataResources(
    //             std::shared_ptr<VulkanContext> contextPtr,
    //             size_t n_data_elements
    //         ) {
    //             this->dynamicResources = create_dynamic_resources(contextPtr, n_data_elements);
    //         }

    //         ~DataResources<T>() {
    //             // Destructor to clean up resources
    //             vkDestroyBuffer(this->contextPtr->device, this->dynamicResources.buffer, nullptr);
    //             vkFreeMemory(this->contextPtr->device, this->dynamicResources.memory, nullptr);
    //             vkDestroyDescriptorPool(this->contextPtr->device, this->dynamicResourcesPtr->descriptorPool, nullptr);
    //             vkDestroyDescriptorSetLayout(
    //                 this->contextPtr->device, this->dynamicResources.descriptorSetLayout, nullptr
    //             );
    //         }
    //     private:
    //         VulkanDynamicResources dynamicResources;
    //         std::shared_ptr<VulkanContext> contextPtr;
    // };

    template<typename T>
    class ComputeEngine {
        public:
            ComputeEngine(
                std::shared_ptr<VulkanContext> contextPtr,
                const char* shaderPath,
                std::vector<std::shared_ptr<AllocatedBuffer>> buffers
            ); 
            ~ComputeEngine();

            template<typename U> void uploadUniformData(const U uniform);
            void uploadData(const std::vector<T> &data);
            void execute();     
            std::vector<T> fetchData();                   

        private:
            std::shared_ptr<VulkanContext> contextPtr; // shared because we can have multiple pipelines per context
            std::shared_ptr<VulkanDynamicResources> dynamicResourcesPtr; // shared because we can have multiple pipelines per data
            VulkanPipelineResources pipelineResources;
    };
};

// Template implementation must be in the header file or a separate file included at the end of the header file.
// Included here beause it looks messy otherwise
#include "mynydd.tpp"