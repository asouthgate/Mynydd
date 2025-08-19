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
        std::shared_ptr<VulkanContext> contextPtr;
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        VulkanDynamicResources(
            std::shared_ptr<VulkanContext> contextPtr,
            std::vector<std::shared_ptr<Buffer>> buffers
        );
        ~VulkanDynamicResources() {
            std::cerr << "Destroying VulkanDynamicResources..." << std::endl;
            if (contextPtr && contextPtr->device != VK_NULL_HANDLE && descriptorPool != VK_NULL_HANDLE) {
            } else {
                std::cerr << "VulkanDynamicResources destructor failure due to invalid dependency handles." << std::endl;
            }
            vkDestroyDescriptorPool(this->contextPtr->device, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(this->contextPtr->device, descriptorSetLayout, nullptr);
            std::cerr << "VulkanDynamicResources destroyed." << std::endl;
        }
        VulkanDynamicResources(const VulkanDynamicResources&) = delete;            // No copy
        VulkanDynamicResources& operator=(const VulkanDynamicResources&) = delete; // No copy
        VulkanDynamicResources(VulkanDynamicResources&&) = default;                // Allow move
        VulkanDynamicResources& operator=(VulkanDynamicResources&&) = default;     // Allow move
        void setBuffers(
            std::shared_ptr<VulkanContext> contextPtr,
            const std::vector<std::shared_ptr<Buffer>>& buffers
        );
    };

    template<typename T>
    class PipelineStep {
        public:
            PipelineStep(
                std::shared_ptr<VulkanContext> contextPtr,
                const char* shaderPath,
                std::vector<std::shared_ptr<Buffer>> buffers,
                uint32_t groupCountX,
                uint32_t groupCountY=1,
                uint32_t groupCountZ=1
            ); 
            ~PipelineStep();
            void execute(size_t numElements); //numElements required for computing nthreads
            std::shared_ptr<VulkanPipelineResources> getPipelineResourcesPtr() const {
                return std::make_shared<VulkanPipelineResources>(pipelineResources);
            }
            std::shared_ptr<VulkanDynamicResources> getDynamicResourcesPtr() const {
                return dynamicResourcesPtr;
            }
            // TODO: make private
            uint32_t groupCountX;
            uint32_t groupCountY;
            uint32_t groupCountZ;
            void setBuffers(
                std::shared_ptr<VulkanContext> contextPtr,
                const std::vector<std::shared_ptr<Buffer>>& buffers
            );


        private:
            std::shared_ptr<VulkanContext> contextPtr; // shared because we can have multiple pipelines per context
            std::shared_ptr<VulkanDynamicResources> dynamicResourcesPtr; // shared because we can have multiple pipelines per data
            VulkanPipelineResources pipelineResources;
    };
};

// Template implementation must be in the header file or a separate file included at the end of the header file.
// Included here beause it looks messy otherwise
#include "mynydd.tpp"