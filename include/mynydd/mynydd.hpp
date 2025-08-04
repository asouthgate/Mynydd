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

    class AllocatedBuffer {
    public:
        AllocatedBuffer() = default;

        AllocatedBuffer(std::shared_ptr<VulkanContext> vkc, size_t size, bool uniform=false);

        // Prevent copying
        AllocatedBuffer(const AllocatedBuffer&) = delete;
        AllocatedBuffer& operator=(const AllocatedBuffer&) = delete;

        // Move implementation
        AllocatedBuffer(AllocatedBuffer&& other) noexcept
            : device(other.device), buffer(other.buffer), memory(other.memory), size(other.size) {
            other.buffer = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
        }

        AllocatedBuffer& operator=(AllocatedBuffer&& other) noexcept {
            if (this != &other) {
                destroy();
                device = other.device;
                buffer = other.buffer;
                memory = other.memory;
                size = other.size;

                other.buffer = VK_NULL_HANDLE;
                other.memory = VK_NULL_HANDLE;
            }
            return *this;
        }

        ~AllocatedBuffer() {
            std::cerr << "Destroying AllocatedBuffer..." << std::endl;
            destroy();
        }

        VkBuffer getBuffer() const { return buffer; }
        VkDeviceMemory getMemory() const { return memory; }
        VkDeviceSize getSize() const { return size; }

        explicit operator bool() const { return buffer != VK_NULL_HANDLE; }

    private:
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;

        void destroy() {
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, memory, nullptr);
            }
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
        }
    };

    struct VulkanDynamicResources {
        std::shared_ptr<VulkanContext> contextPtr;
        std::shared_ptr<AllocatedBuffer> input;
        std::shared_ptr<AllocatedBuffer> output;
        std::shared_ptr<AllocatedBuffer> uniform;
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        VulkanDynamicResources(
            std::shared_ptr<VulkanContext> contextPtr,
            std::shared_ptr<AllocatedBuffer> input,
            std::shared_ptr<AllocatedBuffer> output,
            std::shared_ptr<AllocatedBuffer> uniform
        );
        ~VulkanDynamicResources() {
            std::cerr << "Destroying VulkanDynamicResources..." << std::endl;
            if (contextPtr && contextPtr->device != VK_NULL_HANDLE && descriptorPool != VK_NULL_HANDLE) {
            } else {
                std::cerr << "VulkanDynamicResources destructor failure due to invalid dependency handles." << std::endl;
                throw;
            }
            vkDestroyDescriptorPool(this->contextPtr->device, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(this->contextPtr->device, descriptorSetLayout, nullptr);
            std::cerr << "VulkanDynamicResources destroyed." << std::endl;
        }
        VulkanDynamicResources(const VulkanDynamicResources&) = delete;            // No copy
        VulkanDynamicResources& operator=(const VulkanDynamicResources&) = delete; // No copy
        VulkanDynamicResources(VulkanDynamicResources&&) = default;                // Allow move
        VulkanDynamicResources& operator=(VulkanDynamicResources&&) = default;     // Allow move
    };

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