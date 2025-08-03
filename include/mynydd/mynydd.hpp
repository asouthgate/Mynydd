#pragma once

#include <assert.h>
#include <cstdint>
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
        VkBuffer uniformBuffer;
        VkDeviceMemory uniformMemory;
        VkBuffer outputBuffer;
        VkDeviceMemory outputMemory;
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

    class AllocatedBuffer {
    public:
        AllocatedBuffer() = default;

        AllocatedBuffer(VkDevice device, VkPhysicalDevice physicalDevice, size_t size);

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
                std::shared_ptr<VulkanDynamicResources> dynamicResources,
                const char* shaderPath
            ); 
            ~ComputeEngine();

            template<typename U> void uploadUniformData(const U uniform);
            void uploadData(const std::vector<T> &data);
            void execute();     
            std::vector<T> fetchData();                   

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
#include "mynydd.tpp"