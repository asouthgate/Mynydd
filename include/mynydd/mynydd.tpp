#pragma once

#include "mynydd/mynydd.hpp"
#include <assert.h>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

// #include "mynydd.hpp"

namespace mynydd {
    
    VulkanContext createVulkanContext();
    // Required forward declarations for Vulkan functions used in the template
    VkBuffer createBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage);
    VkDeviceMemory allocateAndBindMemory(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkBuffer buffer,
        VkMemoryPropertyFlags properties
    );
    VkDescriptorSet allocateDescriptorSet(
        VkDevice device,
        VkDescriptorSetLayout descriptorSetLayout,
        VkDescriptorPool &descriptorPool
    );
    void updateDescriptorSet(
        VkDevice device,
        VkDescriptorSet descriptorSet,
        VkBuffer buffer,
        VkDeviceSize size
    );
    VulkanPipelineResources create_pipeline_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        const char* shaderPath,
        VkDescriptorSetLayout &descriptorLayout
    );
    VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex);
    VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool commandPool);
    void recordCommandBuffer(
        VkCommandBuffer commandBuffer,
        VkPipeline pipeline,
        VkPipelineLayout pipelineLayout,
        VkDescriptorSet descriptorSet,
        uint32_t numElements
    );
    void submitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmdBuffer);
    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device);

    template<typename T>
    ComputeEngine<T>::ComputeEngine(
        std::shared_ptr<VulkanContext> contextPtr, 
        std::shared_ptr<VulkanDynamicResources> dynamicResourcesPtr,
        const char* shaderPath
    ) {
        std::cerr << "Creating ComputeEngine resources..." << std::endl;
        this->contextPtr = contextPtr;
        this->dynamicResourcesPtr = dynamicResourcesPtr;
        this->pipelineResources = create_pipeline_resources(contextPtr, shaderPath, this->dynamicResourcesPtr->descriptorSetLayout);
    }

    template<typename T>
    ComputeEngine<T>::~ComputeEngine() {
        std::cerr << "Destroying ComputeEngine resources..." << std::endl;
        try {
            std::cerr << "vkDestroyPipeline" << std::endl;
            if (this->contextPtr && this->contextPtr->device != VK_NULL_HANDLE &&
                this->pipelineResources.pipeline != VK_NULL_HANDLE) {
                std::cerr << "Handles seem valid?\n";
            } else {
                std::cerr << "Invalid handles in vkDestroyPipeline\n";
                throw;
            }
            vkDestroyPipeline(this->contextPtr->device, this->pipelineResources.pipeline, nullptr);
            std::cerr << "vkDestroyPipelineLayout" << std::endl;
            vkDestroyPipelineLayout(this->contextPtr->device, this->pipelineResources.pipelineLayout, nullptr);
            std::cerr << "vkDestroyShaderModule" << std::endl;
            vkDestroyShaderModule(this->contextPtr->device, this->pipelineResources.computeShaderModule, nullptr);
        } catch (const std::exception &e) {
            std::cerr << "Error during ComputeEngine destruction: " << e.what() << std::endl;
            throw;
        }
        std::cerr << "ComputeEngine resources destroyed." << std::endl;
    }

    struct TrivialUniform {
        float dummy = 0.0f;
    };

    template<typename T, typename U = TrivialUniform>
    std::shared_ptr<VulkanDynamicResources> createDataResources(
        std::shared_ptr<VulkanContext> contextPtr,
        size_t n_data_elements
    ) {
        return std::make_shared<VulkanDynamicResources>(
            contextPtr,
            n_data_elements * sizeof(T),
            sizeof(U)
        );
    }

    template<typename T>
    void uploadBufferData(VkDevice device, VkDeviceMemory memory, const std::vector<T>& inputData) {
        void* mapped;
        VkDeviceSize size = sizeof(T) * inputData.size();

        if (vkMapMemory(device, memory, 0, size, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map buffer memory for upload");
        }

        std::memcpy(mapped, inputData.data(), static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    template<typename T>
    template<typename U>
    void ComputeEngine<T>::uploadUniformData(const U uniform) {

        if (sizeof(U) > this->dynamicResourcesPtr->uniformSize) {
            throw std::runtime_error(
                "Uniform size (" + std::to_string(sizeof(U)) + 
                " bytes) does not match expected size (" + 
                std::to_string(this->dynamicResourcesPtr->uniformSize) + " bytes)!"
            );
        }
        void* mapped;
        VkDeviceSize size = sizeof(U);

        if (vkMapMemory(this->contextPtr->device, this->dynamicResourcesPtr->uniformMemory, 0, size, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map uniform buffer memory for upload");
        }

        std::memcpy(mapped, &uniform, static_cast<size_t>(size));
        vkUnmapMemory(this->contextPtr->device, this->dynamicResourcesPtr->uniformMemory);
    }

    template<typename T>
    void ComputeEngine<T>::uploadData(const std::vector<T> &inputData) {
        try {
            if (inputData.empty()) {
                throw std::runtime_error("Data vector is empty");
            }

            this->numElements = static_cast<uint32_t>(inputData.size());
            this->dataSize = sizeof(T) * numElements;

            if (this->dataSize > this->dynamicResourcesPtr->dataSize) {
                throw std::runtime_error("Data size exceeds allocated buffer size");
            }

            uploadBufferData<T>(this->contextPtr->device, this->dynamicResourcesPtr->memory, inputData);
            std::cerr << "Upload complete. Data size: " << this->dataSize << " bytes." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in uploadData: " << e.what() << std::endl;
            throw;
        }
    }

    /**
    * Maps Vulkan device memory and copies data into a CPU vector.
    */
    template<typename T>
    std::vector<T> readBufferData(VkDevice device, VkDeviceMemory memory, VkDeviceSize dataSize, uint32_t numElements) {
        void* mappedData;
        vkMapMemory(device, memory, 0, dataSize, 0, &mappedData);

        T* data = reinterpret_cast<T*>(mappedData);
        std::vector<T> result(data, data + numElements);

        vkUnmapMemory(device, memory);
        return result;
    }

    template<typename T>
    void ComputeEngine<T>::execute() {
        std::cerr<< "Recording command buffer..." << std::endl;
        try {
            if (!this->contextPtr || this->contextPtr->device == VK_NULL_HANDLE) {
                throw std::runtime_error("Invalid Vulkan context or device handle");
            }
            if (!this->dynamicResourcesPtr || this->dynamicResourcesPtr->descriptorSet == VK_NULL_HANDLE) {
                throw std::runtime_error("Invalid dynamic resources or descriptor set handle");
            }
            if (this->pipelineResources.pipeline == VK_NULL_HANDLE || this->pipelineResources.pipelineLayout == VK_NULL_HANDLE) {
                throw std::runtime_error("Invalid pipeline or pipeline layout handle");
            }
            recordCommandBuffer(
                    this->contextPtr->commandBuffer,
                    this->pipelineResources.pipeline,
                    this->pipelineResources.pipelineLayout,
                    this->dynamicResourcesPtr->descriptorSet,
                    this->numElements
            );
        } catch (const std::exception &e) {
            std::cerr << "Error during execution setup: " << e.what() << std::endl;
            throw;
        }
        std::cerr << "Submitting command buffer and waiting for execution..." << std::endl;
        submitAndWait(
            this->contextPtr->device,
            this->contextPtr->computeQueue,
            this->contextPtr->commandBuffer
        );
    }

    template<typename T>
    std::vector<T> ComputeEngine<T>::fetchData() {

        std::vector<T> output = readBufferData<T>(
            this->contextPtr->device,
            this->dynamicResourcesPtr->outputMemory,
            this->dataSize,
            this->numElements
        );

        return output;
    }

}