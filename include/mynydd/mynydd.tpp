#pragma once

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
        std::cerr << "WARNING: TODO RAII" << std::endl;
        try {
            vkFreeCommandBuffers(this->contextPtr->device, this->contextPtr->commandPool, 1, &this->contextPtr->commandBuffer);
            vkDestroyCommandPool(this->contextPtr->device, this->contextPtr->commandPool, nullptr);
            vkDestroyPipeline(this->contextPtr->device, this->pipelineResources.pipeline, nullptr);
            vkDestroyPipelineLayout(this->contextPtr->device, this->pipelineResources.pipelineLayout, nullptr);
            vkDestroyDescriptorPool(this->contextPtr->device, this->dynamicResourcesPtr->descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(this->contextPtr->device, this->dynamicResourcesPtr->descriptorSetLayout, nullptr);
            vkDestroyShaderModule(this->contextPtr->device, this->pipelineResources.computeShaderModule, nullptr);
            vkDestroyDevice(this->contextPtr->device, nullptr);
            vkDestroyInstance(this->contextPtr->instance, nullptr);

        } catch (const std::exception &e) {
            throw;
        }
    }

    VulkanDynamicResources create_dynamic_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        size_t dataSize
    ) {
        // const size_t dataSize = n_data_elements * sizeof(T);

        VkBuffer buffer = createBuffer(
            contextPtr->device, dataSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        );

        VkDeviceMemory bufferMemory = allocateAndBindMemory(
            contextPtr->physicalDevice, 
            contextPtr->device,
            buffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        VkDescriptorSetLayout descriptorLayout =
            createDescriptorSetLayout(contextPtr->device);


        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet =
            allocateDescriptorSet(contextPtr->device, descriptorLayout, descriptorPool);

        updateDescriptorSet(contextPtr->device, descriptorSet, buffer, dataSize);

        return {
            buffer,
            bufferMemory,
            descriptorLayout,
            descriptorPool,
            descriptorSet, // descriptorSet will be created later
            dataSize
        };
    }

    template<typename T>
    VulkanDynamicResources createDataResources(std::shared_ptr<VulkanContext> contextPtr, size_t n_data_elements) {
        // Create dynamic resources with the specified number of data elements
        return create_dynamic_resources(
            contextPtr,
            n_data_elements * sizeof(T)
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
    void ComputeEngine<T>::uploadData(const std::vector<T> &inputData) {

        if (inputData.empty()) {
            throw std::runtime_error("Data vector is empty");
        }

        this->numElements = static_cast<uint32_t>(inputData.size());
        this->dataSize = sizeof(T) * numElements;

        if (this->dataSize > this->dynamicResourcesPtr->dataSize) {
            throw std::runtime_error("Data size exceeds allocated buffer size");
        }

        // Upload to the GPU buffer
        uploadBufferData<T>(this->contextPtr->device, this->dynamicResourcesPtr->memory, inputData);
        std::cerr << "Upload complete. Data size: " << this->dataSize << " bytes." << std::endl;
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
    std::vector<T> ComputeEngine<T>::execute() {
        std::cerr<< "Recording command buffer..." << std::endl;
        recordCommandBuffer(
            this->contextPtr->commandBuffer,
            this->pipelineResources.pipeline,
            this->pipelineResources.pipelineLayout,
            this->dynamicResourcesPtr->descriptorSet,
            this->numElements
        );
        std::cerr << "Submitting command buffer and waiting for execution..." << std::endl;
        submitAndWait(
            this->contextPtr->device,
            this->contextPtr->computeQueue,
            this->contextPtr->commandBuffer
        );

        std::vector<T> output = readBufferData<T>(
            this->contextPtr->device,
            this->dynamicResourcesPtr->memory,
            this->dataSize,
            this->numElements
        );

        return output;
    }
}