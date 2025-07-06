#pragma once

#include <assert.h>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "compute_context.hpp"

namespace mylib {
    
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
    VulkanPipelineResources create_pipeline_resources(std::shared_ptr<VulkanContext> contextPtr, const char* shaderPath);
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

    template<typename T>
    ComputePipeline<T>::ComputePipeline(std::shared_ptr<VulkanContext> contextPtr, const char* shaderPath) {
        std::cerr << "Creating ComputePipeline resources..." << std::endl;
        this->contextPtr = contextPtr;
        this->pipelineResources = create_pipeline_resources(contextPtr, shaderPath);
    }

    template<typename T>
    ComputePipeline<T>::~ComputePipeline() {
        std::cerr << "Destroying ComputePipeline resources..." << std::endl;
        try {
            vkFreeCommandBuffers(this->contextPtr->device, this->dynamicResources.commandPool, 1, &this->dynamicResources.commandBuffer);
            vkDestroyCommandPool(this->contextPtr->device, this->dynamicResources.commandPool, nullptr);
            vkDestroyPipeline(this->contextPtr->device, this->pipelineResources.pipeline, nullptr);
            vkDestroyPipelineLayout(this->contextPtr->device, this->pipelineResources.pipelineLayout, nullptr);
            vkDestroyDescriptorPool(this->contextPtr->device, this->dynamicResources.descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(this->contextPtr->device, this->pipelineResources.descriptorSetLayout, nullptr);
            vkDestroyShaderModule(this->contextPtr->device, this->pipelineResources.computeShaderModule, nullptr);
            vkDestroyDevice(this->contextPtr->device, nullptr);
            vkDestroyInstance(this->contextPtr->instance, nullptr);

        } catch (const std::exception &e) {
            throw;
        }
    }

        template<typename T>
    VulkanDynamicResources create_dynamic_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        VkDescriptorSetLayout descriptorLayout,
        size_t n_data_elements
    ) {
        const size_t dataSize = n_data_elements * sizeof(T);

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

        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet =
            allocateDescriptorSet(contextPtr->device, descriptorLayout, descriptorPool);

        updateDescriptorSet(contextPtr->device, descriptorSet, buffer, dataSize);

        VkCommandPool commandPool = createCommandPool(
            contextPtr->device, contextPtr->computeQueueFamilyIndex
        );

        VkCommandBuffer commandBuffer = allocateCommandBuffer(
            contextPtr->device, commandPool
        );


        return {
            buffer,
            bufferMemory,
            descriptorPool,
            descriptorSet, // descriptorSet will be created later
            commandPool, // commandPool will be created later
            commandBuffer,  // commandBuffer will be created later
            dataSize
        };
    }

    template<typename T>
    void ComputePipeline<T>::createDynamicResources(size_t n_data_elements) {
        // Create dynamic resources with the specified number of data elements
        this->dynamicResources = create_dynamic_resources<T>(
            this->contextPtr,
            this->pipelineResources.descriptorSetLayout,
            n_data_elements
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
    void ComputePipeline<T>::uploadData(const std::vector<T> &data) {

        if (data.empty()) {
            throw std::runtime_error("Data vector is empty");
        }

        this->numElements = static_cast<uint32_t>(data.size());
        this->dataSize = sizeof(T) * numElements;

        if (this->dataSize > this->dynamicResources.dataSize) {
            throw std::runtime_error("Data size exceeds allocated buffer size");
        }

        std::vector<T> inputData(numElements);
        for (uint32_t i = 0; i < numElements; ++i) {
            inputData[i] = static_cast<T>(i);
        }

        // Upload to the GPU buffer
        uploadBufferData<T>(this->contextPtr->device, this->dynamicResources.memory, inputData);
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
    void ComputePipeline<T>::execute() {
        std::cerr<< "Recording command buffer..." << std::endl;
        recordCommandBuffer(
            this->dynamicResources.commandBuffer,
            this->pipelineResources.pipeline,
            this->pipelineResources.pipelineLayout,
            this->dynamicResources.descriptorSet,
            this->numElements
        );
        std::cerr << "Submitting command buffer and waiting for execution..." << std::endl;
        submitAndWait(
            this->contextPtr->device,
            this->contextPtr->computeQueue,
            this->dynamicResources.commandBuffer
        );

        std::vector<T> output = readBufferData<T>(
            this->contextPtr->device,
            this->dynamicResources.memory,
            this->dataSize,
            this->numElements
        );

        // Print the first 10 results
        for (size_t i = 0; i < std::min<size_t>(output.size(), 10); ++i) {
            std::cout << "output[" << i << "] = " << output[i] << std::endl;
        }
    }
}