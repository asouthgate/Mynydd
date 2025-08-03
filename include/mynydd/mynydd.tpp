#pragma once

#include "mynydd/mynydd.hpp"
#include <assert.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace mynydd {
    
    // VulkanContext createVulkanContext();
    // // Required forward declarations for Vulkan functions used in the template
    
    // VkDescriptorSet allocateDescriptorSet(
    //     VkDevice device,
    //     VkDescriptorSetLayout descriptorSetLayout,
    //     VkDescriptorPool &descriptorPool
    // );
    // VulkanPipelineResources create_pipeline_resources(
    //     std::shared_ptr<VulkanContext> contextPtr,
    //     const char* shaderPath,
    //     VkDescriptorSetLayout &descriptorLayout,
    //     std::vector<uint32_t> pushConstantSizes = {}
    // );
    // VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex);
    // VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool commandPool);
    // void recordCommandBuffer(
    //     VkCommandBuffer commandBuffer,
    //     VkPipeline pipeline,
    //     VkPipelineLayout pipelineLayout,
    //     VkDescriptorSet descriptorSet,
    //     uint32_t numElements
    // );
    // void submitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmdBuffer);
    // VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device);

    template<typename T>
    ComputeEngine<T>::ComputeEngine(
        std::shared_ptr<VulkanContext> contextPtr, 
        const char* shaderPath,
        std::vector<std::shared_ptr<AllocatedBuffer>> buffers
    ) : contextPtr(contextPtr) {
        this->dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(
            contextPtr,
            buffers
        );
        assert(this->dynamicResourcesPtr->descriptorSetLayout != VK_NULL_HANDLE);
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

    // template<typename T, typename U = TrivialUniform>
    // std::shared_ptr<VulkanDynamicResources> createDataResources(
    //     std::shared_ptr<VulkanContext> contextPtr,
    //     std::shared_ptr<AllocatedBuffer> input,
    //     size_t n_data_elements
    // ) {
    //     return std::make_shared<VulkanDynamicResources>(
    //         contextPtr,
    //         input,
    //         output,
    //         uniform,
    //         n_data_elements * sizeof(T),
    //         sizeof(U)
    //     );
    // }

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

    template<typename U>
    void uploadUniformData(std::shared_ptr<VulkanContext> vkc, const U uniform, std::shared_ptr<Buffer> buff) {
        if (sizeof(U) > buff->getSize()) {
            throw std::runtime_error(
                "Uniform size (" + std::to_string(sizeof(U)) + 
                " bytes) does not match expected size (" + 
                std::to_string(buff->getSize()) + " bytes)!"
            );
        }
        void* mapped;
        VkDeviceSize size = sizeof(U);

        if (vkMapMemory(vkc->device, buff->getMemory(), 0, size, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map uniform buffer memory for upload");
        }

        std::memcpy(mapped, &uniform, static_cast<size_t>(size));
        vkUnmapMemory(vkc->device, buff->getMemory());
    }


    template<typename T>
    template<typename U>
    void ComputeEngine<T>::uploadUniformData(const U uniform) {

        if (sizeof(U) > this->dynamicResourcesPtr->uniform->getSize()) {
            throw std::runtime_error(
                "Uniform size (" + std::to_string(sizeof(U)) + 
                " bytes) does not match expected size (" + 
                std::to_string(this->dynamicResourcesPtr->uniform->getSize()) + " bytes)!"
            );
        }
        void* mapped;
        VkDeviceSize size = sizeof(U);

        if (vkMapMemory(this->contextPtr->device, this->dynamicResourcesPtr->uniform->getMemory(), 0, size, 0, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map uniform buffer memory for upload");
        }

        std::memcpy(mapped, &uniform, static_cast<size_t>(size));
        vkUnmapMemory(this->contextPtr->device, this->dynamicResourcesPtr->uniform->getMemory());
    }

    template<typename T>
    void ComputeEngine<T>::uploadData(const std::vector<T> &inputData) {
        try {
            if (inputData.empty()) {
                throw std::runtime_error("Data vector is empty");
            }

            this->numElements = static_cast<uint32_t>(inputData.size());
            this->dataSize = sizeof(T) * numElements;
            std::cerr << "WARNING: dataSize is going to be deprecated out of ComputeEngine" << std::endl;
            if (this->dataSize > this->dynamicResourcesPtr->input->getSize()) {
                throw std::runtime_error("Data size exceeds allocated buffer size");
            }

            uploadBufferData<T>(this->contextPtr->device, this->dynamicResourcesPtr->input->getMemory(), inputData);
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
            this->dynamicResourcesPtr->output->getMemory(),
            this->dataSize,
            this->numElements
        );

        return output;
    }

}