#pragma once

#include "mynydd/memory.hpp"
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
    // void updateDescriptorSet(
    //     VkDevice device,
    //     VkDescriptorSet descriptorSet,
    //     const std::vector<std::shared_ptr<Buffer>> &buffers
    // );
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
    PipelineStep<T>::PipelineStep(
        std::shared_ptr<VulkanContext> contextPtr, 
        const char* shaderPath,
        std::vector<std::shared_ptr<Buffer>> buffers,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ
    ) : contextPtr(contextPtr), groupCountX(groupCountX), groupCountY(groupCountY), groupCountZ(groupCountZ) {  
        this->dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(
            contextPtr,
            buffers
        );
        assert(this->dynamicResourcesPtr->descriptorSetLayout != VK_NULL_HANDLE);
        this->pipelineResources = create_pipeline_resources(contextPtr, shaderPath, this->dynamicResourcesPtr->descriptorSetLayout);
    }

    template<typename T>
    PipelineStep<T>::~PipelineStep() {
        std::cerr << "Destroying PipelineStep resources..." << std::endl;
        try {
            std::cerr << "Destroying PipelineStep..." << std::endl;
            if (this->contextPtr && this->contextPtr->device != VK_NULL_HANDLE &&
                this->pipelineResources.pipeline != VK_NULL_HANDLE) {
            } else {
                std::cerr << "Invalid handles in vkDestroyPipeline\n";
                throw;
            }
            vkDestroyPipeline(this->contextPtr->device, this->pipelineResources.pipeline, nullptr);
            vkDestroyPipelineLayout(this->contextPtr->device, this->pipelineResources.pipelineLayout, nullptr);
            vkDestroyShaderModule(this->contextPtr->device, this->pipelineResources.computeShaderModule, nullptr);
        } catch (const std::exception &e) {
            std::cerr << "Error during PipelineStep destruction: " << e.what() << std::endl;
            throw;
        }
        std::cerr << "PipelineStep resources destroyed." << std::endl;
    }

    struct TrivialUniform {
        float dummy = 0.0f;
    };

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
    void uploadData(std::shared_ptr<VulkanContext> vkc, const std::vector<T> &inputData, std::shared_ptr<Buffer> buffer) {
        try {
            if (inputData.empty()) {
                throw std::runtime_error("Data vector is empty");
            }

            size_t numElements = static_cast<uint32_t>(inputData.size());
            size_t dataSize = sizeof(T) * numElements;

            if (dataSize > buffer->getSize()) {
                throw std::runtime_error("Data size exceeds allocated buffer size");
            }

            std::cerr << "About to upload buffer data" << std::endl;
            uploadBufferData<T>(vkc->device, buffer->getMemory(), inputData);
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
    void PipelineStep<T>::execute(size_t numElements) {
        std::cerr << "Warning: PipelineStep::execute is deprecated. Use executeBatch instead." << std::endl;
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
                numElements
            );
        } catch (const std::exception &e) {
            std::cerr << "Error during execution setup: " << e.what() << std::endl;
            throw;
        }
        submitAndWait(
            this->contextPtr->device,
            this->contextPtr->computeQueue,
            this->contextPtr->commandBuffer
        );
    }

    template<typename T>
    std::vector<T> fetchData(std::shared_ptr<VulkanContext> vkc, std::shared_ptr<Buffer> buffer, size_t n_elements) {

        std::vector<T> output = readBufferData<T>(
            vkc->device,
            buffer->getMemory(),
            buffer->getSize(),
            n_elements
        );

        return output;
    }


    template<typename T>
    void recordCommandBuffer(
        VkCommandBuffer cmdBuffer,
        std::shared_ptr<PipelineStep<T>> pipeline_step,
        bool memory_barrier = true
    ) {
            const auto& pipeline      = pipeline_step->getPipelineResourcesPtr()->pipeline;
            const auto& layout        = pipeline_step->getPipelineResourcesPtr()->pipelineLayout;
            const auto& descriptorSet = pipeline_step->getDynamicResourcesPtr()->descriptorSet;

            if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE || descriptorSet == VK_NULL_HANDLE) {
                throw std::runtime_error("Invalid pipeline or descriptor set for engine step.");
            }

            // Bind pipeline and descriptor sets
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, nullptr);

            // Dispatch compute shader
            vkCmdDispatch(cmdBuffer, 
                pipeline_step->groupCountX,
                pipeline_step->groupCountY,
                pipeline_step->groupCountZ
            );

            // Insert memory barrier between shaders (except after last one)
            if (memory_barrier) {
                VkMemoryBarrier memoryBarrier{};
                memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(
                    cmdBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &memoryBarrier,
                    0, nullptr,
                    0, nullptr
                );
            }

    }


    template<typename T>
    void executeBatch(
        std::shared_ptr<VulkanContext> contextPtr,
        const std::vector<std::shared_ptr<PipelineStep<T>>>& PipelineSteps
    ) {
        if (PipelineSteps.empty()) {
            throw std::runtime_error("No compute engines provided for batch execution.");
        }

        if (!contextPtr || contextPtr->device == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid Vulkan context in batch execution.");
        }

        VkCommandBuffer cmdBuffer = contextPtr->commandBuffer;

        // Begin command buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer for batch execution.");
        }

        for (size_t i = 0; i < PipelineSteps.size(); ++i) {
            auto& engine = PipelineSteps[i];
            if (!engine) {
                throw std::runtime_error("Null PipelineStep pointer at index " + std::to_string(i));
            }
            recordCommandBuffer(
                cmdBuffer,
                engine,
                i + 1 < PipelineSteps.size()
            );
        }

        if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end command buffer for batch execution.");
        }

        // Submit command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        VkFence fence;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(contextPtr->device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence for batch execution.");
        }

        if (vkQueueSubmit(contextPtr->computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            vkDestroyFence(contextPtr->device, fence, nullptr);
            throw std::runtime_error("Failed to submit batched command buffer.");
        }
        std::cerr << "Waiting for fence after batch execution..." << std::endl;
        vkWaitForFences(contextPtr->device, 1, &fence, VK_TRUE, UINT64_MAX);
        std::cerr << "Batch execution completed." << std::endl;
        vkDestroyFence(contextPtr->device, fence, nullptr);
        std::cerr << "Batch execution finished successfully." << std::endl;
    }

    template<typename T>
    void PipelineStep<T>::setBuffers(
        std::shared_ptr<VulkanContext> contextPtr,
        const std::vector<std::shared_ptr<Buffer>>& buffers
    ) {
        if (!this->dynamicResourcesPtr) {
            throw std::runtime_error("Dynamic resources pointer is null.");
        }
        this->dynamicResourcesPtr->setBuffers(contextPtr, buffers);
    }


}