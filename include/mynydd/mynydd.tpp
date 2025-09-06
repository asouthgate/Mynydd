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
    std::vector<T> fetchData(std::shared_ptr<VulkanContext> vkc, std::shared_ptr<Buffer> buffer, size_t n_elements) {

        std::vector<T> output = readBufferData<T>(
            vkc->device,
            buffer->getMemory(),
            buffer->getSize(),
            n_elements
        );

        return output;
    }



}