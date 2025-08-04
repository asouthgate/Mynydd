#pragma once

#include <assert.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>


namespace mynydd {
    struct VulkanContext;
    
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
}