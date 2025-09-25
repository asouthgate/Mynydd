#include "../include/mynydd/mynydd.hpp"

namespace mynydd {
    // TODO: should store pointer to context, not device; in fact device should be private
    Buffer::Buffer(std::shared_ptr<VulkanContext> vkc, size_t size, bool uniform)
        : device(vkc->device), size(size) 
    {

        if (uniform) {
            type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        }

        VkBuffer newBuffer = createBuffer(
            device, size,
            uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        );

        VkDeviceMemory newBufferMemory = allocateAndBindMemory(
            vkc->physicalDevice, 
            device,
            newBuffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        this->buffer = newBuffer;
        this->memory = newBufferMemory;
    }

}