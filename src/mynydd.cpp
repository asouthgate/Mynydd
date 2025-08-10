#include <array>
#include <assert.h>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include "../include/mynydd/mynydd.hpp"
using namespace mynydd;

namespace mynydd {

    /**
    * Create a Vulkan instance.
    *
    * This function initializes a Vulkan instance with relative parameters,
    * such as application name, versions, and so on.
    */
    VkInstance createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "NanoVulkan";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Custom";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        VkInstance instance;
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        return instance;
    }

    // 2. Select physical device with compute support
    VkPhysicalDevice pickPhysicalDevice(VkInstance instance,
                                        uint32_t &computeQueueFamilyIndex) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
            throw std::runtime_error("No Vulkan-compatible GPUs found");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto &device : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                                    nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                                    queueFamilies.data());

            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            // Print the number of queue families this GPU has
            std::cout << "Number of queue families: " << queueFamilyCount
                        << std::endl;
            // Print the properties of each queue family
            std::cout << "Queue family " << i << ": "
                        << "Count: " << queueFamilies[i].queueCount
                        << ", Flags: " << queueFamilies[i].queueFlags << std::endl;
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(device, &props);
                std::cout << "Selected device: " << props.deviceName << std::endl;
                computeQueueFamilyIndex = i;
                return device;
            }
        }
    }

    throw std::runtime_error("No suitable GPU with compute queue found");
    }

    VkDevice createLogicalDevice(
        VkPhysicalDevice physicalDevice,
        uint32_t computeQueueFamilyIndex,
        VkQueue &computeQueue
    ) {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        VkDevice device;
        if (
            vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) !=
            VK_SUCCESS
        ) {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
        return device;
    }

    VkBuffer createBuffer(
        VkDevice device,
        VkDeviceSize size,
        VkBufferUsageFlags usage
    ) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        return buffer;
    }

    VkDeviceMemory allocateAndBindMemory(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkBuffer buffer,
        VkMemoryPropertyFlags properties
    ) {
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if (
                (memRequirements.memoryTypeBits & (1 << i)) 
                &&
                (memProps.memoryTypes[i].propertyFlags & properties)
                == properties
            ) {
                memoryTypeIndex = i;
                break;
            }
        }

        if (memoryTypeIndex == UINT32_MAX) {
            throw std::runtime_error("Failed to find suitable memory type");
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        VkDeviceMemory memory;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory");
        }

        vkBindBufferMemory(device, buffer, memory, 0);
        return memory;
    }

    /**
    * Loads a SPIR-V shader module from file.
    */
    VkShaderModule loadShaderModule(VkDevice device, const char *filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader file");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
        file.close();

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = buffer.size() * sizeof(uint32_t);
        createInfo.pCode = buffer.data();

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
            VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
        }

        return shaderModule;
    }

    /**
    * Creates a descriptor set layout with one storage buffer binding.
    * A descriptor set layout defines the structure of a descriptor set,
    * which is used to bind resources (like buffers) to shaders.
    * In turn, a descriptor set is a collection of descriptors that describe
    * resources, for example, a storage buffer that can be accessed by a compute
    * shader. We have to inform the GPU about the resources that will be used in
    * the compute shader. This is analogous to telling a scheduler like SLURM what
    * resources (like CPUs, memory) a job will need.
    */
    VkDescriptorSetLayout createDescriptorSetLayout(
        VkDevice device,
        const std::vector<std::shared_ptr<AllocatedBuffer>>& buffers
) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        size_t bindingIndex = 0;
        for (const auto &buffer : buffers) {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = bindingIndex++;
            binding.descriptorType = buffer->getType();
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(binding);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) !=
            VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
        }

        return layout;
    }

    VkDescriptorSet allocateDescriptorSet(
        VkDevice device,
        VkDescriptorSetLayout layout,
        VkDescriptorPool &pool,
        const std::vector<std::shared_ptr<AllocatedBuffer>>& buffers
    ) {
        std::vector<VkDescriptorPoolSize> poolSizes(buffers.size());

        for (size_t i = 0; i < buffers.size(); ++i) {
            poolSizes[i].type = buffers[i]->getType();
            poolSizes[i].descriptorCount = 1;
        }

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1; // One descriptor set

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;

        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        return descriptorSet;
    }

    /**
    * Binds a buffer to the given descriptor set at binding 0.
    * The descriptor set contains information about the resources that the compute
    * shader will use. In this case, the buffer will be used as a storage buffer,
    * which means it can be read from and written to by the compute shader. For our
    * GPU compute abstraction, this will correspond to dtypes that the compute
    * shader will process.
    */
    void updateDescriptorSet(
        VkDevice device,
        VkDescriptorSet descriptorSet,
        const std::vector<std::shared_ptr<AllocatedBuffer>> &buffers
    ) {
        if (buffers.empty()) {
            throw std::runtime_error("No buffers provided for descriptor set update");
        }

        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> bufferInfos; 
        bufferInfos.reserve(buffers.size());

        for (const auto &buffer : buffers) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = buffer->getBuffer();
            bufferInfo.offset = 0;
            bufferInfo.range = buffer->getSize();

            bufferInfos.push_back(bufferInfo);

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            write.dstBinding = static_cast<uint32_t>(writes.size()); // use different bindings if needed
            write.dstArrayElement = 0;
            write.descriptorType = buffer->getType();
            write.descriptorCount = 1;
            write.pBufferInfo = &bufferInfos.back();

            writes.push_back(write);
        }

        vkUpdateDescriptorSets(
            device,
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr
        );
    }
    
    VkPipeline createComputePipeline(
        VkDevice device,
        VkShaderModule shaderModule,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout &pipelineLayout,
        std::vector<uint32_t> pushConstantSizes = {}
    ) {

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;

        if (!pushConstantSizes.empty()) {
            std::vector<VkPushConstantRange> ranges(pushConstantSizes.size());
            for (size_t j = 0; j < pushConstantSizes.size(); ++j) {
                ranges[j].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                ranges[j].offset = j == 0 ? 0 : ranges[j - 1].offset + ranges[j - 1].size;
                ranges[j].size = pushConstantSizes[j];
            }

            layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantSizes.size());
            layoutInfo.pPushConstantRanges = ranges.data();
        }
        
        if (
            vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) !=
            VK_SUCCESS
        ) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VkPipeline pipeline;
        if (
            vkCreateComputePipelines(
                device,
                VK_NULL_HANDLE,
                1,
                &pipelineInfo,
                nullptr,
                &pipeline
            ) != VK_SUCCESS
        ) {
            throw std::runtime_error("Failed to create compute pipeline");
        }

        return pipeline;
    }

    VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;

        VkCommandPool commandPool;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        return commandPool;
    }

    VkCommandBuffer allocateCommandBuffer(VkDevice device, VkCommandPool pool) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuffer;
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        return cmdBuffer;
    }


    /**
    * Records commands to bind the pipeline and dispatch the compute shader.
    * What commands are recorded? We bind the compute pipeline, bind the descriptor set,
    * and dispatch the compute shader. This is analogous to preparing a job script
    * that specifies what resources (like input data) the job will use and how it
    * will be executed.
    */
    void recordCommandBuffer(VkCommandBuffer cmdBuffer, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet descriptorSet, uint32_t numElements) {
        
        if (cmdBuffer == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid command buffer handle");
        }
        if (pipeline == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid pipeline handle");
        }
        if (layout == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid pipeline layout handle");
        }
        if (descriptorSet == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid descriptor set handle");
        }
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VkResult result = vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("vkBeginCommandBuffer failed with error: " + std::to_string(result));
        }

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, nullptr);

        uint32_t groupCount = (numElements + 63) / 64; // match local_size_x=64 in shader
        vkCmdDispatch(cmdBuffer, groupCount, 1, 1);

        result = vkEndCommandBuffer(cmdBuffer);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("vkEndCommandBuffer failed with error: " + std::to_string(result));
        }
    }


    /**
    * Submits the command buffer and waits for execution to complete.
    */
    void submitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmdBuffer) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        // This fence is used to synchronize the command buffer execution
        // Specifically, 
        // it allows us to wait for the command buffer to finish executing,
        // before we proceed to read the results from the buffer.
        VkFence fence;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fenceInfo, nullptr, &fence);

        if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit command buffer");
        }

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device, fence, nullptr);
    }


    VulkanPipelineResources create_pipeline_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        const char* shaderPath,
        VkDescriptorSetLayout &descriptorLayout,
        std::vector<uint32_t> pushConstantSizes
    ) {
        VkShaderModule shader = loadShaderModule(contextPtr->device, shaderPath);

        VkPipelineLayout pipelineLayout;
        VkPipeline computePipeline = createComputePipeline(
            contextPtr->device, 
            shader,
            descriptorLayout,
            pipelineLayout,
            pushConstantSizes
        );

        return {
            pipelineLayout,
            computePipeline,
            shader
        };
    }

    VulkanContext::VulkanContext() {
        instance = createInstance();
        physicalDevice =
            pickPhysicalDevice(instance, computeQueueFamilyIndex);

        device = createLogicalDevice(
            physicalDevice,
            computeQueueFamilyIndex,
            computeQueue
        );

        commandPool = createCommandPool(
            device, computeQueueFamilyIndex
        );

        commandBuffer = allocateCommandBuffer(
            device, commandPool
        );
    }

    VulkanDynamicResources::VulkanDynamicResources(
        std::shared_ptr<VulkanContext> contextPtr,
        const std::vector<std::shared_ptr<AllocatedBuffer>> buffers
    ) : contextPtr(contextPtr) {

        descriptorSetLayout = createDescriptorSetLayout(contextPtr->device, buffers);

        descriptorSet = allocateDescriptorSet(contextPtr->device, descriptorSetLayout, descriptorPool, buffers);

        updateDescriptorSet(
            contextPtr->device,
            descriptorSet,
            buffers
        );
    }

    void VulkanDynamicResources::setBuffers(
        std::shared_ptr<VulkanContext> contextPtr,
        const std::vector<std::shared_ptr<AllocatedBuffer>>& buffers
    ) {
        if (!contextPtr || contextPtr->device == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid Vulkan context in setBuffers.");
        }
        if (buffers.empty()) {
            throw std::runtime_error("No buffers provided to setBuffers.");
        }

        this->descriptorSet = allocateDescriptorSet(
            contextPtr->device, this->descriptorSetLayout, this->descriptorPool, buffers
        );
        updateDescriptorSet(contextPtr->device, this->descriptorSet, buffers);
    }




}
