#include <array>
#include <assert.h>
#include <cstddef>
#include <cstdint>
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

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT              messageType,
        const VkDebugUtilsMessengerCallbackDataEXT*  pCallbackData,
        void*                                        pUserData) {

        const char* severity = (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)   ? "ERROR" :
                            (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARN"  :
                            (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)    ? "INFO"  : "VERBOSE";

        std::cerr << "[VULKAN VALIDATION][" << severity << "] " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    // Helper to create debug messenger (requires VK_EXT_debug_utils)
    VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance) {
        VkDebugUtilsMessengerCreateInfoEXT dbgCreateInfo{};
        dbgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        dbgCreateInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        dbgCreateInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dbgCreateInfo.pfnUserCallback = debugCallback;
        dbgCreateInfo.pUserData = nullptr;

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        if (func) {
            func(instance, &dbgCreateInfo, nullptr, &messenger);
        }
        return messenger;
    }

    VkInstance createInstanceWithValidation() {
        // 1) desired layers & extensions
        const char* layerNames[] = { "VK_LAYER_KHRONOS_validation" };

        std::vector<const char*> enabledExtensions;
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "NanoVulkan";
        appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
        appInfo.pEngineName = "Custom";
        appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        // Optional: enable synchronization validation explicitly
        VkValidationFeatureEnableEXT enables[] = {
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
        };
        VkValidationFeaturesEXT validationFeatures{};
        validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationFeatures.enabledValidationFeatureCount = 1;
        validationFeatures.pEnabledValidationFeatures = enables;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = layerNames;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();

        // Chain validation features
        createInfo.pNext = &validationFeatures;

        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

        std::cerr << "Available layers:" << std::endl;
        for (auto& l : layers) {
            std::cerr << "  " << l.layerName << std::endl;
        }

        uint32_t extCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());

        std::cerr << "Available extensions:" << std::endl;
        for (auto& e : exts) {
            std::cerr << "  " << e.extensionName << std::endl;
        }

        VkInstance instance;
        VkResult r = vkCreateInstance(&createInfo, nullptr, &instance);
        if (r != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance with validation");
        }

        // Create debug messenger and keep handle in your VulkanContext
        VkDebugUtilsMessengerEXT dbg = createDebugMessenger(instance);
        (void)dbg; // store dbg in context (so you can destroy it later)

        return instance;
    }

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
            std::cerr << "Number of queue families: " << queueFamilyCount
                        << std::endl;
            // Print the properties of each queue family
            std::cerr << "Queue family " << i << ": "
                        << "Count: " << queueFamilies[i].queueCount
                        << ", Flags: " << queueFamilies[i].queueFlags << std::endl;
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(device, &props);
                std::cerr << "Selected device: " << props.deviceName << std::endl;
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

    VkDescriptorSetLayout createDescriptorSetLayout(
        VkDevice device,
        const std::vector<std::shared_ptr<Buffer>>& buffers
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
        const std::vector<std::shared_ptr<Buffer>>& buffers
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

    void updateDescriptorSet(
        VkDevice device,
        VkDescriptorSet descriptorSet,
        const std::vector<std::shared_ptr<Buffer>> &buffers
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
        VkPhysicalDevice physicalDevice,
        VkShaderModule shaderModule,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout &pipelineLayout,
        std::vector<uint32_t> pushConstantSizes = {}
    ) {

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        uint32_t maxPushConstants = props.limits.maxPushConstantsSize;

        if (!pushConstantSizes.empty()) {
            std::vector<VkPushConstantRange> ranges;
            ranges.reserve(pushConstantSizes.size());

            uint32_t offset = 0;
            for (size_t j = 0; j < pushConstantSizes.size(); ++j) {
                uint32_t s = pushConstantSizes[j];

                if (s == 0) {
                    throw std::runtime_error("Push constant size must be > 0");
                }
                if ((s % 4) != 0) {
                    throw std::runtime_error("Push constant size must be a multiple of 4");
                }
                if (offset + s > maxPushConstants) {
                    throw std::runtime_error("Push constants exceed device maxPushConstantsSize");
                }

                VkPushConstantRange r{};
                r.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                r.offset = offset;
                r.size = s;
                ranges.push_back(r);

                offset += s;
            }

            layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
            layoutInfo.pPushConstantRanges = ranges.data();

            // CreatePipelineLayout must be called while `ranges` is in scope
            if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreatePipelineLayout failed");
            }
        } else {
            // no push constants
            layoutInfo.pushConstantRangeCount = 0;
            layoutInfo.pPushConstantRanges = nullptr;
            if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("vkCreatePipelineLayout failed");
            }
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

        std::cerr << "Using pipeline = " << 
            pipeline << ", layout to bind = " << pipelineLayout << ", descriptorSet = " << descriptorSetLayout << std::endl;

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


    VulkanPipelineResources create_pipeline_resources(
        std::shared_ptr<VulkanContext> contextPtr,
        const char* shaderPath,
        VkDescriptorSetLayout &descriptorLayout,
        std::vector<uint32_t> pushConstantSizes
    ) {
        VkShaderModule shader = loadShaderModule(contextPtr->device, shaderPath);

        std::cerr << "Creating pipeline for shader: " << shaderPath << std::endl;
        VkPipelineLayout pipelineLayout;
        VkPipeline computePipeline = createComputePipeline(
            contextPtr->device, 
            contextPtr->physicalDevice,
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
        instance = createInstanceWithValidation();
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
        const std::vector<std::shared_ptr<Buffer>> buffers
    ) : contextPtr(contextPtr) {

        descriptorSetLayout = createDescriptorSetLayout(contextPtr->device, buffers);

        descriptorSet = allocateDescriptorSet(contextPtr->device, descriptorSetLayout, descriptorPool, buffers);

        updateDescriptorSet(
            contextPtr->device,
            descriptorSet,
            buffers
        );
    }


    void recordCommandBuffer(
        VkCommandBuffer cmdBuffer,
        std::shared_ptr<PipelineStep> pipeline_step,
        bool memory_barrier = true
    ) {
            const auto& pipeline      = pipeline_step->getPipelineResourcesPtr()->pipeline;
            const auto& layout        = pipeline_step->getPipelineResourcesPtr()->pipelineLayout;
            const auto& descriptorSet = pipeline_step->getDynamicResourcesPtr()->descriptorSet;

            std::cerr << "Binding pipeline " << pipeline 
                    << " layout=" << layout 
                    << " descriptorSet=" << descriptorSet << std::endl;
            if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE || descriptorSet == VK_NULL_HANDLE) {
                throw std::runtime_error("Invalid pipeline or descriptor set for engine step.");
            }

            // Bind pipeline and descriptor sets
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, nullptr);


            if (pipeline_step->hasPushConstantData()) {
                PushConstantData pcData = pipeline_step->getPushConstantData();
                uint32_t value = 0;
                std::memcpy(&value, pcData.push_data.data(), sizeof(value));
                vkCmdPushConstants(
                    cmdBuffer,
                    pipeline_step->getPipelineResourcesPtr()->pipelineLayout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    pcData.size,
                    pcData.push_data.data()
                );
            }

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

    PipelineStep::PipelineStep(
        std::shared_ptr<VulkanContext> contextPtr, 
        const char* shaderPath,
        std::vector<std::shared_ptr<Buffer>> buffers,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ,
        std::vector<uint32_t> pushConstantSizes
    ) : contextPtr(contextPtr), groupCountX(groupCountX), groupCountY(groupCountY), groupCountZ(groupCountZ) {  
        this->dynamicResourcesPtr = std::make_shared<mynydd::VulkanDynamicResources>(
            contextPtr,
            buffers
        );
        assert(this->dynamicResourcesPtr->descriptorSetLayout != VK_NULL_HANDLE);
        this->pipelineResources = std::make_shared<VulkanPipelineResources>(
            create_pipeline_resources(contextPtr, shaderPath, this->dynamicResourcesPtr->descriptorSetLayout, pushConstantSizes)
        );
    }

    PipelineStep::~PipelineStep() {
        try {
            if (this->contextPtr && this->contextPtr->device != VK_NULL_HANDLE &&
                this->pipelineResources->pipeline != VK_NULL_HANDLE) {
            } else {
                std::cerr << "Invalid handles in vkDestroyPipeline\n";
                throw std::runtime_error("PipelineStep destructor failed");
            }
            vkDestroyPipeline(this->contextPtr->device, this->pipelineResources->pipeline, nullptr);
            vkDestroyPipelineLayout(this->contextPtr->device, this->pipelineResources->pipelineLayout, nullptr);
            vkDestroyShaderModule(this->contextPtr->device, this->pipelineResources->computeShaderModule, nullptr);
        } catch (const std::exception &e) {
            std::cerr << "Error during PipelineStep destruction: " << e.what() << std::endl;
            throw std::runtime_error("PipelineStep destructor failed");
        }
    }

    void executeBatch(
        std::shared_ptr<VulkanContext> contextPtr,
        const std::vector<std::shared_ptr<PipelineStep>>& PipelineSteps,
        bool beginCommandBuffer
    ) {
        if (PipelineSteps.empty()) {
            throw std::runtime_error("No compute engines provided for batch execution.");
        }

        if (!contextPtr || contextPtr->device == VK_NULL_HANDLE) {
            throw std::runtime_error("Invalid Vulkan context in batch execution.");
        }

        VkCommandBuffer cmdBuffer = contextPtr->commandBuffer;

        if (beginCommandBuffer) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("Failed to begin command buffer for batch execution.");
            }        
        } // else already begun
        
        for (size_t i = 0; i < PipelineSteps.size(); ++i) {
            auto& pipelineStep = PipelineSteps[i];
            if (!pipelineStep) {
                throw std::runtime_error("Null PipelineStep pointer at index " + std::to_string(i));
            }
            recordCommandBuffer(
                cmdBuffer,
                pipelineStep,
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
        vkWaitForFences(contextPtr->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(contextPtr->device, fence, nullptr);
    }


}
