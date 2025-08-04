#include <array>
#include <assert.h>
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

        // Check for a suitable physical device with compute capabilities
        // Iterate through the devices and find one with a compute queue
        // computeQueueFamilyIndex will be set to the index of the compute queue
        // family This will pick the first device that has a compute queue
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
                // Set compute queue family index
                // A compute queue family index is a queue family that supports compute
                // operations A queue family is a group of queues that share the same
                // properties A queue is a submission point for commands to the GPU Why
                // do GPUs have multiple queues? Because different types of operations
                // (graphics, compute, transfer) can be executed in parallel Is there
                // one type of queue per operation? No, a queue family can support
                // multiple types of operations What does a queue family correspond to
                // physically? A queue family corresponds to a physical hardware unit
                // that can execute commands How many queue families does a GPU have? It
                // varies by GPU, but typically there are several queue families for
                // different operations
                computeQueueFamilyIndex = i;
                return device;
            }
        }
    }

    throw std::runtime_error("No suitable GPU with compute queue found");
    }

    // 3. Create logical device and get compute queue
    // A logical device is different to a physical device in that
    // it represents an abstraction of the physical device that can be used to
    // interact with the GPU. It allows us to create resources like buffers,
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
    VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {
        VkDescriptorSetLayoutBinding inputBufferBinding{};
        inputBufferBinding.binding = 0;
        inputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        inputBufferBinding.descriptorCount = 1;
        inputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBufferBinding{};
        outputBufferBinding.binding = 1;
        outputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputBufferBinding.descriptorCount = 1;
        outputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding uniformBufferBinding{};
        uniformBufferBinding.binding = 2;
        uniformBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBufferBinding.descriptorCount = 1;
        uniformBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {inputBufferBinding, outputBufferBinding, uniformBufferBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings.data();

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) !=
            VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
        }

        return layout;
    }

    /**
    * Creates a descriptor pool and allocates a descriptor set from it.
    * A descriptor pool is a memory pool that holds descriptors,
    * which are used to bind resources to shaders. These correspond to shader
    * variables that need to be filled with data before dispatching a compute
    * shader. For example, a descriptor set can hold a storage buffer that the
    * compute shader will read from or write to, such as a buffer containing input
    * data or output results, or uniform data that the shader needs to access.
    */
    VkDescriptorSet allocateDescriptorSet(
        VkDevice device,
        VkDescriptorSetLayout layout,
        VkDescriptorPool &pool
    ) {
        std::array<VkDescriptorPoolSize, 3> poolSizes{};

        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;

        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;

        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = 1;


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
        VkBuffer inputBuffer,
        VkDeviceSize inputBufferSize,
        VkBuffer uniformBuffer,
        VkDeviceSize uniformSize,
        VkBuffer outputBuffer,
        VkDeviceSize outputBufferSize
    ) {
        VkDescriptorBufferInfo inputBufferInfo{};
        inputBufferInfo.buffer = inputBuffer;
        inputBufferInfo.offset = 0;
        inputBufferInfo.range = inputBufferSize;

        VkDescriptorBufferInfo outputBufferInfo{};
        outputBufferInfo.buffer = outputBuffer;
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = outputBufferSize;

        // Uniform buffer info
        VkDescriptorBufferInfo uniformBufferInfo{};
        uniformBufferInfo.buffer = uniformBuffer;
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = uniformSize;

        VkWriteDescriptorSet inputWrite{};
        inputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        inputWrite.dstSet = descriptorSet;
        inputWrite.dstBinding = 0; // binding 0: storage buffer
        inputWrite.dstArrayElement = 0;
        inputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        inputWrite.descriptorCount = 1;
        inputWrite.pBufferInfo = &inputBufferInfo;

        VkWriteDescriptorSet outputWrite{};
        outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        outputWrite.dstSet = descriptorSet;
        outputWrite.dstBinding = 1; // binding 0: storage buffer
        outputWrite.dstArrayElement = 0;
        outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputWrite.descriptorCount = 1;
        outputWrite.pBufferInfo = &outputBufferInfo;

        VkWriteDescriptorSet uniformWrite{};
        uniformWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        uniformWrite.dstSet = descriptorSet;
        uniformWrite.dstBinding = 2; // binding 1: uniform buffer
        uniformWrite.dstArrayElement = 0;
        uniformWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformWrite.descriptorCount = 1;
        uniformWrite.pBufferInfo = &uniformBufferInfo;

        std::array<VkWriteDescriptorSet, 3> writes = {
            inputWrite, outputWrite, uniformWrite
        };

        vkUpdateDescriptorSets(
            device,
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr
        );
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
    
    /**
    * Creates a compute pipeline with given shader and descriptor set layout.
    * A compute pipeline is a set of instructions that the GPU will execute for
    * compute operations. For pure compute, it's quite simple, as it only requires
    * a shader module and a descriptor set layout. This is analogous to a data
    * workflow, where the compute shader is the processing step, and the descriptor
    * set layout defines the inputs and outputs of that processing step. Compare
    * this to Nextflow, where a process is defined by a script that specifies how
    * data flows through different steps. A key difference is that in Vulkan, we
    * explicitly create a pipeline layout that includes the descriptor set layout.
    */
    VkPipeline createComputePipeline(
        VkDevice device,
        VkShaderModule shaderModule,
        VkDescriptorSetLayout descriptorSetLayout,
        VkPipelineLayout &pipelineLayout
    ) {

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;

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

    /**
    * Creates a command pool for the given queue family index.
    * A command pool is a memory pool that holds command buffers,
    * which are used to record commands that the GPU will execute.
    * Physically, the command pool resides in the GPU memory and is used to
    * allocate command buffers.
    */
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


    /**
    * Allocates a single command buffer from the given command pool.
    * The pool contains command buffers that can be used to record commands.
    * When we allocate _from_ the command pool, we are essentially reserving
    * a command buffer that we can use to record commands for the GPU to execute.
    */
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
        VkDescriptorSetLayout &descriptorLayout
    ) {
        VkShaderModule shader = loadShaderModule(contextPtr->device, shaderPath);

        VkPipelineLayout pipelineLayout;
        VkPipeline computePipeline = createComputePipeline(
            contextPtr->device, shader, descriptorLayout, pipelineLayout);

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
        // std::shared_ptr<AllocatedBuffer> input,
        // std::shared_ptr<AllocatedBuffer> output,
        // std::shared_ptr<AllocatedBuffer> uniform
        std::vector<std::shared_ptr<AllocatedBuffer>> buffers
    ) : contextPtr(contextPtr) {

        descriptorSetLayout = createDescriptorSetLayout(contextPtr->device);

        descriptorSet = allocateDescriptorSet(contextPtr->device, descriptorSetLayout, descriptorPool);

        updateDescriptorSet(
            contextPtr->device,
            descriptorSet,
            buffers
        );
    }



}
