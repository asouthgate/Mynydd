#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

struct VulkanContext {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue computeQueue;
  uint32_t computeQueueFamilyIndex;
};

// 1. Create Vulkan instance
VkInstance createInstance() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Compute Shader Example";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
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
VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice,
                             uint32_t computeQueueFamilyIndex,
                             VkQueue &computeQueue) {
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
  if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) !=
      VK_SUCCESS)
    throw std::runtime_error("Failed to create logical device");

  vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
  return device;
}

VkBuffer createBuffer(VkDevice device, VkDeviceSize size,
                      VkBufferUsageFlags usage) {
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

VkDeviceMemory allocateAndBindMemory(VkPhysicalDevice physicalDevice,
                                     VkDevice device, VkBuffer buffer,
                                     VkMemoryPropertyFlags properties) {
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

  uint32_t memoryTypeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
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
  if (!file.is_open())
    throw std::runtime_error("Failed to open shader file");

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
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = 0;
  layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  layoutBinding.descriptorCount = 1;
  layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &layoutBinding;

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
VkDescriptorSet allocateDescriptorSet(VkDevice device,
                                      VkDescriptorSetLayout layout,
                                      VkDescriptorPool &pool) {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;

  VkDescriptorSet descriptorSet;
  if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor set");
  }

  return descriptorSet;
}

/**
 * Binds a buffer to the given descriptor set at binding 0.
 * The descriptor set contains information about the resources that the compute
 * shader will use. In this case, the buffer will be used as a storage buffer,
 * which means it can be read from and written to by the compute shader. For our
 * GPU compute abstraction, this will correspond to floats that the compute
 * shader will process.
 */
void updateDescriptorSet(VkDevice device, VkDescriptorSet descriptorSet,
                         VkBuffer buffer, VkDeviceSize size) {
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = size;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descriptorSet;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
VkPipeline createComputePipeline(VkDevice device, VkShaderModule shaderModule,
                                 VkDescriptorSetLayout descriptorSetLayout,
                                 VkPipelineLayout &pipelineLayout) {
  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &descriptorSetLayout;

  if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shaderModule;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = pipelineLayout;

  VkPipeline pipeline;
  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &pipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create compute pipeline");
  }

  return pipeline;
}

// High-level setup
VulkanContext initVulkan() {
  VulkanContext context;

  context.instance = createInstance();
  context.physicalDevice =
      pickPhysicalDevice(context.instance, context.computeQueueFamilyIndex);
  context.device = createLogicalDevice(context.physicalDevice,
                                       context.computeQueueFamilyIndex,
                                       context.computeQueue);

  return context;
}

int main() {
  try {
    VulkanContext context = initVulkan();

    const size_t dataSize = 1024 * sizeof(float);

    // A buffer is a memory object that can be used to store data
    // It can be used for various purposes, such as storing vertex data, index
    // data, uniform data, etc. It physically resides in the GPU memory Data is
    // transferred to the GPU memory using a staging buffer or directly from the
    // host memory The staging buffer is a temporary buffer that is used to
    // transfer data from the host memory to the GPU memory The staging buffer
    // lives in the host memory and is used to copy data to the GPU memory
    VkBuffer buffer = createBuffer(context.device, dataSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkDeviceMemory bufferMemory =
        allocateAndBindMemory(context.physicalDevice, context.device, buffer,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    const char *shaderPath =
        "shader.comp.spv"; // SPIR-V compiled compute shader

    VkShaderModule shader = loadShaderModule(context.device, shaderPath);
    VkDescriptorSetLayout descriptorLayout =
        createDescriptorSetLayout(context.device);

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet =
        allocateDescriptorSet(context.device, descriptorLayout, descriptorPool);

    updateDescriptorSet(context.device, descriptorSet, buffer, dataSize);

    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline = createComputePipeline(
        context.device, shader, descriptorLayout, pipelineLayout);

    // Clean up
    vkDestroyPipeline(context.device, computePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptorLayout, nullptr);
    vkDestroyShaderModule(context.device, shader, nullptr);
    vkDestroyDevice(context.device, nullptr);
    vkDestroyInstance(context.instance, nullptr);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}