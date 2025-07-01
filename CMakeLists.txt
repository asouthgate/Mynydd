cmake_minimum_required(VERSION 3.10)
project(nanovulkan)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHADER_DIR ${SOURCE_DIR}/shad)
set(SPIRV_DIR ${CMAKE_BINARY_DIR}/shaders)

# Output shader.spv to build directory
file(MAKE_DIRECTORY ${SPIRV_DIR})

# Add Vulkan
find_package(Vulkan REQUIRED)

# Shader compilation rule
set(SHADER_SRC ${SHADER_DIR}/shader.glsl)
set(SHADER_SPV ${SPIRV_DIR}/shader.glsl.spv)

add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND glslangValidator -V ${SHADER_SRC} -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader to SPIR-V"
)

# Add an executable that depends on shader compilation
add_executable(nanovulkan ${SOURCE_DIR}/main.cpp ${SHADER_SPV})
target_include_directories(nanovulkan PRIVATE ${SOURCE_DIR})
target_link_libraries(nanovulkan PRIVATE Vulkan::Vulkan)
