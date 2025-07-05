cmake_minimum_required(VERSION 3.10)
project(nanovulkan)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHADER_DIR ${SOURCE_DIR}/shad)
set(SPIRV_DIR ${CMAKE_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SPIRV_DIR})

find_package(Vulkan REQUIRED)

set(SHADER_SRC ${SHADER_DIR}/shader.comp)
set(SHADER_SPV ${SPIRV_DIR}/shader.comp.spv)

add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND glslangValidator -V ${SHADER_SRC} -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader to SPIR-V"
)

add_library(compute_context SHARED
    ${SOURCE_DIR}/compute_context.cpp
)

target_include_directories(compute_context PUBLIC ${SOURCE_DIR})
target_link_libraries(compute_context PRIVATE Vulkan::Vulkan)

add_executable(nanovulkan
    ${SOURCE_DIR}/main.cpp
    ${SHADER_SPV}
)

target_include_directories(nanovulkan PRIVATE ${SOURCE_DIR})
target_link_libraries(nanovulkan PRIVATE Vulkan::Vulkan compute_context)

# tests

find_package(Catch2 REQUIRED)
enable_testing()

add_executable(tests
    tests/test_compute.cpp
    ${SHADER_SPV}
)

target_include_directories(tests PRIVATE ${SOURCE_DIR})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Vulkan::Vulkan compute_context)

add_test(NAME ComputePipelineTest COMMAND tests)