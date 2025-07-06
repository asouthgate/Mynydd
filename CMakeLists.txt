cmake_minimum_required(VERSION 3.10)
project(nanovulkan)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(SHADER_DIR ${SOURCE_DIR}/shad)
set(SPIRV_DIR ${CMAKE_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SPIRV_DIR})

find_package(Vulkan REQUIRED)

file(GLOB SHADER_SRC_FILES "${SHADER_DIR}/*.comp")

message(STATUS "Found shaders: ${SHADER_SRC_FILES}")

set(SHADER_SPV_FILES)

foreach(SHADER ${SHADER_SRC_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    message(STATUS "Adding shader: ${SHADER_NAME}")
    set(SPIRV_OUT "${SPIRV_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPIRV_OUT}
        COMMAND glslangValidator -V ${SHADER} -o ${SPIRV_OUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
        VERBATIM
    )
    
    list(APPEND SHADER_SPV_FILES ${SPIRV_OUT})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SHADER_SPV_FILES})

add_library(compute_context SHARED
    ${SOURCE_DIR}/compute_context.cpp
)

target_include_directories(compute_context PUBLIC ${SOURCE_DIR})
target_link_libraries(compute_context PRIVATE Vulkan::Vulkan)

# tests

find_package(Catch2 REQUIRED)
enable_testing()

add_executable(tests
    tests/test_compute.cpp
    ${SHADER_SPV_FILES}     # Use correct variable here
)

target_include_directories(tests PRIVATE ${SOURCE_DIR})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Vulkan::Vulkan compute_context)

add_dependencies(tests compile_shaders)

add_test(NAME ComputePipelineTest COMMAND tests)