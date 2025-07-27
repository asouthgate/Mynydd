cmake_minimum_required(VERSION 3.10)
project(nanovulkan)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Vulkan REQUIRED)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)


add_library(mynydd SHARED
    ${SOURCE_DIR}/mynydd.cpp
    ${SOURCE_DIR}/memory.cpp
    ${SOURCE_DIR}/pipelines/radix_sort.cpp
)

target_include_directories(mynydd PUBLIC ${INCLUDE_DIR})
target_link_libraries(mynydd PRIVATE Vulkan::Vulkan)

# tests

find_package(Catch2 REQUIRED)
set(TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/tests)
enable_testing()

# test shaders

set(TEST_SHADER_DIR ${TEST_SRC_DIR}/shaders)
set(SRC_SHADER_DIR ${SOURCE_DIR}/pipelines/shaders)

set(SPIRV_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})
file(GLOB SHADER_SRC_FILES "${SRC_SHADER_DIR}/*.comp" "${TEST_SHADER_DIR}/*.comp")
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

add_executable(tests
    ${TEST_SRC_DIR}/test_morton_helpers.cpp 
    ${TEST_SRC_DIR}/test_morton.cpp
    ${TEST_SRC_DIR}/test_mynydd.cpp
    ${TEST_SRC_DIR}/test_shader_kernels.cpp
    ${TEST_SRC_DIR}/test_radix_sort.cpp
    ${TEST_SRC_DIR}/test_particle_index.cpp
    ${TEST_SRC_DIR}/test_transpose.cpp
    ${TEST_SRC_DIR}/test_workgroup_scan.cpp
    ${SHADER_SPV_FILES}
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Vulkan::Vulkan mynydd)

add_dependencies(tests compile_shaders)

add_test(NAME compute_tests COMMAND tests "[vulkan]")
add_test(NAME morton COMMAND tests "[morton]")
add_test(NAME shader_kernel_tests COMMAND tests "[shader_kernels]")
add_test(NAME radix_sort COMMAND tests "[sort]")
add_test(NAME morton_sort COMMAND tests "[morton_sort]")
add_test(NAME transpose COMMAND tests "[transpose]")
add_test(NAME index COMMAND tests "[index]")


# === Compile example folders ===

file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_SOURCE_DIR}/examples ${CMAKE_SOURCE_DIR}/examples/*)

foreach(EXAMPLE ${EXAMPLE_DIRS})
    set(EXAMPLE_PATH ${CMAKE_SOURCE_DIR}/examples/${EXAMPLE})
    set(EXAMPLE_SHADER_DIR ${EXAMPLE_PATH})
    set(EXAMPLE_SPIRV_DIR ${CMAKE_BINARY_DIR}/examples/${EXAMPLE})
    file(MAKE_DIRECTORY ${EXAMPLE_SPIRV_DIR})

    # Collect shaders
    file(GLOB EXAMPLE_SHADER_SRC_FILES "${EXAMPLE_SHADER_DIR}/*.comp")
    set(EXAMPLE_SHADER_SPV_FILES)

    foreach(SHADER ${EXAMPLE_SHADER_SRC_FILES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPIRV_OUT "${EXAMPLE_SPIRV_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPIRV_OUT}
            COMMAND glslangValidator -V ${SHADER} -o ${SPIRV_OUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${EXAMPLE}/${SHADER_NAME} to SPIR-V"
            VERBATIM
        )

        list(APPEND EXAMPLE_SHADER_SPV_FILES ${SPIRV_OUT})
    endforeach()

    add_custom_target(compile_${EXAMPLE}_shaders ALL DEPENDS ${EXAMPLE_SHADER_SPV_FILES})

    # Add executable
    add_executable(${EXAMPLE}_example ${EXAMPLE_PATH}/main.cpp ${EXAMPLE_SHADER_SPV_FILES})

    target_include_directories(${EXAMPLE}_example PRIVATE ${INCLUDE_DIR})
    target_link_libraries(${EXAMPLE}_example PRIVATE Vulkan::Vulkan mynydd)

    add_dependencies(${EXAMPLE}_example compile_${EXAMPLE}_shaders)
endforeach()