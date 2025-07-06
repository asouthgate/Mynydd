
# MYNYDD

Mynydd, or Mountain yn Gymraeg, is a library to reduce boilerplate in Vulkan Compute applications. Vulkan, itself meaning Volcano, is useful for cross-platform GPU compute, but sadly is sometimes too scary and complex for those without a lot of time. This package is not designed to replace or really abstract all that much; the user must still understand how pipelines, shaders, buffers and commands work, but with a bit of luck will not require so much typing to start a project. With this simplification, some of Vulkan's power is unsurprisingly lost.

# COMPILATION

To compile this project, run:

```
mkdir build
cd build
cmake ..
cmake --build .
ctest --verbose
```
