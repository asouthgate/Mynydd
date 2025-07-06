
# MYNYDD

`mynydd`, or Mountain yn Gymraeg, is a library to reduce boilerplate in Vulkan Compute applications. Vulkan, named after 'volcano', is useful for cross-platform GPU compute, but sadly is too complex for easy use.

This package is not designed to replace or really abstract all that much; the user must still understand how pipelines, buffers and commands work, but with a bit of luck will not require so much typing to start a project.

# COMPILATION

To compile this project, run:

```
mkdir build
cd build
cmake ..
cmake --build .
ctest --verbose
```
