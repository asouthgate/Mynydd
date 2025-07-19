
# MYNYDD

Mynydd, meaning mountain, is a library to reduce boilerplate for those seeking to use Vulkan Compute functionality for scientific code. Vulkan, itself meaning Volcano, is useful for cross-platform GPU compute, but sadly is sometimes too scary and complex for those without a lot of time. This package is not designed to replace or really abstract all that much; the user must still understand how pipelines, shaders, buffers and commands work, but with a bit of luck will not require so much typing to start a project.

# WHY VULKAN?

* We may prefer to not be tied to a given GPU vendor
* We may want to be reliably cross platform
* Vulkan is a good skill to learn & a lightweight wrapper doesn't get in the way of this learning
* Vulkan graphics functionality can be mixed in if desired

# BUILD

To build this project, run:

```
mkdir build
cd build
cmake ..
cmake --build .
ctest --verbose
```
