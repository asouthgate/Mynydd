// This header provides compatibility for shader kernels so that they can be tested in C++ code.
// This is not perfect, and should not be relied on too extensively
// The best tests will use a GPU context
// Fine-grained kernel logic can be interoperated

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/compatibility.hpp>
#include <cmath>

namespace mynydd_shader_interoperability {
    using vec2 = glm::vec2;
    using vec3 = glm::vec3;
    using vec4 = glm::vec4;
    using mat2 = glm::mat2;
    using mat3 = glm::mat3;
    using mat4 = glm::mat4;

    // Scalar function helpers
    inline float fract(float x) { return x - std::floor(x); }
    inline float mod(float x, float y) { return x - y * std::floor(x / y); }
    inline float clamp(float x, float minVal, float maxVal) { return glm::clamp(x, minVal, maxVal); }
    inline float mix(float a, float b, float t) { return glm::mix(a, b, t); }
    inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
    inline float smoothstep(float edge0, float edge1, float x) {
        return glm::smoothstep(edge0, edge1, x);
    }
    inline float sign(float x) { return (x > 0.0f) - (x < 0.0f); }

    using glm::pow;
    using glm::exp;
    using glm::log;
    using glm::sin;
    using glm::cos;
    using glm::tan;
    using glm::abs;
    using glm::sqrt;
    using glm::floor;
    using glm::ceil;
    using glm::fract;
    using glm::mod;
    using glm::clamp;
    using glm::mix;
    using glm::step;
    using glm::smoothstep;
    using glm::sign;
    using glm::dot;
    using glm::normalize;
    using glm::length;
    using glm::cross;
    using glm::max;
}

using namespace mynydd_shader_interoperability;
