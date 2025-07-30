#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

#include <catch2/catch_approx.hpp>

#include <mynydd/shader_interop.hpp>
#include "kernels.comp.kern"

TEST_CASE("test_spiky_kernel", "[sph]") {
    float h = 1.329f;
    float r = 0.39881f;

    // CHECK(std::fabs(debrun_spiky_kernel(r, h) - 0.6179313391538699f) < 1e-7f);
    CHECK(debrun_spiky_kernel(r, h) == Catch::Approx(0.6179313391538699f).margin(1e-7f));
    REQUIRE(debrun_spiky_kernel(-0.000001f, h) == 0.0f);
    REQUIRE(debrun_spiky_kernel(1.33f, h) == 0.0f);
}

TEST_CASE("test_kernel_dwdr", "[sph]") {
    float h = 1.329f;
    CHECK(std::fabs(debrun_spiky_kernel_dwdr(0.0f, h)) > 1.0f); // doesn't disappear at origin
}

TEST_CASE("test_kernel_grad", "[sph]") {
    float dx = 0.1361f;
    float dy = 0.9981f;
    float h = 1.8f;
    float r = cal_r(dx, dy);

    auto grad = debrun_spiky_kernel_grad(dx, dy, h);

    CHECK(std::fabs(grad.x - dx * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
    CHECK(std::fabs(grad.y - dy * debrun_spiky_kernel_dwdr(r, h) / r) < 1e-6f);
    CHECK(grad.x < 0.0f); // we expect the gradient to be pointing downwards
    CHECK(grad.y < 0.0f);
}

TEST_CASE("test_debrun_spiky_kernel_lap", "[sph]") {
    float r = 4.601086828130937f;
    float h = 1.2f;
    float l = debrun_spiky_kernel_lap(r, h);

    CHECK(std::fabs(-27.948690054856538f * get_debrun_coeff(h) - l) < 1e-5f);
}
