#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mynydd/shader_interop.hpp>
#include "shaders/kernels.comp.kern"

TEST_CASE("Shader kernels produce expected results", "[shader_kernels]") {
    REQUIRE(cubic_spline_2d_kernel(0.50, 1.0) == Catch::Approx(0.71875 * cubic_spline_2d_fac(1.0)));
    REQUIRE(cubic_spline_2d_kernel(1.00, 1.0) == 0.25 * cubic_spline_2d_fac(1.0));
    REQUIRE(cubic_spline_2d_kernel(1.50, 1.0) == 0.03125 * cubic_spline_2d_fac(1.0));
    REQUIRE(cubic_spline_2d_kernel(2.00, 1.0) == 0.0);
    REQUIRE(cubic_spline_2d_kernel(2.01, 1.0) == 0.0);
    REQUIRE(cubic_spline_2d_kernel(10.0, 1.0) == 0.0);
    SUCCEED("Compute shader executed for 1.0/floats.");
}