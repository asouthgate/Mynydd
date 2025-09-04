#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

template<typename T>
void requireNotJustZeroes(const std::vector<T>& data) {
    uint32_t nonZeroCount = 0;
    for (const auto& v : data) {
        if (v != T(0)) {
            nonZeroCount++;
        }
    }
    REQUIRE(nonZeroCount > 0);
}

#endif