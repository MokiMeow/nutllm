/* Correctness tests for transformer tensor operations.
 *
 * These run before every benchmark. Optimised paths are never timed unless
 * they first agree with the permanent scalar/double-precision references. */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ops.hpp"
#include "selftest.hpp"

namespace {

void fill_random(float *values, size_t size, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    for (size_t index = 0; index < size; index++)
        values[index] = distribution(random);
}

double max_difference(const float *left, const float *right, size_t size) {
    double difference = 0.0;
    for (size_t index = 0; index < size; index++) {
        difference = std::max(
            difference,
            std::fabs(double(left[index]) - double(right[index])));
    }
    return difference;
}

bool check(bool condition, const char *name, double difference = 0.0) {
    std::printf("  %-4s %-24s diff=%.2e\n",
                condition ? "ok" : "FAIL", name, difference);
    return condition;
}

bool test_vector_width_edges() {
    bool passed = true;
    for (size_t size : {7u, 8u, 9u, 33u, 64u}) {
        Matrix logits(2, size);
        fill_random(logits.data(), logits.size(), unsigned(size));
        std::vector<float> reference(logits.size());
        softmax_reference(logits.data(), reference.data(), 2, size);
        softmax_inplace(logits);
        const double softmax_difference =
            max_difference(logits.data(), reference.data(), logits.size());

        std::vector<float> input(size), weight(size), rms_reference(size);
        std::vector<float> rms_output(size);
        fill_random(input.data(), size, unsigned(size + 100));
        fill_random(weight.data(), size, unsigned(size + 200));
        rmsnorm_reference(input.data(), weight.data(), rms_reference.data(),
                          size, 1e-5f);
        rmsnorm(input.data(), weight.data(), rms_output.data(), size, 1e-5f);
        const double rms_difference =
            max_difference(rms_reference.data(), rms_output.data(), size);

        Matrix weights(size + 1, size);
        fill_random(weights.data(), weights.size(), unsigned(size + 300));
        std::vector<float> matvec_reference_output(size + 1);
        std::vector<float> matvec_output(size + 1);
        matvec_reference(weights, input.data(), matvec_reference_output.data());
        matvec(weights, input.data(), matvec_output.data());
        const double matvec_difference =
            max_difference(matvec_reference_output.data(), matvec_output.data(),
                           size + 1);

        char name[64];
        std::snprintf(name, sizeof(name), "width edges n=%zu", size);
        passed = check(
                     softmax_difference < 2e-6 &&
                         rms_difference < 2e-5 &&
                         matvec_difference < 2e-5,
                     name,
                     std::max({softmax_difference, rms_difference,
                               matvec_difference})) &&
                 passed;
    }
    return passed;
}

bool test_hand_fixtures() {
    bool passed = true;

    Matrix large(1, 3);
    large.at(0, 0) = 1000.0f;
    large.at(0, 1) = 1000.0f;
    large.at(0, 2) = 999.0f;
    softmax_inplace(large);
    const float softmax_sum =
        large.at(0, 0) + large.at(0, 1) + large.at(0, 2);
    passed = check(std::isfinite(softmax_sum) &&
                       std::fabs(softmax_sum - 1.0f) < 1e-6f,
                   "softmax large logits",
                   std::fabs(double(softmax_sum) - 1.0)) &&
             passed;

    const float rms_input[] = {3.0f, 4.0f};
    const float rms_weight[] = {1.0f, 2.0f};
    float rms_output[2];
    rmsnorm(rms_input, rms_weight, rms_output, 2, 0.0f);
    const float inverse_rms = 1.0f / std::sqrt(12.5f);
    const double rms_fixture_difference = std::max(
        std::fabs(double(rms_output[0] - 3.0f * inverse_rms)),
        std::fabs(double(rms_output[1] - 8.0f * inverse_rms)));
    passed = check(rms_fixture_difference < 1e-6,
                   "rmsnorm fixture", rms_fixture_difference) &&
             passed;

    Matrix input(1, 2), identity(2, 2), up(2, 2), expected(1, 2), got(1, 2);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = -1.0f;
    identity.zero();
    up.zero();
    identity.at(0, 0) = identity.at(1, 1) = 1.0f;
    up.at(0, 0) = up.at(1, 1) = 2.0f;
    swiglu_reference(input, identity, up, expected);
    swiglu(input, identity, up, got);
    const double swiglu_difference =
        max_difference(expected.data(), got.data(), got.size());
    passed = check(swiglu_difference < 1e-6,
                   "swiglu fixture", swiglu_difference) &&
             passed;

    float query[] = {1.0f, 0.0f};
    float key[] = {0.0f, 1.0f};
    rope(query, key, 1, 2, 1, 10000.0f);
    const float cosine = std::cos(1.0f);
    const float sine = std::sin(1.0f);
    const double rope_difference = std::max({
        std::fabs(double(query[0] - cosine)),
        std::fabs(double(query[1] - sine)),
        std::fabs(double(key[0] + sine)),
        std::fabs(double(key[1] - cosine)),
    });
    passed = check(rope_difference < 1e-6,
                   "rope q/k fixture", rope_difference) &&
             passed;

    float residual[] = {1.0f, 2.0f, 3.0f};
    const float update[] = {0.5f, -1.0f, 4.0f};
    add_inplace(residual, update, 3);
    passed = check(residual[0] == 1.5f && residual[1] == 1.0f &&
                       residual[2] == 7.0f,
                   "residual add fixture") &&
             passed;
    return passed;
}

} // namespace

bool run_ops_selftests() {
    std::printf("\ntensor ops (optimised vs reference, tolerance 2e-5)\n");
    return test_vector_width_edges() && test_hand_fixtures();
}
