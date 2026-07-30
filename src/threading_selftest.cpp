#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "matmul.hpp"
#include "ops.hpp"
#include "selftest.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void fill_random(Matrix &matrix, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t index = 0; index < matrix.size(); index++)
        matrix.data()[index] = distribution(rng);
}

float maximum_difference(const float *left, const float *right, size_t size) {
    float difference = 0.0f;
    for (size_t index = 0; index < size; index++)
        difference =
            std::max(difference, std::fabs(left[index] - right[index]));
    return difference;
}

template <typename Function>
double median_seconds(Function function, size_t runs) {
    std::vector<double> times;
    times.reserve(runs);
    for (size_t run = 0; run < runs; run++) {
        const auto start = Clock::now();
        function();
        times.push_back(
            std::chrono::duration<double>(Clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

bool run_threading_selftests() {
    Matrix left(9, 17);
    Matrix right(17, 11);
    Matrix reference(9, 11);
    Matrix candidate(9, 11);
    fill_random(left, 808);
    fill_random(right, 809);
    matmul_blocked(left, right, reference);
    bool matmul_ok = true;
    float worst = 0.0f;
    for (size_t threads : {1u, 2u, 4u, 32u}) {
        matmul_threaded(left, right, candidate, threads);
        const float difference = maximum_difference(
            reference.data(), candidate.data(), reference.size());
        worst = std::max(worst, difference);
        matmul_ok = matmul_ok && difference < 1e-5f;
    }

    Matrix weights(13, 29);
    fill_random(weights, 810);
    std::vector<float> vector(weights.cols(), 0.25f);
    std::vector<float> matvec_reference_output(weights.rows());
    std::vector<float> matvec_candidate(weights.rows());
    matvec(weights, vector.data(), matvec_reference_output.data());
    matvec_threaded(weights, vector.data(), matvec_candidate.data(), 4);
    const float matvec_difference =
        maximum_difference(matvec_reference_output.data(),
                           matvec_candidate.data(), weights.rows());
    const bool matvec_ok = matvec_difference < 1e-5f;
    const size_t workers_after_warmup = threaded_worker_start_count();
    matvec_threaded(weights, vector.data(), matvec_candidate.data(), 4);
    const bool worker_reuse_ok =
        threaded_worker_start_count() == workers_after_warmup;

    std::vector<float> linear_input(weights.rows(), 0.125f);
    std::vector<float> linear_reference(weights.cols(), 0.0f);
    std::vector<float> linear_candidate(weights.cols());
    for (size_t row = 0; row < weights.rows(); row++) {
        for (size_t column = 0; column < weights.cols(); column++)
            linear_reference[column] +=
                linear_input[row] * weights.at(row, column);
    }
    linear_threaded(linear_input.data(), weights, linear_candidate.data(), 3);
    const float linear_difference =
        maximum_difference(linear_reference.data(), linear_candidate.data(),
                           weights.cols());
    const bool linear_ok = linear_difference < 1e-5f;
    std::printf(
        "  %s threaded row splits          gemm=%.2e matvec=%.2e "
        "linear=%.2e workers=%s\n",
        matmul_ok && matvec_ok && linear_ok && worker_reuse_ok ?
            "ok" : "FAIL",
        double(worst), double(matvec_difference), double(linear_difference),
        worker_reuse_ok ? "reused" : "recreated");
    return matmul_ok && matvec_ok && linear_ok && worker_reuse_ok;
}

void run_threading_benchmark() {
    const size_t maximum_threads =
        std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    Matrix left(768, 768);
    Matrix right(768, 768);
    Matrix output(768, 768);
    fill_random(left, 811);
    fill_random(right, 812);
    Matrix weights(4096, 4096);
    fill_random(weights, 813);
    std::vector<float> vector(weights.cols(), 0.125f);
    std::vector<float> result(weights.rows());
    struct Timing {
        size_t threads;
        double gemm;
        double matvec;
    };
    std::vector<Timing> timings;
    for (size_t threads : {1u, 2u, 4u}) {
        if (threads > maximum_threads)
            continue;
        const double gemm = median_seconds(
            [&]() { matmul_threaded(left, right, output, threads); }, 3);
        const double matvec_time = median_seconds(
            [&]() {
                matvec_threaded(weights, vector.data(), result.data(),
                                threads);
            },
            5);
        timings.push_back({threads, gemm, matvec_time});
    }
    const double gemm_baseline = timings.front().gemm;
    const double matvec_baseline = timings.front().matvec;

    std::printf("\nthread scaling (768 GEMM / 4096 matvec, median)\n");
    std::printf("  threads   GEMM ms  speedup  efficiency   matvec ms  speedup\n");
    for (const Timing &timing : timings) {
        std::printf("  %-7zu %9.2f %8.2fx %10.1f%% %10.2f %8.2fx\n",
                    timing.threads, timing.gemm * 1e3,
                    gemm_baseline / timing.gemm,
                    100.0 * gemm_baseline / timing.gemm /
                        double(timing.threads),
                    timing.matvec * 1e3,
                    matvec_baseline / timing.matvec);
    }
}
