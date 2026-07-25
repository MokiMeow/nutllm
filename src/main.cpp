/* nutllm milestone 0 — verify and benchmark the compute core.
 *
 *   nutllm            run correctness checks, then benchmark at N=512
 *   nutllm 1024       benchmark at a chosen square size
 *
 * Correctness first: every optimised kernel must agree with the naive one.
 * A fast matmul that is subtly wrong would poison every layer above it. */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cli.hpp"
#include "matmul.hpp"
#include "selftest.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using MatmulFn = void (*)(const Matrix &, const Matrix &, Matrix &);

void fill_random(Matrix &m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < m.size(); i++)
        m.data()[i] = dist(rng);
}

/* Max absolute difference between two matrices. */
double max_diff(const Matrix &x, const Matrix &y) {
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); i++)
        worst = std::max(worst, std::fabs(double(x.data()[i]) - double(y.data()[i])));
    return worst;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/* Time a kernel, taking the best of several runs (least noise). */
double best_seconds(MatmulFn fn, const Matrix &a, const Matrix &b, Matrix &c,
                    int runs) {
    double best = 1e300;
    for (int r = 0; r < runs; r++) {
        const auto start = Clock::now();
        fn(a, b, c);
        best = std::min(best, seconds_since(start));
    }
    return best;
}

double gflops(size_t n, double seconds) {
    /* One multiply and one add per inner iteration: 2*N^3 total. */
    const double flop = 2.0 * double(n) * double(n) * double(n);
    return flop / seconds / 1e9;
}

bool run_correctness() {
    std::printf("correctness (each kernel vs the naive reference)\n");
    bool ok = true;

    for (size_t n : {1u, 7u, 8u, 9u, 64u, 96u}) {
        Matrix a(n, n), b(n, n), reference(n, n), got(n, n);
        fill_random(a, 1234 + unsigned(n));
        fill_random(b, 5678 + unsigned(n));

        matmul_naive(a, b, reference);

        matmul_blocked(a, b, got);
        const double d_blocked = max_diff(reference, got);

        matmul_simd(a, b, got);
        const double d_simd = max_diff(reference, got);

        /* float accumulation reorders, so exact equality is not expected;
         * this tolerance is far tighter than any real error would be. */
        const double tolerance = 1e-3;
        const bool pass = d_blocked < tolerance && d_simd < tolerance;
        ok = ok && pass;

        std::printf("  %-4s n=%-4zu blocked_diff=%.2e simd_diff=%.2e\n",
                    pass ? "ok" : "FAIL", n, d_blocked, d_simd);
    }
    return ok;
}

void run_benchmark(size_t n) {
    Matrix a(n, n), b(n, n), c(n, n);
    fill_random(a, 42);
    fill_random(b, 43);

    /* The naive kernel is O(n^3) with terrible locality; keep its run count
     * low so the benchmark stays quick at large n. */
    const int naive_runs = n <= 512 ? 3 : 1;
    const int fast_runs = 5;

    const double t_naive = best_seconds(matmul_naive, a, b, c, naive_runs);
    const double t_blocked = best_seconds(matmul_blocked, a, b, c, fast_runs);
    const double t_simd = best_seconds(matmul_simd, a, b, c, fast_runs);

    std::printf("\nbenchmark  %zux%zu x %zux%zu  (best of %d/%d runs)\n",
                n, n, n, n, naive_runs, fast_runs);
    std::printf("  %-10s %9s %9s %8s\n", "kernel", "ms", "GFLOP/s", "speedup");
    std::printf("  %-10s %9.2f %9.2f %7.1fx\n", "naive",
                t_naive * 1e3, gflops(n, t_naive), 1.0);
    std::printf("  %-10s %9.2f %9.2f %7.1fx\n", "blocked",
                t_blocked * 1e3, gflops(n, t_blocked), t_naive / t_blocked);
    std::printf("  %-10s %9.2f %9.2f %7.1fx\n", simd_available() ? "simd" : "simd*",
                t_simd * 1e3, gflops(n, t_simd), t_naive / t_simd);
    if (!simd_available())
        std::printf("  * built without AVX2/FMA; simd fell back to blocked\n");
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--model")
        return run_model_cli(argc, argv);

    size_t n = 512;
    if (argc > 1)
        n = static_cast<size_t>(std::strtoul(argv[1], nullptr, 10));
    if (n == 0) {
        std::fprintf(stderr, "usage: nutllm [matrix-size]\n");
        return 2;
    }

    std::printf("nutllm correctness gate (AVX2/FMA: %s)\n\n",
                simd_available() ? "yes" : "no");

    if (!run_correctness() || !run_ops_selftests() ||
        !run_transformer_selftests() || !run_loader_selftests() ||
        !run_kvcache_selftests()) {
        std::fprintf(stderr, "\nFAIL: a kernel disagrees with the reference\n");
        return 1;
    }

    run_kvcache_benchmark();
    run_benchmark(n);
    return 0;
}
