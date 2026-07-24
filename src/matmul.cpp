/* The compute core of the inference engine.
 *
 * Every transformer is, in wall-clock terms, a pile of matrix multiplies. So
 * the first thing worth building — and measuring — is matmul itself. The three
 * versions here are the same mathematics with progressively better memory
 * behaviour, which is where nearly all of the speedup comes from. */

#include <algorithm>
#include <stdexcept>

#include "matmul.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define NUTLLM_HAVE_AVX2 1
#endif

namespace {

void check_shapes(const Matrix &a, const Matrix &b, const Matrix &c) {
    if (a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols())
        throw std::invalid_argument("matmul: shape mismatch");
}

/* Tile sizes chosen so an A-tile, B-tile and C-tile stay resident in L1/L2.
 * These are the classic starting values; tuning them per machine is a
 * milestone-5 concern. */
constexpr size_t kBlockM = 64;
constexpr size_t kBlockN = 256;
constexpr size_t kBlockK = 128;

} // namespace

bool simd_available() {
#ifdef NUTLLM_HAVE_AVX2
    return true;
#else
    return false;
#endif
}

void matmul_naive(const Matrix &a, const Matrix &b, Matrix &c) {
    check_shapes(a, b, c);
    const size_t M = a.rows(), K = a.cols(), N = b.cols();
    const float *A = a.data();
    const float *B = b.data();
    float *C = c.data();

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            /* B[k * N + j] strides by N floats each step: a new cache line
             * almost every iteration. This is the slow part. */
            for (size_t k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void matmul_blocked(const Matrix &a, const Matrix &b, Matrix &c) {
    check_shapes(a, b, c);
    const size_t M = a.rows(), K = a.cols(), N = b.cols();
    const float *A = a.data();
    const float *B = b.data();
    float *C = c.data();

    c.zero();

    for (size_t i0 = 0; i0 < M; i0 += kBlockM) {
        const size_t i_max = std::min(i0 + kBlockM, M);
        for (size_t k0 = 0; k0 < K; k0 += kBlockK) {
            const size_t k_max = std::min(k0 + kBlockK, K);
            for (size_t j0 = 0; j0 < N; j0 += kBlockN) {
                const size_t j_max = std::min(j0 + kBlockN, N);

                for (size_t i = i0; i < i_max; i++) {
                    for (size_t k = k0; k < k_max; k++) {
                        /* Hoisted out of the j loop: one broadcast value. */
                        const float aik = A[i * K + k];
                        const float *brow = &B[k * N];
                        float *crow = &C[i * N];
                        /* Contiguous walk along j for both B and C. */
                        for (size_t j = j0; j < j_max; j++)
                            crow[j] += aik * brow[j];
                    }
                }
            }
        }
    }
}

#ifdef NUTLLM_HAVE_AVX2
namespace {

/* Fallback for tiles narrower/shorter than the micro-kernel: one row at a
 * time, C reloaded per k. Correct, just not register-blocked. */
void simd_edge_tile(const float *A, const float *B, float *C,
                    size_t K, size_t N,
                    size_t i0, size_t i_max, size_t k0, size_t k_max,
                    size_t j0, size_t j_max) {
    for (size_t i = i0; i < i_max; i++) {
        float *crow = &C[i * N];
        for (size_t k = k0; k < k_max; k++) {
            const float aik = A[i * K + k];
            const __m256 av = _mm256_set1_ps(aik);
            const float *brow = &B[k * N];
            size_t j = j0;
            for (; j + 8 <= j_max; j += 8) {
                __m256 bv = _mm256_loadu_ps(brow + j);
                __m256 cv = _mm256_loadu_ps(crow + j);
                _mm256_storeu_ps(crow + j, _mm256_fmadd_ps(av, bv, cv));
            }
            for (; j < j_max; j++)
                crow[j] += aik * brow[j];
        }
    }
}

} // namespace
#endif

void matmul_simd(const Matrix &a, const Matrix &b, Matrix &c) {
#ifndef NUTLLM_HAVE_AVX2
    matmul_blocked(a, b, c);
#else
    check_shapes(a, b, c);
    const size_t M = a.rows(), K = a.cols(), N = b.cols();
    const float *A = a.data();
    const float *B = b.data();
    float *C = c.data();

    c.zero();

    /* Register-blocked micro-kernel: 4 rows x 16 columns of C are held in 8
     * YMM accumulators for the whole k loop, so C is loaded and stored once
     * per tile instead of once per k. Register pressure: 8 accumulators + 2
     * B vectors + 4 A broadcasts = 14 of the 16 YMM registers.
     *
     * This is the difference that beats compiler auto-vectorisation: the
     * compiler vectorises the arithmetic, but it still round-trips C to
     * memory on every k. Keeping the accumulators in registers turns a
     * memory-bound loop into a compute-bound one. */
    constexpr size_t MR = 4;  /* rows per micro-tile   */
    constexpr size_t NR = 16; /* columns per micro-tile (2 x 8 floats) */

    for (size_t i0 = 0; i0 < M; i0 += kBlockM) {
        const size_t i_max = std::min(i0 + kBlockM, M);
        for (size_t k0 = 0; k0 < K; k0 += kBlockK) {
            const size_t k_max = std::min(k0 + kBlockK, K);
            for (size_t j0 = 0; j0 < N; j0 += kBlockN) {
                const size_t j_max = std::min(j0 + kBlockN, N);

                size_t i = i0;
                for (; i + MR <= i_max; i += MR) {
                    size_t j = j0;
                    for (; j + NR <= j_max; j += NR) {
                        float *c0 = &C[(i + 0) * N + j];
                        float *c1 = &C[(i + 1) * N + j];
                        float *c2 = &C[(i + 2) * N + j];
                        float *c3 = &C[(i + 3) * N + j];

                        __m256 acc00 = _mm256_loadu_ps(c0);
                        __m256 acc01 = _mm256_loadu_ps(c0 + 8);
                        __m256 acc10 = _mm256_loadu_ps(c1);
                        __m256 acc11 = _mm256_loadu_ps(c1 + 8);
                        __m256 acc20 = _mm256_loadu_ps(c2);
                        __m256 acc21 = _mm256_loadu_ps(c2 + 8);
                        __m256 acc30 = _mm256_loadu_ps(c3);
                        __m256 acc31 = _mm256_loadu_ps(c3 + 8);

                        for (size_t k = k0; k < k_max; k++) {
                            const float *brow = &B[k * N + j];
                            const __m256 b0 = _mm256_loadu_ps(brow);
                            const __m256 b1 = _mm256_loadu_ps(brow + 8);

                            __m256 av = _mm256_set1_ps(A[(i + 0) * K + k]);
                            acc00 = _mm256_fmadd_ps(av, b0, acc00);
                            acc01 = _mm256_fmadd_ps(av, b1, acc01);

                            av = _mm256_set1_ps(A[(i + 1) * K + k]);
                            acc10 = _mm256_fmadd_ps(av, b0, acc10);
                            acc11 = _mm256_fmadd_ps(av, b1, acc11);

                            av = _mm256_set1_ps(A[(i + 2) * K + k]);
                            acc20 = _mm256_fmadd_ps(av, b0, acc20);
                            acc21 = _mm256_fmadd_ps(av, b1, acc21);

                            av = _mm256_set1_ps(A[(i + 3) * K + k]);
                            acc30 = _mm256_fmadd_ps(av, b0, acc30);
                            acc31 = _mm256_fmadd_ps(av, b1, acc31);
                        }

                        _mm256_storeu_ps(c0, acc00);
                        _mm256_storeu_ps(c0 + 8, acc01);
                        _mm256_storeu_ps(c1, acc10);
                        _mm256_storeu_ps(c1 + 8, acc11);
                        _mm256_storeu_ps(c2, acc20);
                        _mm256_storeu_ps(c2 + 8, acc21);
                        _mm256_storeu_ps(c3, acc30);
                        _mm256_storeu_ps(c3 + 8, acc31);
                    }
                    /* Columns left over after the 16-wide tiles. */
                    if (j < j_max)
                        simd_edge_tile(A, B, C, K, N, i, i + MR, k0, k_max, j, j_max);
                }
                /* Rows left over after the 4-row tiles. */
                if (i < i_max)
                    simd_edge_tile(A, B, C, K, N, i, i_max, k0, k_max, j0, j_max);
            }
        }
    }
#endif
}
