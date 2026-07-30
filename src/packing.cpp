/* Packed-panel matrix multiplication.
 *
 * The existing SIMD kernel reads directly from row-major matrices. This
 * experimental path copies bounded A and B panels into reusable contiguous
 * buffers first, then measures whether the extra copy buys enough locality to
 * pay for itself at the matrix sizes used by the project. */

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "matmul.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define NUTLLM_HAVE_PACKED_AVX2 1
#endif

namespace {

constexpr size_t kPackM = 64;
constexpr size_t kPackN = 128;
constexpr size_t kPackK = 128;

struct Scratch {
    std::vector<float> a;
    std::vector<float> b;
};

thread_local Scratch scratch;

void check_shapes(const Matrix &a, const Matrix &b, const Matrix &c) {
    if (a.cols() != b.rows() || c.rows() != a.rows() ||
        c.cols() != b.cols())
        throw std::invalid_argument("packed matmul: shape mismatch");
}

void accumulate(const float *packed_a, const float *packed_b, float *output,
                size_t rows, size_t inner, size_t columns,
                size_t output_stride) {
    for (size_t row = 0; row < rows; row++) {
        float *output_row = output + row * output_stride;
        const float *a_row = packed_a + row * inner;
        for (size_t k = 0; k < inner; k++) {
            const float value = a_row[k];
            const float *b_row = packed_b + k * columns;
            size_t column = 0;
#ifdef NUTLLM_HAVE_PACKED_AVX2
            const __m256 broadcast = _mm256_set1_ps(value);
            for (; column + 8 <= columns; column += 8) {
                const __m256 weights = _mm256_loadu_ps(b_row + column);
                const __m256 current = _mm256_loadu_ps(output_row + column);
                _mm256_storeu_ps(
                    output_row + column,
                    _mm256_fmadd_ps(broadcast, weights, current));
            }
#endif
            for (; column < columns; column++)
                output_row[column] += value * b_row[column];
        }
    }
}

} // namespace

void matmul_packed(const Matrix &a, const Matrix &b, Matrix &c) {
    check_shapes(a, b, c);
    const size_t rows = a.rows();
    const size_t inner = a.cols();
    const size_t columns = b.cols();
    c.zero();

    for (size_t row0 = 0; row0 < rows; row0 += kPackM) {
        const size_t packed_rows = std::min(kPackM, rows - row0);
        for (size_t inner0 = 0; inner0 < inner; inner0 += kPackK) {
            const size_t packed_inner = std::min(kPackK, inner - inner0);
            scratch.a.resize(packed_rows * packed_inner);
            for (size_t row = 0; row < packed_rows; row++) {
                const float *source =
                    a.data() + (row0 + row) * inner + inner0;
                std::copy(source, source + packed_inner,
                          scratch.a.data() + row * packed_inner);
            }

            for (size_t column0 = 0; column0 < columns;
                 column0 += kPackN) {
                const size_t packed_columns =
                    std::min(kPackN, columns - column0);
                scratch.b.resize(packed_inner * packed_columns);
                for (size_t k = 0; k < packed_inner; k++) {
                    const float *source =
                        b.data() + (inner0 + k) * columns + column0;
                    std::copy(source, source + packed_columns,
                              scratch.b.data() + k * packed_columns);
                }

                accumulate(scratch.a.data(), scratch.b.data(),
                           c.data() + row0 * columns + column0,
                           packed_rows, packed_inner, packed_columns, columns);
            }
        }
    }
}
