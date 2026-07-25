/* Transformer tensor operations.
 *
 * Reference implementations accumulate in double precision where practical.
 * Optimised implementations preserve the same formulas while using contiguous
 * loops and AVX2 when available. The references stay permanently as oracles. */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "matmul.hpp"
#include "ops.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#define NUTLLM_OPS_AVX2 1
#endif

namespace {

void require_pointer(const void *pointer, size_t size, const char *operation) {
    if (size != 0 && pointer == nullptr)
        throw std::invalid_argument(operation);
}

size_t checked_elements(size_t first, size_t second, const char *operation) {
    if (first == 0 || second == 0 ||
        first > std::numeric_limits<size_t>::max() / second)
        throw std::invalid_argument(operation);
    return first * second;
}

void check_swiglu_shapes(const Matrix &input, const Matrix &gate_weight,
                         const Matrix &up_weight, const Matrix &output) {
    if (input.cols() != gate_weight.rows() ||
        input.cols() != up_weight.rows() ||
        gate_weight.cols() != up_weight.cols() ||
        output.rows() != input.rows() ||
        output.cols() != gate_weight.cols())
        throw std::invalid_argument("swiglu: shape mismatch");
}

#ifdef NUTLLM_OPS_AVX2
float horizontal_sum(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    float total = 0.0f;
    for (float lane : lanes)
        total += lane;
    return total;
}
#endif

} // namespace

void softmax_reference(const float *input, float *output,
                       size_t rows, size_t cols) {
    const size_t elements =
        checked_elements(rows, cols, "softmax: invalid dimensions");
    require_pointer(input, elements, "softmax: null input");
    require_pointer(output, elements, "softmax: null output");

    for (size_t row = 0; row < rows; row++) {
        const float *source = input + row * cols;
        float *destination = output + row * cols;
        const float maximum = *std::max_element(source, source + cols);
        double sum = 0.0;
        for (size_t column = 0; column < cols; column++) {
            const double value = std::exp(double(source[column] - maximum));
            destination[column] = float(value);
            sum += value;
        }
        for (size_t column = 0; column < cols; column++)
            destination[column] = float(double(destination[column]) / sum);
    }
}

void softmax_inplace(Matrix &matrix) {
    const size_t cols = matrix.cols();
    for (size_t row = 0; row < matrix.rows(); row++) {
        float *values = matrix.data() + row * cols;
        const float maximum = *std::max_element(values, values + cols);
        float sum = 0.0f;
        for (size_t column = 0; column < cols; column++) {
            values[column] = std::exp(values[column] - maximum);
            sum += values[column];
        }
        const float inverse = 1.0f / sum;
        for (size_t column = 0; column < cols; column++)
            values[column] *= inverse;
    }
}

void rmsnorm_reference(const float *input, const float *weight, float *output,
                       size_t size, float epsilon) {
    require_pointer(input, size, "rmsnorm: null input");
    require_pointer(weight, size, "rmsnorm: null weight");
    require_pointer(output, size, "rmsnorm: null output");
    if (size == 0 || epsilon < 0.0f || !std::isfinite(epsilon))
        throw std::invalid_argument("rmsnorm: invalid size or epsilon");
    double squares = 0.0;
    for (size_t index = 0; index < size; index++)
        squares += double(input[index]) * double(input[index]);
    const double scale = 1.0 / std::sqrt(squares / double(size) + epsilon);
    for (size_t index = 0; index < size; index++)
        output[index] = float(double(input[index]) * scale * weight[index]);
}

void rmsnorm(const float *input, const float *weight, float *output,
             size_t size, float epsilon) {
    require_pointer(input, size, "rmsnorm: null input");
    require_pointer(weight, size, "rmsnorm: null weight");
    require_pointer(output, size, "rmsnorm: null output");
    if (size == 0 || epsilon < 0.0f || !std::isfinite(epsilon))
        throw std::invalid_argument("rmsnorm: invalid size or epsilon");

    float squares = 0.0f;
    size_t index = 0;
#ifdef NUTLLM_OPS_AVX2
    __m256 sum = _mm256_setzero_ps();
    for (; index + 8 <= size; index += 8) {
        const __m256 values = _mm256_loadu_ps(input + index);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(values, values));
    }
    squares = horizontal_sum(sum);
#endif
    for (; index < size; index++)
        squares += input[index] * input[index];
    const float scale = 1.0f / std::sqrt(squares / float(size) + epsilon);
    for (size_t output_index = 0; output_index < size; output_index++)
        output[output_index] = input[output_index] * scale * weight[output_index];
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

void swiglu_reference(const Matrix &input, const Matrix &gate_weight,
                      const Matrix &up_weight, Matrix &output) {
    check_swiglu_shapes(input, gate_weight, up_weight, output);
    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t column = 0; column < output.cols(); column++) {
            double gate = 0.0;
            double up = 0.0;
            for (size_t inner = 0; inner < input.cols(); inner++) {
                gate += double(input.at(row, inner)) *
                        double(gate_weight.at(inner, column));
                up += double(input.at(row, inner)) *
                      double(up_weight.at(inner, column));
            }
            const double activated = gate / (1.0 + std::exp(-gate));
            output.at(row, column) = float(activated * up);
        }
    }
}

void swiglu(const Matrix &input, const Matrix &gate_weight,
            const Matrix &up_weight, Matrix &output) {
    check_swiglu_shapes(input, gate_weight, up_weight, output);
    Matrix gate(input.rows(), gate_weight.cols());
    Matrix up(input.rows(), up_weight.cols());
    matmul_simd(input, gate_weight, gate);
    matmul_simd(input, up_weight, up);
    for (size_t index = 0; index < output.size(); index++)
        output.data()[index] = silu(gate.data()[index]) * up.data()[index];
}

void rope_reference(float *query, float *key, size_t sequence,
                    size_t dimensions, size_t position_offset, float theta) {
    const size_t elements =
        checked_elements(sequence, dimensions, "rope: invalid dimensions");
    require_pointer(query, elements, "rope: null query");
    require_pointer(key, elements, "rope: null key");
    if (dimensions % 2 != 0 || theta <= 0.0f || !std::isfinite(theta))
        throw std::invalid_argument("rope: invalid shape or theta");

    for (size_t position = 0; position < sequence; position++) {
        for (size_t pair = 0; pair < dimensions; pair += 2) {
            const double frequency =
                std::pow(double(theta), -double(pair) / double(dimensions));
            const double angle = double(position_offset + position) * frequency;
            const float cosine = float(std::cos(angle));
            const float sine = float(std::sin(angle));
            for (float *values : {query, key}) {
                const size_t offset = position * dimensions + pair;
                const float even = values[offset];
                const float odd = values[offset + 1];
                values[offset] = even * cosine - odd * sine;
                values[offset + 1] = even * sine + odd * cosine;
            }
        }
    }
}

void rope(float *query, float *key, size_t sequence, size_t dimensions,
          size_t position_offset, float theta) {
    rope_reference(query, key, sequence, dimensions, position_offset, theta);
}

void add_inplace(float *destination, const float *source, size_t size) {
    require_pointer(destination, size, "add: null destination");
    require_pointer(source, size, "add: null source");
    size_t index = 0;
#ifdef NUTLLM_OPS_AVX2
    for (; index + 8 <= size; index += 8) {
        const __m256 left = _mm256_loadu_ps(destination + index);
        const __m256 right = _mm256_loadu_ps(source + index);
        _mm256_storeu_ps(destination + index, _mm256_add_ps(left, right));
    }
#endif
    for (; index < size; index++)
        destination[index] += source[index];
}

void matvec_reference(const Matrix &matrix, const float *vector, float *output) {
    require_pointer(vector, matrix.cols(), "matvec: null vector");
    require_pointer(output, matrix.rows(), "matvec: null output");
    for (size_t row = 0; row < matrix.rows(); row++) {
        double sum = 0.0;
        for (size_t column = 0; column < matrix.cols(); column++)
            sum += double(matrix.at(row, column)) * double(vector[column]);
        output[row] = float(sum);
    }
}

void matvec(const Matrix &matrix, const float *vector, float *output) {
    require_pointer(vector, matrix.cols(), "matvec: null vector");
    require_pointer(output, matrix.rows(), "matvec: null output");
    for (size_t row = 0; row < matrix.rows(); row++) {
        const float *weights = matrix.data() + row * matrix.cols();
        float sum = 0.0f;
        size_t column = 0;
#ifdef NUTLLM_OPS_AVX2
        __m256 accumulator = _mm256_setzero_ps();
        for (; column + 8 <= matrix.cols(); column += 8) {
            const __m256 left = _mm256_loadu_ps(weights + column);
            const __m256 right = _mm256_loadu_ps(vector + column);
#ifdef __FMA__
            accumulator = _mm256_fmadd_ps(left, right, accumulator);
#else
            accumulator =
                _mm256_add_ps(accumulator, _mm256_mul_ps(left, right));
#endif
        }
        sum = horizontal_sum(accumulator);
#endif
        for (; column < matrix.cols(); column++)
            sum += weights[column] * vector[column];
        output[row] = sum;
    }
}
