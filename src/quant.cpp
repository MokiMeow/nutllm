/* Per-block symmetric INT8 and INT4 quantisation.
 *
 * INT4 packing is deliberately explicit: flattened even-indexed weights use
 * the low nibble, odd-indexed weights use the high nibble. Each nibble is a
 * signed four-bit two's-complement value (-7..7; -8 is never emitted). */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "matmul.hpp"
#include "quant.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define NUTLLM_QUANT_AVX2 1
#endif

namespace {

uint16_t positive_float_to_half(float value) {
    if (!(value > 0.0f) || !std::isfinite(value))
        throw std::invalid_argument("quantize: scale must be finite positive");
    if (value > 65504.0f)
        throw std::range_error("quantize: scale exceeds fp16");
    const float minimum_normal = std::ldexp(1.0f, -14);
    if (value < minimum_normal) {
        long mantissa = std::lround(std::ldexp(double(value), 24));
        mantissa = std::max(1L, std::min(1023L, mantissa));
        return uint16_t(mantissa);
    }

    int exponent = 0;
    const double fraction = std::frexp(double(value), &exponent);
    exponent--;
    long mantissa =
        std::lround((fraction * 2.0 - 1.0) * 1024.0);
    if (mantissa == 1024) {
        mantissa = 0;
        exponent++;
    }
    if (exponent > 15)
        throw std::range_error("quantize: rounded scale exceeds fp16");
    return uint16_t((exponent + 15) << 10) | uint16_t(mantissa);
}

float positive_half_to_float(uint16_t value) {
    const unsigned exponent = (value >> 10) & 0x1f;
    const unsigned mantissa = value & 0x3ff;
    if (exponent == 0)
        return std::ldexp(float(mantissa), -24);
    return std::ldexp(1.0f + float(mantissa) / 1024.0f,
                      int(exponent) - 15);
}

size_t checked_elements(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 ||
        rows > std::numeric_limits<size_t>::max() / cols)
        throw std::invalid_argument("quantize: invalid dimensions");
    return rows * cols;
}

#ifdef NUTLLM_QUANT_AVX2
float horizontal_sum(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    float total = 0.0f;
    for (float lane : lanes)
        total += lane;
    return total;
}

float dot_int8_blocks(const uint8_t *weights, const uint16_t *scales,
                      const float *vector, size_t size) {
    __m256 total = _mm256_setzero_ps();
    for (size_t block = 0; block < size / 32; block++) {
        const __m256 scale =
            _mm256_set1_ps(positive_half_to_float(scales[block]));
        for (size_t offset = 0; offset < 32; offset += 8) {
            const __m128i packed = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(
                    weights + block * 32 + offset));
            const __m256 values = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed)), scale);
            total = _mm256_fmadd_ps(
                values, _mm256_loadu_ps(vector + block * 32 + offset),
                total);
        }
    }
    return horizontal_sum(total);
}

float dot_int4_blocks(const uint8_t *weights, const uint16_t *scales,
                      const float *vector, size_t size) {
    __m256 total = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i sign = _mm_set1_epi8(0x08);
    for (size_t block = 0; block < size / 32; block++) {
        const __m128i packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(weights + block * 16));
        __m128i low = _mm_and_si128(packed, mask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        low = _mm_sub_epi8(_mm_xor_si128(low, sign), sign);
        high = _mm_sub_epi8(_mm_xor_si128(high, sign), sign);
        const __m128i pairs[2] = {
            _mm_unpacklo_epi8(low, high),
            _mm_unpackhi_epi8(low, high)};
        const __m256 scale =
            _mm256_set1_ps(positive_half_to_float(scales[block]));
        for (size_t half = 0; half < 2; half++) {
            for (size_t offset = 0; offset < 16; offset += 8) {
                const __m128i bytes =
                    offset == 0 ? pairs[half] :
                                  _mm_srli_si128(pairs[half], 8);
                const __m256 values = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),
                    scale);
                const size_t vector_offset =
                    block * 32 + half * 16 + offset;
                total = _mm256_fmadd_ps(
                    values, _mm256_loadu_ps(vector + vector_offset),
                    total);
            }
        }
    }
    return horizontal_sum(total);
}
#endif

} // namespace

QuantizedMatrix::QuantizedMatrix(size_t rows, size_t cols, size_t block_size,
                                 QuantType type)
    : rows_(rows), cols_(cols), block_size_(block_size), type_(type) {
    const size_t elements = checked_elements(rows, cols);
    if (block_size == 0)
        throw std::invalid_argument("quantize: zero block size");
    if (elements > std::numeric_limits<size_t>::max() - (block_size - 1))
        throw std::length_error("quantize: block count overflow");
    const size_t block_count = (elements + block_size - 1) / block_size;
    scales_.resize(block_count);
    data_.resize(type == QuantType::int8 ? elements : (elements + 1) / 2);
}

QuantizedMatrix QuantizedMatrix::quantize(const Matrix &source,
                                          QuantType type,
                                          size_t block_size) {
    return quantize(source.data(), source.rows(), source.cols(), type,
                    block_size);
}

QuantizedMatrix QuantizedMatrix::quantize(const float *source, size_t rows,
                                          size_t cols, QuantType type,
                                          size_t block_size) {
    if (source == nullptr)
        throw std::invalid_argument("quantize: null source");
    QuantizedMatrix result(rows, cols, block_size, type);
    const int limit = type == QuantType::int8 ? 127 : 7;
    for (size_t block = 0; block < result.blocks(); block++) {
        const size_t begin = block * block_size;
        const size_t end = std::min(begin + block_size, result.size());
        float maximum = 0.0f;
        for (size_t index = begin; index < end; index++) {
            if (!std::isfinite(source[index]))
                throw std::invalid_argument(
                    "quantize: non-finite source weight");
            maximum = std::max(maximum, std::fabs(source[index]));
        }
        const float exact_scale =
            maximum == 0.0f ? std::ldexp(1.0f, -24) :
                              maximum / float(limit);
        result.scales_[block] = positive_float_to_half(exact_scale);
        const float stored_scale =
            positive_half_to_float(result.scales_[block]);
        for (size_t index = begin; index < end; index++) {
            int quantized =
                int(std::lround(source[index] / stored_scale));
            quantized = std::max(-limit, std::min(limit, quantized));
            if (type == QuantType::int8) {
                result.data_[index] =
                    uint8_t(unsigned(quantized) & 0xff);
            } else {
                const uint8_t nibble = uint8_t(quantized) & 0x0f;
                if (index % 2 == 0)
                    result.data_[index / 2] =
                        uint8_t((result.data_[index / 2] & 0xf0) | nibble);
                else
                    result.data_[index / 2] =
                        uint8_t((result.data_[index / 2] & 0x0f) |
                                (nibble << 4));
            }
        }
    }
    return result;
}

int QuantizedMatrix::quantized_value(size_t index) const {
    if (index >= size())
        throw std::out_of_range("quantized matrix: index out of range");
    if (type_ == QuantType::int8)
        return data_[index] >= 128 ? int(data_[index]) - 256 :
                                     int(data_[index]);
    const uint8_t packed = data_[index / 2];
    const unsigned nibble =
        index % 2 == 0 ? packed & 0x0f : (packed >> 4) & 0x0f;
    return nibble >= 8 ? int(nibble) - 16 : int(nibble);
}

float QuantizedMatrix::scale(size_t block) const {
    if (block >= scales_.size())
        throw std::out_of_range("quantized matrix: block out of range");
    return positive_half_to_float(scales_[block]);
}

float QuantizedMatrix::value(size_t row, size_t column) const {
    if (row >= rows_ || column >= cols_)
        throw std::out_of_range("quantized matrix: index out of range");
    const size_t index = row * cols_ + column;
    return float(quantized_value(index)) * scale(index / block_size_);
}

float QuantizedMatrix::error_bound(size_t row, size_t column) const {
    if (row >= rows_ || column >= cols_)
        throw std::out_of_range("quantized matrix: index out of range");
    const size_t index = row * cols_ + column;
    const float stored_scale = scale(index / block_size_);
    const float half_rounding =
        std::max(stored_scale / 2047.0f, std::ldexp(1.0f, -25));
    const float limit = type_ == QuantType::int8 ? 127.0f : 7.0f;
    return stored_scale * 0.5f + (limit + 0.5f) * half_rounding;
}

size_t QuantizedMatrix::storage_bytes() const {
    if (scales_.size() >
        (std::numeric_limits<size_t>::max() - data_.size()) /
            sizeof(uint16_t))
        throw std::overflow_error("quantized matrix: storage size overflow");
    return data_.size() + scales_.size() * sizeof(uint16_t);
}

uint8_t QuantizedMatrix::packed_byte(size_t index) const {
    if (type_ != QuantType::int4 || index >= data_.size())
        throw std::out_of_range("quantized matrix: packed byte out of range");
    return data_[index];
}

Matrix dequantize(const QuantizedMatrix &source) {
    Matrix result(source.rows(), source.cols());
    for (size_t row = 0; row < source.rows(); row++)
        for (size_t column = 0; column < source.cols(); column++)
            result.at(row, column) = source.value(row, column);
    return result;
}

void matvec_quantized(const QuantizedMatrix &matrix, const float *vector,
                      float *output) {
    matvec_quantized_threaded(matrix, vector, output, 1);
}

void matvec_quantized_threaded(const QuantizedMatrix &matrix,
                               const float *vector, float *output,
                               size_t thread_count) {
    if (vector == nullptr || output == nullptr)
        throw std::invalid_argument("quantized matvec: null pointer");
    if (thread_count == 0)
        throw std::invalid_argument("quantized matvec: zero threads");
    auto rows = [&matrix, vector, output](size_t begin, size_t end) {
        for (size_t row = begin; row < end; row++) {
#ifdef NUTLLM_QUANT_AVX2
            if (matrix.block_size_ == 32 && matrix.cols_ % 32 == 0) {
                const size_t scale_offset =
                    row * matrix.cols_ / matrix.block_size_;
                if (matrix.type_ == QuantType::int8) {
                    output[row] = dot_int8_blocks(
                        matrix.data_.data() + row * matrix.cols_,
                        matrix.scales_.data() + scale_offset, vector,
                        matrix.cols_);
                } else {
                    output[row] = dot_int4_blocks(
                        matrix.data_.data() + row * matrix.cols_ / 2,
                        matrix.scales_.data() + scale_offset, vector,
                        matrix.cols_);
                }
                continue;
            }
#endif
            double sum = 0.0;
            const size_t row_begin = row * matrix.cols_;
            const size_t row_end = row_begin + matrix.cols_;
            size_t index = row_begin;
            while (index < row_end) {
                const size_t block = index / matrix.block_size_;
                const size_t block_end = std::min(
                    row_end, (block + 1) * matrix.block_size_);
                const float scale =
                    positive_half_to_float(matrix.scales_[block]);
                for (; index < block_end; index++) {
                    int quantized = 0;
                    if (matrix.type_ == QuantType::int8) {
                        const uint8_t value = matrix.data_[index];
                        quantized =
                            value >= 128 ? int(value) - 256 : int(value);
                    } else {
                        const uint8_t packed = matrix.data_[index / 2];
                        const unsigned nibble =
                            index % 2 == 0 ? packed & 0x0f :
                                             (packed >> 4) & 0x0f;
                        quantized = nibble >= 8 ? int(nibble) - 16 :
                                                 int(nibble);
                    }
                    sum += double(float(quantized) * scale) *
                           double(vector[index - row_begin]);
                }
            }
            output[row] = float(sum);
        }
    };
    run_threaded_rows(matrix.rows(), thread_count, rows);
}

void matmul_quantized(const Matrix &left, const QuantizedMatrix &right,
                      Matrix &output) {
    if (left.cols() != right.rows() || output.rows() != left.rows() ||
        output.cols() != right.cols())
        throw std::invalid_argument("quantized matmul: shape mismatch");
    output.zero();
    for (size_t row = 0; row < left.rows(); row++) {
        for (size_t inner = 0; inner < left.cols(); inner++) {
            const float left_value = left.at(row, inner);
            for (size_t column = 0; column < right.cols(); column++)
                output.at(row, column) +=
                    left_value * right.value(inner, column);
        }
    }
}
