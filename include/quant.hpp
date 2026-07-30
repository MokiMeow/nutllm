#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensor.hpp"

enum class QuantType {
    int8,
    int4,
};

class QuantizedMatrix {
public:
    static QuantizedMatrix quantize(const Matrix &source, QuantType type,
                                    size_t block_size = 32);
    static QuantizedMatrix quantize(const float *source, size_t rows,
                                    size_t cols, QuantType type,
                                    size_t block_size = 32);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    size_t block_size() const { return block_size_; }
    size_t blocks() const { return scales_.size(); }
    QuantType type() const { return type_; }
    size_t storage_bytes() const;
    uint8_t packed_byte(size_t index) const;

    float value(size_t row, size_t column) const;
    float error_bound(size_t row, size_t column) const;

private:
    friend void matvec_quantized_threaded(
        const QuantizedMatrix &matrix, const float *vector, float *output,
        size_t thread_count);

    QuantizedMatrix(size_t rows, size_t cols, size_t block_size,
                    QuantType type);
    int quantized_value(size_t index) const;
    float scale(size_t block) const;

    size_t rows_;
    size_t cols_;
    size_t block_size_;
    QuantType type_;
    std::vector<uint8_t> data_;
    std::vector<uint16_t> scales_;
};

Matrix dequantize(const QuantizedMatrix &source);
void matvec_quantized(const QuantizedMatrix &matrix, const float *vector,
                      float *output);
void matvec_quantized_threaded(const QuantizedMatrix &matrix,
                               const float *vector, float *output,
                               size_t thread_count);
void matmul_quantized(const Matrix &left, const QuantizedMatrix &right,
                      Matrix &output);
