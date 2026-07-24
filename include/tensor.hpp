#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

/* A dense row-major 2-D float matrix with aligned storage.
 *
 * Alignment matters: AVX2 loads/stores want 32-byte alignment, and aligned
 * access avoids a split-cache-line penalty on every vector operation. */
class Matrix {
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        if (rows == 0 || cols == 0)
            throw std::invalid_argument("Matrix: zero dimension");
        /* Round the byte count up to the alignment, as aligned_alloc requires. */
        const size_t bytes = rows * cols * sizeof(float);
        const size_t padded = ((bytes + kAlign - 1) / kAlign) * kAlign;
        data_ = static_cast<float *>(std::aligned_alloc(kAlign, padded));
        if (data_ == nullptr)
            throw std::bad_alloc();
    }

    ~Matrix() { std::free(data_); }

    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;

    Matrix(Matrix &&other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;
        other.rows_ = other.cols_ = 0;
    }

    float *data() { return data_; }
    const float *data() const { return data_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    float &at(size_t r, size_t c) { return data_[r * cols_ + c]; }
    float at(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    void fill(float value) {
        for (size_t i = 0; i < size(); i++)
            data_[i] = value;
    }

    void zero() { fill(0.0f); }

    static constexpr size_t kAlign = 64; /* cache line, also covers AVX2's 32 */

private:
    float *data_ = nullptr;
    size_t rows_ = 0;
    size_t cols_ = 0;
};
