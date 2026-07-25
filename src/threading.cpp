/* Dependency-free row-parallel kernels.
 *
 * Output rows are independent, so workers own disjoint contiguous ranges and
 * need no locks in the hot loop. Thread count is capped at the row count. */

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

#include "matmul.hpp"

namespace {

size_t validate_threads(size_t requested, size_t rows) {
    if (requested == 0)
        throw std::invalid_argument("threaded kernel: zero threads");
    return std::min(requested, rows);
}

template <typename Function>
void parallel_rows(size_t rows, size_t requested, Function function) {
    const size_t threads = validate_threads(requested, rows);
    std::vector<std::thread> workers;
    workers.reserve(threads > 0 ? threads - 1 : 0);
    for (size_t thread = 1; thread < threads; thread++) {
        const size_t begin = rows * thread / threads;
        const size_t end = rows * (thread + 1) / threads;
        workers.emplace_back(function, begin, end);
    }
    function(0, rows / threads);
    for (std::thread &worker : workers)
        worker.join();
}

} // namespace

void matmul_threaded(const Matrix &a, const Matrix &b, Matrix &c,
                     size_t thread_count) {
    if (a.cols() != b.rows() || c.rows() != a.rows() ||
        c.cols() != b.cols())
        throw std::invalid_argument("threaded matmul: shape mismatch");
    c.zero();
    parallel_rows(a.rows(), thread_count,
                  [&a, &b, &c](size_t begin, size_t end) {
        for (size_t row = begin; row < end; row++) {
            float *output = c.data() + row * c.cols();
            for (size_t inner = 0; inner < a.cols(); inner++) {
                const float value = a.at(row, inner);
                const float *weights = b.data() + inner * b.cols();
                for (size_t column = 0; column < b.cols(); column++)
                    output[column] += value * weights[column];
            }
        }
    });
}

void matvec_threaded(const Matrix &matrix, const float *vector, float *output,
                     size_t thread_count) {
    if (vector == nullptr || output == nullptr)
        throw std::invalid_argument("threaded matvec: null pointer");
    parallel_rows(matrix.rows(), thread_count,
                  [&matrix, vector, output](size_t begin, size_t end) {
        for (size_t row = begin; row < end; row++) {
            const float *weights =
                matrix.data() + row * matrix.cols();
            float sum = 0.0f;
            for (size_t column = 0; column < matrix.cols(); column++)
                sum += weights[column] * vector[column];
            output[row] = sum;
        }
    });
}
