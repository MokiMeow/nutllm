/* Dependency-free row-parallel kernels.
 *
 * Output rows are independent, so workers own disjoint contiguous ranges and
 * need no locks in the hot loop. Thread count is capped at the row count. */

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "matmul.hpp"

namespace {

class RowExecutor {
public:
    RowExecutor(size_t threads, std::atomic<size_t> &worker_starts)
        : threads_(threads) {
        workers_.reserve(threads - 1);
        for (size_t worker = 1; worker < threads; worker++) {
            workers_.emplace_back([this, worker, &worker_starts]() {
                worker_starts.fetch_add(1, std::memory_order_relaxed);
                worker_loop(worker);
            });
        }
    }

    ~RowExecutor() {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        start_.notify_all();
        for (std::thread &worker : workers_)
            worker.join();
    }

    RowExecutor(const RowExecutor &) = delete;
    RowExecutor &operator=(const RowExecutor &) = delete;

    void run(size_t rows,
             const std::function<void(size_t, size_t)> &function) {
        std::lock_guard<std::mutex> run_lock(run_mutex_);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            rows_ = rows;
            function_ = function;
            completed_ = 0;
            error_ = nullptr;
            generation_++;
        }
        start_.notify_all();
        try {
            function(0, rows / threads_);
        } catch (...) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            error_ = std::current_exception();
        }
        std::unique_lock<std::mutex> lock(state_mutex_);
        finished_.wait(lock, [this]() {
            return completed_ == workers_.size();
        });
        const std::exception_ptr error = error_;
        function_ = {};
        lock.unlock();
        if (error != nullptr)
            std::rethrow_exception(error);
    }

private:
    void worker_loop(size_t worker) {
        size_t observed_generation = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            start_.wait(lock, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_)
                return;
            observed_generation = generation_;
            const size_t rows = rows_;
            const std::function<void(size_t, size_t)> function = function_;
            lock.unlock();
            try {
                function(rows * worker / threads_,
                         rows * (worker + 1) / threads_);
            } catch (...) {
                lock.lock();
                if (error_ == nullptr)
                    error_ = std::current_exception();
                lock.unlock();
            }
            lock.lock();
            completed_++;
            if (completed_ == workers_.size())
                finished_.notify_one();
        }
    }

    size_t threads_;
    std::vector<std::thread> workers_;
    std::mutex run_mutex_;
    std::mutex state_mutex_;
    std::condition_variable start_;
    std::condition_variable finished_;
    std::function<void(size_t, size_t)> function_;
    std::exception_ptr error_;
    size_t rows_ = 0;
    size_t generation_ = 0;
    size_t completed_ = 0;
    bool stopping_ = false;
};

struct ExecutorRegistry {
    std::mutex mutex;
    std::map<size_t, std::unique_ptr<RowExecutor>> executors;
    std::atomic<size_t> worker_starts{0};
};

ExecutorRegistry &registry() {
    static ExecutorRegistry instance;
    return instance;
}

} // namespace

void run_threaded_rows(
    size_t rows, size_t thread_count,
    const std::function<void(size_t, size_t)> &function) {
    if (rows == 0)
        throw std::invalid_argument("threaded kernel: zero rows");
    if (thread_count == 0)
        throw std::invalid_argument("threaded kernel: zero threads");
    const size_t threads = std::min(thread_count, rows);
    if (threads == 1) {
        function(0, rows);
        return;
    }
    ExecutorRegistry &state = registry();
    RowExecutor *executor = nullptr;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        std::unique_ptr<RowExecutor> &entry = state.executors[threads];
        if (!entry)
            entry =
                std::make_unique<RowExecutor>(threads, state.worker_starts);
        executor = entry.get();
    }
    executor->run(rows, function);
}

size_t threaded_worker_start_count() {
    return registry().worker_starts.load(std::memory_order_relaxed);
}

void matmul_threaded(const Matrix &a, const Matrix &b, Matrix &c,
                     size_t thread_count) {
    if (a.cols() != b.rows() || c.rows() != a.rows() ||
        c.cols() != b.cols())
        throw std::invalid_argument("threaded matmul: shape mismatch");
    c.zero();
    run_threaded_rows(a.rows(), thread_count,
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
    run_threaded_rows(matrix.rows(), thread_count,
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

void linear_threaded(const float *input, const Matrix &weights, float *output,
                     size_t thread_count) {
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("threaded linear: null pointer");
    run_threaded_rows(weights.cols(), thread_count,
                      [input, &weights, output](size_t begin, size_t end) {
        std::fill(output + begin, output + end, 0.0f);
        for (size_t row = 0; row < weights.rows(); row++) {
            const float value = input[row];
            const float *weight_row =
                weights.data() + row * weights.cols();
            for (size_t column = begin; column < end; column++)
                output[column] += value * weight_row[column];
        }
    });
}
