#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "perplexity.hpp"

double perplexity(const Matrix &logits,
                  const std::vector<size_t> &targets) {
    if (targets.size() != logits.rows() || targets.empty())
        throw std::invalid_argument("perplexity: target count mismatch");
    double negative_log_likelihood = 0.0;
    for (size_t row = 0; row < logits.rows(); row++) {
        if (targets[row] >= logits.cols())
            throw std::out_of_range("perplexity: target out of range");
        float maximum = logits.at(row, 0);
        for (size_t column = 1; column < logits.cols(); column++)
            maximum = std::max(maximum, logits.at(row, column));
        double denominator = 0.0;
        for (size_t column = 0; column < logits.cols(); column++)
            denominator +=
                std::exp(double(logits.at(row, column) - maximum));
        negative_log_likelihood +=
            std::log(denominator) +
            double(maximum - logits.at(row, targets[row]));
    }
    return std::exp(negative_log_likelihood / double(logits.rows()));
}
