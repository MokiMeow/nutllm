/* Multi-head causal self-attention.
 *
 * Storage is row-major [sequence][head][head_dimension]. Batch is fixed at one
 * for this educational inference engine. Heads are contiguous slices of each
 * sequence row; they are a layout, not separate matrices. */

#include <cmath>
#include <limits>
#include <stdexcept>

#include "matmul.hpp"
#include "model.hpp"
#include "ops.hpp"

void ModelConfig::validate() const {
    if (layers == 0 || heads == 0 || dim == 0 || head_dim == 0 ||
        ffn_dim == 0 || vocab_size == 0 || max_seq == 0 ||
        heads > std::numeric_limits<size_t>::max() / head_dim ||
        heads * head_dim != dim || head_dim % 2 != 0 ||
        eos_token >= vocab_size)
        throw std::invalid_argument("model config: invalid dimensions");
}

LayerWeights::LayerWeights(const ModelConfig &config)
    : query(config.dim, config.dim),
      key(config.dim, config.dim),
      value(config.dim, config.dim),
      output(config.dim, config.dim),
      gate(config.dim, config.ffn_dim),
      up(config.dim, config.ffn_dim),
      down(config.ffn_dim, config.dim),
      attention_norm(config.dim, 1.0f),
      ffn_norm(config.dim, 1.0f) {
    config.validate();
}

namespace {

void project(const Matrix &input, const Matrix &weights, Matrix &output,
             bool reference) {
    if (reference)
        matmul_naive(input, weights, output);
    else
        matmul_simd(input, weights, output);
}

} // namespace

void attention_forward(const Matrix &input, const LayerWeights &weights,
                       const ModelConfig &config, Matrix &output,
                       bool reference) {
    config.validate();
    if (input.cols() != config.dim || output.rows() != input.rows() ||
        output.cols() != config.dim || input.rows() > config.max_seq)
        throw std::invalid_argument("attention: shape exceeds model config");

    const size_t sequence = input.rows();
    Matrix query(sequence, config.dim);
    Matrix key(sequence, config.dim);
    Matrix value(sequence, config.dim);
    project(input, weights.query, query, reference);
    project(input, weights.key, key, reference);
    project(input, weights.value, value, reference);

    for (size_t position = 0; position < sequence; position++) {
        for (size_t head = 0; head < config.heads; head++) {
            const size_t offset =
                position * config.dim + head * config.head_dim;
            rope(query.data() + offset, key.data() + offset, 1,
                 config.head_dim, position);
        }
    }

    Matrix context(sequence, config.dim);
    context.zero();
    const float inverse_scale = 1.0f / std::sqrt(float(config.head_dim));
    for (size_t head = 0; head < config.heads; head++) {
        Matrix scores(sequence, sequence);
        for (size_t row = 0; row < sequence; row++) {
            for (size_t column = 0; column < sequence; column++) {
                if (column > row) {
                    scores.at(row, column) =
                        -std::numeric_limits<float>::infinity();
                    continue;
                }
                double dot = 0.0;
                for (size_t dimension = 0;
                     dimension < config.head_dim; dimension++) {
                    const size_t query_offset =
                        row * config.dim + head * config.head_dim + dimension;
                    const size_t key_offset =
                        column * config.dim +
                        head * config.head_dim + dimension;
                    dot += double(query.data()[query_offset]) *
                           double(key.data()[key_offset]);
                }
                scores.at(row, column) = float(dot) * inverse_scale;
            }
        }
        softmax_inplace(scores);

        for (size_t row = 0; row < sequence; row++) {
            for (size_t dimension = 0;
                 dimension < config.head_dim; dimension++) {
                double sum = 0.0;
                for (size_t column = 0; column <= row; column++) {
                    const size_t value_offset =
                        column * config.dim +
                        head * config.head_dim + dimension;
                    sum += double(scores.at(row, column)) *
                           double(value.data()[value_offset]);
                }
                const size_t context_offset =
                    row * config.dim + head * config.head_dim + dimension;
                context.data()[context_offset] = float(sum);
            }
        }
    }
    project(context, weights.output, output, reference);
}
