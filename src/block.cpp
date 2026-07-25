/* Pre-normalised Llama-style decoder blocks.
 *
 * x + attention(rmsnorm(x)), followed by
 * x + down(swiglu(rmsnorm(x))). */

#include <cstring>
#include <stdexcept>

#include "matmul.hpp"
#include "model.hpp"
#include "ops.hpp"

namespace {

void normalise_rows(const Matrix &input, const std::vector<float> &weight,
                    Matrix &output, bool reference) {
    if (weight.size() != input.cols() ||
        output.rows() != input.rows() || output.cols() != input.cols())
        throw std::invalid_argument("decoder: normalization shape mismatch");
    for (size_t row = 0; row < input.rows(); row++) {
        const float *source = input.data() + row * input.cols();
        float *destination = output.data() + row * output.cols();
        if (reference) {
            rmsnorm_reference(source, weight.data(), destination,
                              input.cols(), 1e-5f);
        } else {
            rmsnorm(source, weight.data(), destination,
                    input.cols(), 1e-5f);
        }
    }
}

void copy_matrix(const Matrix &source, Matrix &destination) {
    if (source.rows() != destination.rows() ||
        source.cols() != destination.cols())
        throw std::invalid_argument("decoder: copy shape mismatch");
    std::memcpy(destination.data(), source.data(),
                source.size() * sizeof(float));
}

} // namespace

void decoder_block(const Matrix &input, const LayerWeights &weights,
                   const ModelConfig &config, Matrix &output,
                   bool reference) {
    config.validate();
    if (input.cols() != config.dim || output.rows() != input.rows() ||
        output.cols() != input.cols())
        throw std::invalid_argument("decoder: block shape mismatch");

    Matrix normalized(input.rows(), config.dim);
    Matrix attention(input.rows(), config.dim);
    Matrix residual(input.rows(), config.dim);
    normalise_rows(input, weights.attention_norm, normalized, reference);
    attention_forward(normalized, weights, config, attention, reference);
    copy_matrix(input, residual);
    add_inplace(residual.data(), attention.data(), residual.size());

    normalise_rows(residual, weights.ffn_norm, normalized, reference);
    Matrix activated(input.rows(), config.ffn_dim);
    if (reference)
        swiglu_reference(normalized, weights.gate, weights.up, activated);
    else
        swiglu(normalized, weights.gate, weights.up, activated);
    Matrix feed_forward(input.rows(), config.dim);
    if (reference)
        matmul_naive(activated, weights.down, feed_forward);
    else
        matmul_simd(activated, weights.down, feed_forward);
    copy_matrix(residual, output);
    add_inplace(output.data(), feed_forward.data(), output.size());
}

void decoder_stack(const Matrix &input,
                   const std::vector<LayerWeights> &layers,
                   const ModelConfig &config, Matrix &output,
                   bool reference) {
    config.validate();
    if (layers.size() != config.layers ||
        output.rows() != input.rows() || output.cols() != input.cols())
        throw std::invalid_argument("decoder: stack shape mismatch");

    Matrix current(input.rows(), input.cols());
    Matrix next(input.rows(), input.cols());
    copy_matrix(input, current);
    for (const LayerWeights &layer : layers) {
        decoder_block(current, layer, config, next, reference);
        current = std::move(next);
        next = Matrix(input.rows(), input.cols());
    }
    copy_matrix(current, output);
}
