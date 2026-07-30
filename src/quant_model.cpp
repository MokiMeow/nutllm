/* Incremental decoder for block-quantized llama2.c projection weights.
 *
 * Embeddings, normalization weights, and the output classifier remain fp32.
 * Every large transformer projection is consumed directly from INT8/INT4
 * storage by the quantized matvec kernel. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "matmul.hpp"
#include "ops.hpp"
#include "quant_model.hpp"

namespace {

void checked_add(size_t &total, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - total)
        throw std::overflow_error("quantized model: storage size overflow");
    total += value;
}

void rotate_vector(float *values, size_t dimensions, size_t position) {
    for (size_t pair = 0; pair < dimensions; pair += 2) {
        const double frequency =
            std::pow(10000.0, -double(pair) / double(dimensions));
        const float cosine = float(std::cos(double(position) * frequency));
        const float sine = float(std::sin(double(position) * frequency));
        const float even = values[pair];
        const float odd = values[pair + 1];
        values[pair] = even * cosine - odd * sine;
        values[pair + 1] = even * sine + odd * cosine;
    }
}

void add_inplace(std::vector<float> &destination,
                 const std::vector<float> &source) {
    for (size_t index = 0; index < destination.size(); index++)
        destination[index] += source[index];
}

void attention_step(const float *normalized,
                    const QuantizedLayerWeights &weights,
                    const ModelConfig &config, size_t layer, size_t position,
                    KVCache &cache, std::vector<float> &output,
                    size_t thread_count) {
    std::vector<float> query(config.dim);
    std::vector<float> key(config.kv_dim());
    std::vector<float> value(config.kv_dim());
    std::vector<float> context(config.dim, 0.0f);
    matvec_quantized_threaded(weights.query, normalized, query.data(),
                              thread_count);
    matvec_quantized_threaded(weights.key, normalized, key.data(),
                              thread_count);
    matvec_quantized_threaded(weights.value, normalized, value.data(),
                              thread_count);
    for (size_t head = 0; head < config.heads; head++)
        rotate_vector(query.data() + head * config.head_dim,
                      config.head_dim, position);
    for (size_t head = 0; head < config.effective_kv_heads(); head++)
        rotate_vector(key.data() + head * config.head_dim,
                      config.head_dim, position);
    cache.store(layer, position, key.data(), value.data());

    const float scale = 1.0f / std::sqrt(float(config.head_dim));
    std::vector<float> scores(position + 1);
    const size_t query_heads_per_kv =
        config.heads / config.effective_kv_heads();
    for (size_t head = 0; head < config.heads; head++) {
        const size_t kv_head = head / query_heads_per_kv;
        const size_t query_offset = head * config.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t cached_position = 0; cached_position <= position;
             cached_position++) {
            const float *cached_key =
                cache.key(layer, kv_head, cached_position);
            double dot = 0.0;
            for (size_t dimension = 0; dimension < config.head_dim;
                 dimension++)
                dot += double(query[query_offset + dimension]) *
                       double(cached_key[dimension]);
            scores[cached_position] = float(dot) * scale;
            maximum = std::max(maximum, scores[cached_position]);
        }
        double total = 0.0;
        for (float &score : scores) {
            score = std::exp(score - maximum);
            total += score;
        }
        for (size_t dimension = 0; dimension < config.head_dim;
             dimension++) {
            double sum = 0.0;
            for (size_t cached_position = 0; cached_position <= position;
                 cached_position++) {
                const float *cached_value =
                    cache.value(layer, kv_head, cached_position);
                sum += double(scores[cached_position] / float(total)) *
                       double(cached_value[dimension]);
            }
            context[query_offset + dimension] = float(sum);
        }
    }
    matvec_quantized_threaded(weights.output, context.data(), output.data(),
                              thread_count);
}

} // namespace

QuantizedLayerWeights::QuantizedLayerWeights(
    QuantizedMatrix query_weight, QuantizedMatrix key_weight,
    QuantizedMatrix value_weight, QuantizedMatrix output_weight,
    QuantizedMatrix gate_weight, QuantizedMatrix up_weight,
    QuantizedMatrix down_weight, std::vector<float> attention_norm_weight,
    std::vector<float> ffn_norm_weight)
    : query(std::move(query_weight)),
      key(std::move(key_weight)),
      value(std::move(value_weight)),
      output(std::move(output_weight)),
      gate(std::move(gate_weight)),
      up(std::move(up_weight)),
      down(std::move(down_weight)),
      attention_norm(std::move(attention_norm_weight)),
      ffn_norm(std::move(ffn_norm_weight)) {}

QuantizedModelWeights::QuantizedModelWeights(
    const ModelConfig &model_config, QuantType type, bool allocate_lm_head)
    : config(model_config),
      embeddings(config.vocab_size, config.dim),
      final_norm(config.dim),
      lm_head(allocate_lm_head ? Matrix(config.vocab_size, config.dim)
                               : Matrix()),
      quant_type(type) {
    config.validate();
    layers.reserve(config.layers);
}

size_t QuantizedModelWeights::storage_bytes() const {
    size_t total = embeddings.size() * sizeof(float);
    checked_add(total, final_norm.size() * sizeof(float));
    checked_add(total, lm_head.size() * sizeof(float));
    for (const QuantizedLayerWeights &layer : layers) {
        checked_add(total, layer.query.storage_bytes());
        checked_add(total, layer.key.storage_bytes());
        checked_add(total, layer.value.storage_bytes());
        checked_add(total, layer.output.storage_bytes());
        checked_add(total, layer.gate.storage_bytes());
        checked_add(total, layer.up.storage_bytes());
        checked_add(total, layer.down.storage_bytes());
        checked_add(total, layer.attention_norm.size() * sizeof(float));
        checked_add(total, layer.ffn_norm.size() * sizeof(float));
    }
    return total;
}

std::vector<float> decode_quantized_token(
    const QuantizedModelWeights &model, int token, KVCache &cache,
    size_t thread_count) {
    model.config.validate();
    if (!cache.compatible(model.config))
        throw std::invalid_argument(
            "quantized decode: cache configuration mismatch");
    if (model.layers.size() != model.config.layers)
        throw std::invalid_argument("quantized decode: layer count mismatch");
    if (thread_count == 0)
        throw std::invalid_argument("quantized decode: zero threads");
    if (token < 0 || size_t(token) >= model.config.vocab_size)
        throw std::out_of_range("quantized decode: token out of range");
    if (cache.length() >= cache.capacity())
        throw std::length_error("quantized decode: context is full");

    const size_t position = cache.length();
    std::vector<float> hidden(model.config.dim);
    std::memcpy(hidden.data(),
                model.embeddings.data() + size_t(token) * model.config.dim,
                model.config.dim * sizeof(float));
    std::vector<float> normalized(model.config.dim);
    for (size_t layer = 0; layer < model.config.layers; layer++) {
        const QuantizedLayerWeights &weights = model.layers[layer];
        rmsnorm(hidden.data(), weights.attention_norm.data(),
                normalized.data(), model.config.dim, 1e-5f);
        std::vector<float> attention(model.config.dim);
        attention_step(normalized.data(), weights, model.config, layer,
                       position, cache, attention, thread_count);
        add_inplace(hidden, attention);

        rmsnorm(hidden.data(), weights.ffn_norm.data(), normalized.data(),
                model.config.dim, 1e-5f);
        std::vector<float> gate(model.config.ffn_dim);
        std::vector<float> up(model.config.ffn_dim);
        matvec_quantized_threaded(weights.gate, normalized.data(),
                                  gate.data(), thread_count);
        matvec_quantized_threaded(weights.up, normalized.data(), up.data(),
                                  thread_count);
        for (size_t index = 0; index < model.config.ffn_dim; index++)
            gate[index] = silu(gate[index]) * up[index];
        std::vector<float> feed_forward(model.config.dim);
        matvec_quantized_threaded(weights.down, gate.data(),
                                  feed_forward.data(), thread_count);
        add_inplace(hidden, feed_forward);
    }
    cache.set_length(position + 1);
    return hidden;
}

std::vector<float> prefill_quantized(const QuantizedModelWeights &model,
                                     const std::vector<int> &tokens,
                                     KVCache &cache,
                                     size_t thread_count) {
    if (tokens.empty() || tokens.size() > cache.capacity())
        throw std::invalid_argument("quantized prefill: invalid prompt");
    cache.reset();
    std::vector<float> hidden;
    for (int token : tokens)
        hidden =
            decode_quantized_token(model, token, cache, thread_count);
    return hidden;
}
