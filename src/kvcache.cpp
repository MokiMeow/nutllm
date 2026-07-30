/* Preallocated K/V cache and incremental Llama-style decoder.
 *
 * Keys and values are laid out [layer][head][position][head_dim], making one
 * head's complete attention history contiguous. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "kvcache.hpp"
#include "matmul.hpp"
#include "ops.hpp"

namespace {

size_t checked_elements(const ModelConfig &config) {
    config.validate();
    if (config.layers >
        std::numeric_limits<size_t>::max() / config.max_seq ||
        config.layers * config.max_seq >
            std::numeric_limits<size_t>::max() / config.kv_dim())
        throw std::length_error("kv cache: dimensions overflow");
    return config.layers * config.max_seq * config.kv_dim();
}

void normalize_rows(const Matrix &input, const std::vector<float> &weights,
                    Matrix &output) {
    for (size_t row = 0; row < input.rows(); row++)
        rmsnorm(input.data() + row * input.cols(), weights.data(),
                output.data() + row * output.cols(), input.cols(), 1e-5f);
}

void copy_embedding(const ModelWeights &model, int token, float *output) {
    if (token < 0 || size_t(token) >= model.config.vocab_size)
        throw std::out_of_range("kv cache: token id out of range");
    std::memcpy(output,
                model.embeddings.data() + size_t(token) * model.config.dim,
                model.config.dim * sizeof(float));
}

void project_row(const float *input, const Matrix &weights, float *output,
                 size_t thread_count) {
    linear_threaded(input, weights, output, thread_count);
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

void cache_layer_prefill(const Matrix &input, const LayerWeights &weights,
                         const ModelConfig &config, size_t layer,
                         KVCache &cache) {
    Matrix normalized(input.rows(), config.dim);
    Matrix key(input.rows(), config.kv_dim());
    Matrix value(input.rows(), config.kv_dim());
    normalize_rows(input, weights.attention_norm, normalized);
    matmul_simd(normalized, weights.key, key);
    matmul_simd(normalized, weights.value, value);
    for (size_t position = 0; position < input.rows(); position++) {
        for (size_t head = 0; head < config.effective_kv_heads(); head++) {
            const size_t offset =
                position * config.kv_dim() + head * config.head_dim;
            rotate_vector(key.data() + offset, config.head_dim, position);
        }
        cache.store(layer, position,
                    key.data() + position * config.kv_dim(),
                    value.data() + position * config.kv_dim());
    }
}

void attention_step(const float *normalized, const LayerWeights &weights,
                    const ModelConfig &config, size_t layer, size_t position,
                    KVCache &cache, float *output, size_t thread_count) {
    std::vector<float> query(config.dim);
    std::vector<float> key(config.kv_dim());
    std::vector<float> value(config.kv_dim());
    std::vector<float> context(config.dim, 0.0f);
    project_row(normalized, weights.query, query.data(), thread_count);
    project_row(normalized, weights.key, key.data(), thread_count);
    project_row(normalized, weights.value, value.data(), thread_count);
    for (size_t head = 0; head < config.heads; head++) {
        const size_t offset = head * config.head_dim;
        rotate_vector(query.data() + offset, config.head_dim, position);
    }
    for (size_t head = 0; head < config.effective_kv_heads(); head++) {
        const size_t offset = head * config.head_dim;
        rotate_vector(key.data() + offset, config.head_dim, position);
    }
    cache.store(layer, position, key.data(), value.data());

    const float scale = 1.0f / std::sqrt(float(config.head_dim));
    std::vector<float> scores(position + 1);
    const size_t query_heads_per_kv =
        config.heads / config.effective_kv_heads();
    for (size_t head = 0; head < config.heads; head++) {
        const size_t kv_head = head / query_heads_per_kv;
        const size_t head_offset = head * config.head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t cached_position = 0; cached_position <= position;
             cached_position++) {
            const float *cached_key =
                cache.key(layer, kv_head, cached_position);
            double dot = 0.0;
            for (size_t dimension = 0; dimension < config.head_dim;
                 dimension++)
                dot += double(query[head_offset + dimension]) *
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
            context[head_offset + dimension] = float(sum);
        }
    }
    project_row(context.data(), weights.output, output, thread_count);
}

void decoder_layer_step(std::vector<float> &hidden,
                        const LayerWeights &weights,
                        const ModelConfig &config, size_t layer,
                        size_t position, KVCache &cache,
                        size_t thread_count) {
    std::vector<float> normalized(config.dim);
    std::vector<float> attention(config.dim);
    rmsnorm(hidden.data(), weights.attention_norm.data(), normalized.data(),
            config.dim, 1e-5f);
    attention_step(normalized.data(), weights, config, layer, position,
                   cache, attention.data(), thread_count);
    add_inplace(hidden.data(), attention.data(), config.dim);

    rmsnorm(hidden.data(), weights.ffn_norm.data(), normalized.data(),
            config.dim, 1e-5f);
    std::vector<float> gate(config.ffn_dim);
    std::vector<float> up(config.ffn_dim);
    std::vector<float> activated(config.ffn_dim);
    project_row(normalized.data(), weights.gate, gate.data(), thread_count);
    project_row(normalized.data(), weights.up, up.data(), thread_count);
    for (size_t index = 0; index < config.ffn_dim; index++)
        activated[index] = silu(gate[index]) * up[index];
    std::vector<float> feed_forward(config.dim);
    project_row(activated.data(), weights.down, feed_forward.data(),
                thread_count);
    add_inplace(hidden.data(), feed_forward.data(), config.dim);
}

} // namespace

KVCache::KVCache(const ModelConfig &config)
    : config_(config), keys_(checked_elements(config)),
      values_(checked_elements(config)) {
    config_.validate();
}

size_t KVCache::memory_bytes() const {
    if (keys_.size() >
        std::numeric_limits<size_t>::max() / (2 * sizeof(float)))
        throw std::overflow_error("kv cache: byte size overflow");
    return keys_.size() * 2 * sizeof(float);
}

bool KVCache::compatible(const ModelConfig &config) const {
    return config_.layers == config.layers && config_.heads == config.heads &&
           config_.dim == config.dim && config_.head_dim == config.head_dim &&
           config_.ffn_dim == config.ffn_dim &&
           config_.vocab_size == config.vocab_size &&
           config_.max_seq == config.max_seq &&
           config_.eos_token == config.eos_token &&
           config_.effective_kv_heads() == config.effective_kv_heads();
}

size_t KVCache::offset(size_t layer, size_t head, size_t position) const {
    if (layer >= config_.layers ||
        head >= config_.effective_kv_heads() ||
        position >= config_.max_seq)
        throw std::out_of_range("kv cache: index out of range");
    return ((layer * config_.effective_kv_heads() + head) *
                config_.max_seq +
            position) *
           config_.head_dim;
}

const float *KVCache::key(size_t layer, size_t head, size_t position) const {
    return keys_.data() + offset(layer, head, position);
}

const float *KVCache::value(size_t layer, size_t head,
                            size_t position) const {
    return values_.data() + offset(layer, head, position);
}

void KVCache::store(size_t layer, size_t position, const float *key_data,
                    const float *value_data) {
    if (key_data == nullptr || value_data == nullptr)
        throw std::invalid_argument("kv cache: null source");
    for (size_t head = 0; head < config_.effective_kv_heads(); head++) {
        const size_t destination = offset(layer, head, position);
        std::memcpy(keys_.data() + destination,
                    key_data + head * config_.head_dim,
                    config_.head_dim * sizeof(float));
        std::memcpy(values_.data() + destination,
                    value_data + head * config_.head_dim,
                    config_.head_dim * sizeof(float));
    }
}

void KVCache::set_length(size_t length) {
    if (length > config_.max_seq)
        throw std::length_error("kv cache: context exceeds capacity");
    length_ = length;
}

Matrix prefill_tokens(const ModelWeights &model,
                      const std::vector<int> &tokens, KVCache &cache,
                      size_t thread_count) {
    model.config.validate();
    if (!cache.compatible(model.config))
        throw std::invalid_argument("kv cache: model configuration mismatch");
    if (model.layers.size() != model.config.layers)
        throw std::invalid_argument("kv cache: layer count mismatch");
    if (tokens.empty() || tokens.size() > model.config.max_seq)
        throw std::invalid_argument("kv cache: invalid prefill length");
    if (thread_count == 0)
        throw std::invalid_argument("kv cache: zero threads");
    cache.reset();
    Matrix current(tokens.size(), model.config.dim);
    for (size_t row = 0; row < tokens.size(); row++)
        copy_embedding(model, tokens[row],
                       current.data() + row * model.config.dim);
    for (size_t layer = 0; layer < model.config.layers; layer++) {
        cache_layer_prefill(current, model.layers[layer], model.config,
                            layer, cache);
        Matrix next(tokens.size(), model.config.dim);
        decoder_block(current, model.layers[layer], model.config, next);
        current = std::move(next);
    }
    cache.set_length(tokens.size());
    return current;
}

std::vector<float> decode_token(const ModelWeights &model, int token,
                                KVCache &cache, size_t thread_count) {
    model.config.validate();
    if (!cache.compatible(model.config))
        throw std::invalid_argument("kv cache: model configuration mismatch");
    if (model.layers.size() != model.config.layers)
        throw std::invalid_argument("kv cache: layer count mismatch");
    if (thread_count == 0)
        throw std::invalid_argument("kv cache: zero threads");
    if (cache.length() >= model.config.max_seq)
        throw std::length_error("kv cache: context is full");
    const size_t position = cache.length();
    std::vector<float> hidden(model.config.dim);
    copy_embedding(model, token, hidden.data());
    for (size_t layer = 0; layer < model.config.layers; layer++)
        decoder_layer_step(hidden, model.layers[layer], model.config, layer,
                           position, cache, thread_count);
    cache.set_length(position + 1);
    return hidden;
}
