#pragma once

#include <cstddef>
#include <vector>

#include "loader.hpp"

class KVCache {
public:
    explicit KVCache(const ModelConfig &config);

    size_t length() const { return length_; }
    size_t capacity() const { return config_.max_seq; }
    size_t memory_bytes() const;
    bool compatible(const ModelConfig &config) const;
    void reset() { length_ = 0; }

    const float *key(size_t layer, size_t head, size_t position) const;
    const float *value(size_t layer, size_t head, size_t position) const;
    void store(size_t layer, size_t position, const float *key,
               const float *value);
    void set_length(size_t length);

private:
    size_t offset(size_t layer, size_t head, size_t position) const;

    ModelConfig config_;
    std::vector<float> keys_;
    std::vector<float> values_;
    size_t length_ = 0;
};

Matrix prefill_tokens(const ModelWeights &model,
                      const std::vector<int> &tokens, KVCache &cache);
std::vector<float> decode_token(const ModelWeights &model, int token,
                                KVCache &cache);
