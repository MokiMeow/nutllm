#pragma once

#include <cstddef>
#include <vector>

#include "kvcache.hpp"
#include "quant.hpp"

struct QuantizedLayerWeights {
    QuantizedLayerWeights(QuantizedMatrix query_weight,
                          QuantizedMatrix key_weight,
                          QuantizedMatrix value_weight,
                          QuantizedMatrix output_weight,
                          QuantizedMatrix gate_weight,
                          QuantizedMatrix up_weight,
                          QuantizedMatrix down_weight,
                          std::vector<float> attention_norm_weight,
                          std::vector<float> ffn_norm_weight);

    QuantizedMatrix query;
    QuantizedMatrix key;
    QuantizedMatrix value;
    QuantizedMatrix output;
    QuantizedMatrix gate;
    QuantizedMatrix up;
    QuantizedMatrix down;
    std::vector<float> attention_norm;
    std::vector<float> ffn_norm;
};

struct QuantizedModelWeights {
    ModelConfig config;
    Matrix embeddings;
    std::vector<QuantizedLayerWeights> layers;
    std::vector<float> final_norm;
    Matrix lm_head;
    bool tied_embeddings = false;
    QuantType quant_type;

    QuantizedModelWeights(const ModelConfig &model_config,
                          QuantType type, bool allocate_lm_head);
    const Matrix &output_weights() const {
        return tied_embeddings ? embeddings : lm_head;
    }
    size_t storage_bytes() const;
};

std::vector<float> prefill_quantized(const QuantizedModelWeights &model,
                                     const std::vector<int> &tokens,
                                     KVCache &cache, size_t thread_count);
std::vector<float> decode_quantized_token(
    const QuantizedModelWeights &model, int token, KVCache &cache,
    size_t thread_count);
