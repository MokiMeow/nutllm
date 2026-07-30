#pragma once

#include <cstddef>
#include <vector>

#include "tensor.hpp"

struct ModelConfig {
    size_t layers = 0;
    size_t heads = 0;
    size_t dim = 0;
    size_t head_dim = 0;
    size_t ffn_dim = 0;
    size_t vocab_size = 0;
    size_t max_seq = 0;
    size_t eos_token = 0;
    size_t kv_heads = 0;

    void validate() const;
    size_t effective_kv_heads() const {
        return kv_heads == 0 ? heads : kv_heads;
    }
    size_t kv_dim() const { return effective_kv_heads() * head_dim; }
};

struct LayerWeights {
    explicit LayerWeights(const ModelConfig &config);

    Matrix query;
    Matrix key;
    Matrix value;
    Matrix output;
    Matrix gate;
    Matrix up;
    Matrix down;
    std::vector<float> attention_norm;
    std::vector<float> ffn_norm;
};

void attention_forward(const Matrix &input, const LayerWeights &weights,
                       const ModelConfig &config, Matrix &output,
                       bool reference = false);

void decoder_block(const Matrix &input, const LayerWeights &weights,
                   const ModelConfig &config, Matrix &output,
                   bool reference = false);

void decoder_stack(const Matrix &input,
                   const std::vector<LayerWeights> &layers,
                   const ModelConfig &config, Matrix &output,
                   bool reference = false);
