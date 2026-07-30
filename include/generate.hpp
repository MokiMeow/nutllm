#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "loader.hpp"

struct QuantizedModelWeights;

struct SamplingConfig {
    size_t max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t seed = 1;
    size_t threads = 1;
};

struct GenerationStats {
    size_t prefill_tokens = 0;
    size_t generated_tokens = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
};

std::vector<int> generate_tokens(const ModelWeights &model,
                                 const std::vector<int> &prompt,
                                 const SamplingConfig &sampling);
std::vector<int> generate_tokens_cached(const ModelWeights &model,
                                        const std::vector<int> &prompt,
                                        const SamplingConfig &sampling,
                                        GenerationStats *stats = nullptr);
std::vector<int> generate_tokens_cached(const QuantizedModelWeights &model,
                                        const std::vector<int> &prompt,
                                        const SamplingConfig &sampling,
                                        GenerationStats *stats = nullptr);
