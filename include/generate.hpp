#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "loader.hpp"

struct SamplingConfig {
    size_t max_tokens = 32;
    float temperature = 0.0f;
    float top_p = 1.0f;
    uint32_t seed = 1;
};

std::vector<int> generate_tokens(const ModelWeights &model,
                                 const std::vector<int> &prompt,
                                 const SamplingConfig &sampling);
