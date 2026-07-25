#pragma once

#include <cstddef>
#include <vector>

#include "tensor.hpp"

double perplexity(const Matrix &logits,
                  const std::vector<size_t> &targets);
