/* Autoregressive generation without a KV cache.
 *
 * Milestone 3 intentionally recomputes the prefix each step. Milestone 4
 * replaces that quadratic history work while retaining this path as the
 * differential reference. */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include "generate.hpp"
#include "ops.hpp"

namespace {

size_t choose_token(const std::vector<float> &logits,
                    const SamplingConfig &sampling, std::mt19937 &rng) {
    if (logits.empty())
        throw std::invalid_argument("sampling: empty logits");
    if (sampling.temperature <= 0.0f)
        return size_t(std::max_element(logits.begin(), logits.end()) -
                      logits.begin());
    if (!std::isfinite(sampling.temperature) ||
        sampling.top_p <= 0.0f || sampling.top_p > 1.0f)
        throw std::invalid_argument("sampling: invalid temperature or top-p");

    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<std::pair<float, size_t>> probabilities;
    probabilities.reserve(logits.size());
    double total = 0.0;
    for (size_t token = 0; token < logits.size(); token++) {
        const float probability =
            std::exp((logits[token] - maximum) / sampling.temperature);
        if (!std::isfinite(probability))
            throw std::runtime_error("sampling: non-finite probability");
        probabilities.emplace_back(probability, token);
        total += probability;
    }
    if (!(total > 0.0))
        throw std::runtime_error("sampling: zero probability mass");
    std::sort(probabilities.begin(), probabilities.end(),
              [](const auto &left, const auto &right) {
                  if (left.first != right.first)
                      return left.first > right.first;
                  return left.second < right.second;
              });

    const double cutoff = sampling.top_p * total;
    double nucleus = 0.0;
    size_t count = 0;
    do {
        nucleus += probabilities[count].first;
        count++;
    } while (count < probabilities.size() && nucleus < cutoff);

    std::uniform_real_distribution<double> draw(0.0, nucleus);
    const double target = draw(rng);
    double cumulative = 0.0;
    for (size_t index = 0; index < count; index++) {
        cumulative += probabilities[index].first;
        if (target <= cumulative)
            return probabilities[index].second;
    }
    return probabilities[count - 1].second;
}

void logits_for_prefix(const ModelWeights &model,
                       const std::vector<int> &tokens,
                       std::vector<float> &logits) {
    if (tokens.empty() || tokens.size() > model.config.max_seq)
        throw std::invalid_argument("generation: invalid prefix length");
    Matrix embedded(tokens.size(), model.config.dim);
    for (size_t row = 0; row < tokens.size(); row++) {
        if (tokens[row] < 0 ||
            size_t(tokens[row]) >= model.config.vocab_size)
            throw std::out_of_range("generation: token id out of range");
        const float *source =
            model.embeddings.data() + size_t(tokens[row]) * model.config.dim;
        std::memcpy(embedded.data() + row * model.config.dim, source,
                    model.config.dim * sizeof(float));
    }
    Matrix hidden(tokens.size(), model.config.dim);
    decoder_stack(embedded, model.layers, model.config, hidden);
    std::vector<float> normalized(model.config.dim);
    rmsnorm(hidden.data() + (tokens.size() - 1) * model.config.dim,
            model.final_norm.data(), normalized.data(), model.config.dim,
            1e-5f);
    matvec(model.lm_head, normalized.data(), logits.data());
}

} // namespace

std::vector<int> generate_tokens(const ModelWeights &model,
                                 const std::vector<int> &prompt,
                                 const SamplingConfig &sampling) {
    model.config.validate();
    if (prompt.empty())
        throw std::invalid_argument("generation: prompt cannot be empty");
    if (prompt.size() > model.config.max_seq)
        throw std::length_error("generation: prompt exceeds context");

    std::vector<int> context = prompt;
    std::vector<int> generated;
    std::vector<float> logits(model.config.vocab_size);
    std::mt19937 rng(sampling.seed);
    while (generated.size() < sampling.max_tokens &&
           context.size() < model.config.max_seq) {
        logits_for_prefix(model, context, logits);
        const size_t token = choose_token(logits, sampling, rng);
        if (token == model.config.eos_token)
            break;
        generated.push_back(int(token));
        context.push_back(int(token));
    }
    return generated;
}
