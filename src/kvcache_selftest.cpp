#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "generate.hpp"
#include "kvcache.hpp"
#include "selftest.hpp"

namespace {

void fill_random(Matrix &matrix, std::mt19937 &rng, float scale) {
    std::uniform_real_distribution<float> distribution(-scale, scale);
    for (size_t index = 0; index < matrix.size(); index++)
        matrix.data()[index] = distribution(rng);
}

void randomize_model(ModelWeights &model) {
    std::mt19937 rng(20260725);
    fill_random(model.embeddings, rng, 0.5f);
    fill_random(model.lm_head, rng, 0.2f);
    for (LayerWeights &layer : model.layers) {
        fill_random(layer.query, rng, 0.2f);
        fill_random(layer.key, rng, 0.2f);
        fill_random(layer.value, rng, 0.2f);
        fill_random(layer.output, rng, 0.2f);
        fill_random(layer.gate, rng, 0.2f);
        fill_random(layer.up, rng, 0.2f);
        fill_random(layer.down, rng, 0.2f);
        for (float &weight : layer.attention_norm)
            weight = 1.0f;
        for (float &weight : layer.ffn_norm)
            weight = 1.0f;
    }
}

Matrix uncached_hidden(const ModelWeights &model,
                       const std::vector<int> &tokens) {
    Matrix embedded(tokens.size(), model.config.dim);
    for (size_t row = 0; row < tokens.size(); row++)
        std::memcpy(
            embedded.data() + row * model.config.dim,
            model.embeddings.data() + size_t(tokens[row]) * model.config.dim,
            model.config.dim * sizeof(float));
    Matrix hidden(tokens.size(), model.config.dim);
    decoder_stack(embedded, model.layers, model.config, hidden);
    return hidden;
}

float last_row_diff(const Matrix &reference,
                    const std::vector<float> &candidate) {
    float worst = 0.0f;
    const float *last =
        reference.data() + (reference.rows() - 1) * reference.cols();
    for (size_t index = 0; index < reference.cols(); index++)
        worst = std::max(worst, std::fabs(last[index] - candidate[index]));
    return worst;
}

} // namespace

bool run_kvcache_selftests() {
    try {
        const ModelConfig config{2, 4, 8, 2, 12, 8, 16, 7, 2};
        ModelWeights random_model(config);
        randomize_model(random_model);
        KVCache cache(config);
        const std::vector<int> prefix{0, 1, 2};
        Matrix cached_prefix =
            prefill_tokens(random_model, prefix, cache);
        Matrix reference_prefix = uncached_hidden(random_model, prefix);
        std::vector<float> cached_prefix_last(config.dim);
        std::memcpy(
            cached_prefix_last.data(),
            cached_prefix.data() +
                (cached_prefix.rows() - 1) * cached_prefix.cols(),
            config.dim * sizeof(float));
        const float prefill_diff =
            last_row_diff(reference_prefix, cached_prefix_last);

        const std::vector<float> cached_next =
            decode_token(random_model, 3, cache, 3);
        const std::vector<int> extended{0, 1, 2, 3};
        Matrix reference_next = uncached_hidden(random_model, extended);
        const float decode_diff = last_row_diff(reference_next, cached_next);
        const size_t expected_bytes =
            2 * config.layers * config.max_seq * config.kv_dim() *
            sizeof(float);
        const bool layout_ok =
            cache.length() == extended.size() &&
            cache.memory_bytes() == expected_bytes;

        ModelWeights tiny =
            load_model("tests/fixtures/tiny.safetensors");
        SamplingConfig sampling;
        sampling.max_tokens = 8;
        SamplingConfig threaded_sampling = sampling;
        threaded_sampling.threads = 3;
        bool token_ids_ok = true;
        for (const std::vector<int> &prompt :
             {std::vector<int>{0}, std::vector<int>{1},
              std::vector<int>{2}, std::vector<int>{0, 1}}) {
            token_ids_ok =
                token_ids_ok &&
                generate_tokens(tiny, prompt, sampling) ==
                    generate_tokens_cached(tiny, prompt, sampling) &&
                generate_tokens_cached(tiny, prompt, sampling) ==
                    generate_tokens_cached(tiny, prompt,
                                           threaded_sampling);
        }

        const bool numerical_ok =
            prefill_diff < 1e-5f && decode_diff < 1e-5f;
        std::printf(
            "  %s kv cache differential       prefill=%.2e decode=%.2e\n",
            numerical_ok ? "ok" : "FAIL", double(prefill_diff),
            double(decode_diff));
        std::printf(
            "  %s cached/uncached token ids  prompts=4 bytes=%zu\n",
            token_ids_ok && layout_ok ? "ok" : "FAIL", cache.memory_bytes());
        return numerical_ok && token_ids_ok && layout_ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL kv cache selftest: %s\n", error.what());
        return false;
    }
}

void run_kvcache_benchmark() {
    using Clock = std::chrono::steady_clock;
    const ModelConfig config{1, 4, 64, 16, 128, 256, 256, 255};
    ModelWeights model(config);
    randomize_model(model);
    std::printf("\nkv-cache benchmark (1 layer, dim=64, 4 heads, 1 thread)\n");
    std::printf("  context   prefill tok/s   decode us/token\n");
    for (size_t length : {16u, 64u, 128u}) {
        std::vector<int> tokens(length);
        for (size_t index = 0; index < length; index++)
            tokens[index] = int(index % (config.vocab_size - 1));
        std::vector<double> prefill_times;
        std::vector<double> decode_times;
        for (size_t run = 0; run < 3; run++) {
            KVCache cache(config);
            const auto prefill_start = Clock::now();
            (void)prefill_tokens(model, tokens, cache);
            prefill_times.push_back(std::chrono::duration<double>(
                Clock::now() - prefill_start).count());
            const auto decode_start = Clock::now();
            (void)decode_token(model, 1, cache);
            decode_times.push_back(std::chrono::duration<double>(
                Clock::now() - decode_start).count());
        }
        std::sort(prefill_times.begin(), prefill_times.end());
        std::sort(decode_times.begin(), decode_times.end());
        std::printf("  %-9zu %12.0f %17.1f\n", length,
                    double(length) / prefill_times[1],
                    decode_times[1] * 1e6);
    }
    KVCache cache(config);
    std::printf("  allocated cache: %zu bytes (2*L*S*D*sizeof(float))\n",
                cache.memory_bytes());
}
