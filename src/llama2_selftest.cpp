#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "generate.hpp"
#include "llama2.hpp"
#include "selftest.hpp"

bool run_llama2_selftests() {
    const char *model_path = std::getenv("NUTLLM_REAL_MODEL");
    const char *tokenizer_path = std::getenv("NUTLLM_REAL_TOKENIZER");
    if (model_path == nullptr && tokenizer_path == nullptr) {
        std::printf(
            "  skip external checkpoint (set NUTLLM_REAL_MODEL/TOKENIZER)\n");
        return true;
    }
    if (model_path == nullptr || tokenizer_path == nullptr) {
        std::printf("  FAIL both real-checkpoint paths must be set\n");
        return false;
    }
    try {
        ModelWeights model = load_llama2c_model(model_path);
        Llama2Tokenizer tokenizer = Llama2Tokenizer::load(
            tokenizer_path, model.config.vocab_size);
        const std::string prompt = "Once upon a time";
        const std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        const std::string unicode = "A café";
        if (tokenizer.decode(prompt_tokens) != prompt ||
            tokenizer.decode(tokenizer.encode(unicode)) != unicode) {
            std::printf("  FAIL external tokenizer roundtrip\n");
            return false;
        }
        SamplingConfig sampling;
        sampling.max_tokens = 16;
        const auto start = std::chrono::steady_clock::now();
        const std::vector<int> generated =
            generate_tokens_cached(model, prompt_tokens, sampling);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::vector<int> complete = prompt_tokens;
        complete.insert(complete.end(), generated.begin(), generated.end());
        const std::string text = tokenizer.decode(complete);
        const std::string expected =
            "Once upon a time, there was a little girl named Lily.";
        const bool config_ok =
            model.config.dim == 288 && model.config.layers == 6 &&
            model.config.heads == 6 && model.config.vocab_size == 32000;
        const bool text_ok = text.compare(0, expected.size(), expected) == 0;
        std::printf("  %s external 15M checkpoint   %.2f tok/s\n",
                    config_ok && text_ok ? "ok" : "FAIL",
                    generated.empty() ? 0.0 :
                        double(generated.size()) / seconds);
        std::printf("       %s\n", text.c_str());
        return config_ok && text_ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL external checkpoint: %s\n", error.what());
        return false;
    }
}
