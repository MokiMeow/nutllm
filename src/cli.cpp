#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "generate.hpp"
#include "llama2.hpp"
#include "tokenizer.hpp"

namespace {

size_t parse_size(const char *text, const char *option) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string("invalid ") + option);
    return size_t(value);
}

float parse_float(const char *text, const char *option) {
    errno = 0;
    char *end = nullptr;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0')
        throw std::invalid_argument(std::string("invalid ") + option);
    return value;
}

const char *take_value(int argc, char **argv, int &index) {
    if (index + 1 >= argc)
        throw std::invalid_argument(std::string("missing value for ") +
                                    argv[index]);
    return argv[++index];
}

void usage() {
    std::fprintf(
        stderr,
        "usage: nutllm --model FILE --vocab FILE --merges FILE "
        "--prompt TEXT [options]\n"
        "   or: nutllm --model FILE --tokenizer FILE "
        "--prompt TEXT [--max-tokens N] [--temperature T] "
        "[--top-p P] [--seed N] [--threads N] "
        "[--quant fp32|int8|int4] [--stats]\n");
}

void print_stats(const std::string &format, size_t threads,
                 double load_seconds, size_t model_bytes,
                 const GenerationStats &stats) {
    const double prefill_rate =
        stats.prefill_seconds > 0.0 ?
            double(stats.prefill_tokens) / stats.prefill_seconds : 0.0;
    const double decode_rate =
        stats.decode_seconds > 0.0 ?
            double(stats.generated_tokens) / stats.decode_seconds : 0.0;
    std::fprintf(
        stderr,
        "stats format=%s threads=%zu model_mib=%.2f load_s=%.3f "
        "prefill_tokens=%zu prefill_tok_s=%.3f decode_tokens=%zu "
        "decode_tok_s=%.3f\n",
        format.c_str(), threads,
        double(model_bytes) / (1024.0 * 1024.0), load_seconds,
        stats.prefill_tokens, prefill_rate, stats.generated_tokens,
        decode_rate);
}

} // namespace

int run_model_cli(int argc, char **argv) {
    try {
        std::string model_path;
        std::string vocabulary_path;
        std::string merges_path;
        std::string tokenizer_path;
        std::string quant_format = "fp32";
        std::string prompt;
        bool show_stats = false;
        SamplingConfig sampling;
        for (int index = 1; index < argc; index++) {
            const std::string option = argv[index];
            if (option == "--model")
                model_path = take_value(argc, argv, index);
            else if (option == "--vocab")
                vocabulary_path = take_value(argc, argv, index);
            else if (option == "--merges")
                merges_path = take_value(argc, argv, index);
            else if (option == "--tokenizer")
                tokenizer_path = take_value(argc, argv, index);
            else if (option == "--quant")
                quant_format = take_value(argc, argv, index);
            else if (option == "--stats")
                show_stats = true;
            else if (option == "--prompt")
                prompt = take_value(argc, argv, index);
            else if (option == "--max-tokens")
                sampling.max_tokens =
                    parse_size(take_value(argc, argv, index), "--max-tokens");
            else if (option == "--temperature")
                sampling.temperature =
                    parse_float(take_value(argc, argv, index),
                                "--temperature");
            else if (option == "--top-p")
                sampling.top_p =
                    parse_float(take_value(argc, argv, index), "--top-p");
            else if (option == "--threads")
                sampling.threads =
                    parse_size(take_value(argc, argv, index), "--threads");
            else if (option == "--seed") {
                const size_t seed =
                    parse_size(take_value(argc, argv, index), "--seed");
                if (seed > std::numeric_limits<uint32_t>::max())
                    throw std::invalid_argument("invalid --seed");
                sampling.seed = uint32_t(seed);
            } else {
                throw std::invalid_argument("unknown option " + option);
            }
        }
        const bool llama2_format = !tokenizer_path.empty();
        const bool byte_bpe_format =
            !vocabulary_path.empty() && !merges_path.empty();
        const bool any_byte_bpe_option =
            !vocabulary_path.empty() || !merges_path.empty();
        if (model_path.empty() || prompt.empty() ||
            (llama2_format && any_byte_bpe_option) ||
            (!llama2_format && !byte_bpe_format) ||
            (!llama2_format && quant_format != "fp32") ||
            (quant_format != "fp32" && quant_format != "int8" &&
             quant_format != "int4")) {
            usage();
            return 2;
        }
        if (llama2_format) {
            if (quant_format == "fp32") {
                const auto load_start = std::chrono::steady_clock::now();
                ModelWeights model = load_llama2c_model(model_path);
                const double load_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - load_start).count();
                Llama2Tokenizer tokenizer = Llama2Tokenizer::load(
                    tokenizer_path, model.config.vocab_size);
                const std::vector<int> prompt_tokens =
                    tokenizer.encode(prompt);
                GenerationStats stats;
                const std::vector<int> generated = generate_tokens_cached(
                    model, prompt_tokens, sampling,
                    show_stats ? &stats : nullptr);
                std::vector<int> complete = prompt_tokens;
                complete.insert(complete.end(), generated.begin(),
                                generated.end());
                std::printf("%s\n", tokenizer.decode(complete).c_str());
                if (show_stats)
                    print_stats(quant_format, sampling.threads, load_seconds,
                                model.storage_bytes(), stats);
            } else {
                const QuantType type = quant_format == "int8" ?
                                           QuantType::int8 :
                                           QuantType::int4;
                const auto load_start = std::chrono::steady_clock::now();
                QuantizedModelWeights model =
                    load_llama2c_quantized_model(model_path, type);
                const double load_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - load_start).count();
                Llama2Tokenizer tokenizer = Llama2Tokenizer::load(
                    tokenizer_path, model.config.vocab_size);
                const std::vector<int> prompt_tokens =
                    tokenizer.encode(prompt);
                GenerationStats stats;
                const std::vector<int> generated = generate_tokens_cached(
                    model, prompt_tokens, sampling,
                    show_stats ? &stats : nullptr);
                std::vector<int> complete = prompt_tokens;
                complete.insert(complete.end(), generated.begin(),
                                generated.end());
                std::printf("%s\n", tokenizer.decode(complete).c_str());
                if (show_stats)
                    print_stats(quant_format, sampling.threads, load_seconds,
                                model.storage_bytes(), stats);
            }
        } else {
            ModelWeights model = load_model(model_path);
            Tokenizer tokenizer =
                Tokenizer::load(vocabulary_path, merges_path);
            if (tokenizer.vocabulary_size() != model.config.vocab_size)
                throw std::runtime_error(
                    "tokenizer vocabulary does not match model");
            const std::vector<int> prompt_tokens = tokenizer.encode(prompt);
            const std::vector<int> generated =
                generate_tokens_cached(model, prompt_tokens, sampling);
            std::vector<int> complete = prompt_tokens;
            complete.insert(complete.end(), generated.begin(),
                            generated.end());
            std::printf("%s\n", tokenizer.decode(complete).c_str());
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "nutllm: %s\n", error.what());
        return 1;
    }
}
