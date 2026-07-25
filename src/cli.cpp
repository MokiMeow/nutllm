#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "generate.hpp"
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
        "--prompt TEXT [--max-tokens N] [--temperature T] "
        "[--top-p P] [--seed N]\n");
}

} // namespace

int run_model_cli(int argc, char **argv) {
    try {
        std::string model_path;
        std::string vocabulary_path;
        std::string merges_path;
        std::string prompt;
        SamplingConfig sampling;
        for (int index = 1; index < argc; index++) {
            const std::string option = argv[index];
            if (option == "--model")
                model_path = take_value(argc, argv, index);
            else if (option == "--vocab")
                vocabulary_path = take_value(argc, argv, index);
            else if (option == "--merges")
                merges_path = take_value(argc, argv, index);
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
        if (model_path.empty() || vocabulary_path.empty() ||
            merges_path.empty() || prompt.empty()) {
            usage();
            return 2;
        }
        ModelWeights model = load_model(model_path);
        Tokenizer tokenizer =
            Tokenizer::load(vocabulary_path, merges_path);
        if (tokenizer.vocabulary_size() != model.config.vocab_size)
            throw std::runtime_error(
                "tokenizer vocabulary does not match model");
        const std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        const std::vector<int> generated =
            generate_tokens(model, prompt_tokens, sampling);
        std::vector<int> complete = prompt_tokens;
        complete.insert(complete.end(), generated.begin(), generated.end());
        std::printf("%s\n", tokenizer.decode(complete).c_str());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "nutllm: %s\n", error.what());
        return 1;
    }
}
