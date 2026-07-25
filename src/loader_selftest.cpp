#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "generate.hpp"
#include "loader.hpp"
#include "selftest.hpp"
#include "tokenizer.hpp"

namespace {

bool expect_rejected_truncation() {
    const char *source_path = "tests/fixtures/tiny.safetensors";
    const char *truncated_path = "build/truncated.safetensors";
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream truncated(truncated_path, std::ios::binary);
    char bytes[12] = {};
    source.read(bytes, sizeof(bytes));
    truncated.write(bytes, source.gcount());
    truncated.close();
    try {
        (void)SafeTensorFile::load(truncated_path);
    } catch (const std::runtime_error &error) {
        return std::string(error.what()).find("truncated") !=
               std::string::npos;
    }
    return false;
}

} // namespace

bool run_loader_selftests() {
    try {
        ModelWeights model =
            load_model("tests/fixtures/tiny.safetensors");
        Tokenizer tokenizer = Tokenizer::load(
            "tests/fixtures/tiny.vocab", "tests/fixtures/tiny.merges");
        const std::vector<int> prompt = tokenizer.encode("H");
        SamplingConfig greedy;
        greedy.max_tokens = 8;
        const std::vector<int> first =
            generate_tokens(model, prompt, greedy);
        const std::vector<int> second =
            generate_tokens(model, prompt, greedy);
        std::vector<int> complete = prompt;
        complete.insert(complete.end(), first.begin(), first.end());

        SamplingConfig sampled;
        sampled.max_tokens = 8;
        sampled.temperature = 0.7f;
        sampled.top_p = 0.9f;
        sampled.seed = 42;
        const std::vector<int> sampled_first =
            generate_tokens(model, prompt, sampled);
        const std::vector<int> sampled_second =
            generate_tokens(model, prompt, sampled);

        const bool generation_ok =
            model.config.layers == 2 && model.config.eos_token == 3 &&
            first == second && first == std::vector<int>({1, 2}) &&
            tokenizer.decode(complete) == "Hi!" &&
            sampled_first == sampled_second;
        const bool malformed_ok = expect_rejected_truncation();
        std::printf(
            "  %s loader mmap/config + deterministic generation: %s\n",
            generation_ok && malformed_ok ? "ok" : "FAIL",
            generation_ok ? "Hi!" : "unexpected");
        std::printf("  %s truncated checkpoint rejected clearly\n",
                    malformed_ok ? "ok" : "FAIL");
        return generation_ok && malformed_ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL loader selftest: %s\n", error.what());
        return false;
    }
}
