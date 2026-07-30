#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "generate.hpp"
#include "loader.hpp"
#include "llama2.hpp"
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

size_t expected_storage_bytes(const ModelWeights &model) {
    size_t elements =
        model.embeddings.size() + model.final_norm.size() +
        model.lm_head.size();
    for (const LayerWeights &layer : model.layers) {
        elements += layer.query.size() + layer.key.size() +
                    layer.value.size() + layer.output.size() +
                    layer.gate.size() + layer.up.size() +
                    layer.down.size() + layer.attention_norm.size() +
                    layer.ffn_norm.size();
    }
    return elements * sizeof(float);
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
            model.storage_bytes() == expected_storage_bytes(model) &&
            first == second && first == std::vector<int>({1, 2}) &&
            tokenizer.decode(complete) == "Hi!" &&
            sampled_first == sampled_second;
        const bool malformed_ok = expect_rejected_truncation();
        ModelWeights llama2 =
            load_llama2c_model("tests/fixtures/tiny-llama2.bin");
        Llama2Tokenizer llama2_tokenizer = Llama2Tokenizer::load(
            "tests/fixtures/tiny-llama2.tokenizer",
            llama2.config.vocab_size);
        const std::vector<int> llama2_prompt =
            llama2_tokenizer.encode("H");
        const std::vector<int> llama2_generated =
            generate_tokens_cached(llama2, llama2_prompt, greedy);
        std::vector<int> llama2_complete = llama2_prompt;
        llama2_complete.insert(llama2_complete.end(),
                               llama2_generated.begin(),
                               llama2_generated.end());
        ModelWeights llama2_shared =
            load_llama2c_model("tests/fixtures/tiny-llama2-shared.bin");
        QuantizedModelWeights llama2_int8 =
            load_llama2c_quantized_model(
                "tests/fixtures/tiny-llama2.bin", QuantType::int8);
        QuantizedModelWeights llama2_int4 =
            load_llama2c_quantized_model(
                "tests/fixtures/tiny-llama2.bin", QuantType::int4);
        const std::vector<int> llama2_int8_generated =
            generate_tokens_cached(llama2_int8, llama2_prompt, greedy);
        const std::vector<int> llama2_int4_generated =
            generate_tokens_cached(llama2_int4, llama2_prompt, greedy);
        const bool llama2_ok =
            llama2.config.eos_token == 1 &&
            !llama2.tied_embeddings && llama2.lm_head.data() != nullptr &&
            llama2_shared.tied_embeddings &&
            llama2_shared.lm_head.data() == nullptr &&
            llama2_shared.output_weights().data() ==
                llama2_shared.embeddings.data() &&
            llama2_generated == std::vector<int>({4, 5}) &&
            llama2_int8_generated == llama2_generated &&
            llama2_int4_generated == llama2_generated &&
            llama2_tokenizer.decode(llama2_complete) == "Hi!";
        std::printf(
            "  %s loader mmap/config + deterministic generation: %s\n",
            generation_ok && malformed_ok ? "ok" : "FAIL",
            generation_ok ? "Hi!" : "unexpected");
        std::printf("  %s truncated checkpoint rejected clearly\n",
                    malformed_ok ? "ok" : "FAIL");
        std::printf("  %s llama2.c fp32/INT8/INT4 + tokenizer: Hi!\n",
                    llama2_ok ? "ok" : "FAIL");
        return generation_ok && malformed_ok && llama2_ok;
    } catch (const std::exception &error) {
        std::printf("  FAIL loader selftest: %s\n", error.what());
        return false;
    }
}
