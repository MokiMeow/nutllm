/* End-to-end synthetic transformer and byte-level BPE tests. */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "model.hpp"
#include "selftest.hpp"
#include "tokenizer.hpp"

namespace {

void identity(Matrix &matrix) {
    matrix.zero();
    const size_t diagonal = std::min(matrix.rows(), matrix.cols());
    for (size_t index = 0; index < diagonal; index++)
        matrix.at(index, index) = 1.0f;
}

void fill_random(Matrix &matrix, std::mt19937 &random, float scale) {
    std::uniform_real_distribution<float> distribution(-scale, scale);
    for (size_t index = 0; index < matrix.size(); index++)
        matrix.data()[index] = distribution(random);
}

double max_difference(const Matrix &left, const Matrix &right,
                      size_t rows = 0) {
    const size_t compared_rows = rows == 0 ? left.rows() : rows;
    if (left.cols() != right.cols() ||
        compared_rows > left.rows() || compared_rows > right.rows())
        return std::numeric_limits<double>::infinity();
    double difference = 0.0;
    for (size_t index = 0; index < compared_rows * left.cols(); index++) {
        difference = std::max(
            difference,
            std::fabs(double(left.data()[index]) -
                      double(right.data()[index])));
    }
    return difference;
}

bool report(bool condition, const char *name, double difference = 0.0) {
    std::printf("  %-4s %-28s diff=%.2e\n",
                condition ? "ok" : "FAIL", name, difference);
    return condition;
}

ModelConfig test_config(size_t sequence = 4) {
    return ModelConfig{
        2, 2, 4, 2, 6, 256, sequence,
    };
}

std::vector<LayerWeights> random_layers(const ModelConfig &config) {
    std::mt19937 random(12345);
    std::vector<LayerWeights> layers;
    layers.reserve(config.layers);
    for (size_t layer = 0; layer < config.layers; layer++) {
        layers.emplace_back(config);
        LayerWeights &weights = layers.back();
        fill_random(weights.query, random, 0.15f);
        fill_random(weights.key, random, 0.15f);
        fill_random(weights.value, random, 0.15f);
        fill_random(weights.output, random, 0.15f);
        fill_random(weights.gate, random, 0.15f);
        fill_random(weights.up, random, 0.15f);
        fill_random(weights.down, random, 0.15f);
    }
    return layers;
}

bool attention_fixture_matches() {
    ModelConfig config{1, 1, 2, 2, 2, 256, 2};
    LayerWeights weights(config);
    identity(weights.query);
    identity(weights.key);
    identity(weights.value);
    identity(weights.output);
    Matrix input(2, 2);
    input.at(0, 0) = 1.0f;
    input.at(0, 1) = 0.0f;
    input.at(1, 0) = 0.0f;
    input.at(1, 1) = 1.0f;
    Matrix output(2, 2);
    attention_forward(input, weights, config, output);

    std::ifstream fixture("tests/fixtures/attention_identity.txt");
    Matrix expected(2, 2);
    if (!fixture ||
        !(fixture >> expected.at(0, 0) >> expected.at(0, 1) >>
          expected.at(1, 0) >> expected.at(1, 1)))
        return report(false, "attention fixture file");
    const double difference = max_difference(expected, output);
    return report(difference < 2e-6,
                  "attention fixture", difference);
}

bool causal_mask_blocks_future() {
    ModelConfig config{1, 2, 4, 2, 4, 256, 3};
    LayerWeights weights(config);
    identity(weights.query);
    identity(weights.key);
    identity(weights.value);
    identity(weights.output);

    Matrix original(3, 4);
    for (size_t index = 0; index < original.size(); index++)
        original.data()[index] = float(index + 1) * 0.1f;
    Matrix perturbed(3, 4);
    for (size_t index = 0; index < original.size(); index++)
        perturbed.data()[index] = original.data()[index];
    for (size_t column = 0; column < 4; column++)
        perturbed.at(2, column) += 1000.0f;

    Matrix before(3, 4);
    Matrix after(3, 4);
    attention_forward(original, weights, config, before);
    attention_forward(perturbed, weights, config, after);
    const double difference = max_difference(before, after, 2);
    return report(difference == 0.0,
                  "causal future perturbation", difference);
}

bool decoder_matches_reference() {
    const ModelConfig config = test_config();
    std::vector<LayerWeights> layers = random_layers(config);
    Matrix input(4, config.dim);
    std::mt19937 random(999);
    fill_random(input, random, 0.5f);
    Matrix reference(4, config.dim);
    Matrix output(4, config.dim);
    Matrix repeated(4, config.dim);
    decoder_stack(input, layers, config, reference, true);
    decoder_stack(input, layers, config, output, false);
    decoder_stack(input, layers, config, repeated, false);
    const double reference_difference = max_difference(reference, output);
    const double repeat_difference = max_difference(output, repeated);
    bool finite = true;
    for (size_t index = 0; index < output.size(); index++)
        finite = finite && std::isfinite(output.data()[index]);
    return report(finite && reference_difference < 2e-5 &&
                      repeat_difference == 0.0,
                  "decoder stack reference",
                  std::max(reference_difference, repeat_difference));
}

std::string hex_encode(const std::string &token) {
    static const char digits[] = "0123456789abcdef";
    std::string output;
    for (unsigned char byte : token) {
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 15]);
    }
    return output;
}

Tokenizer make_tokenizer() {
    std::vector<std::string> vocabulary;
    vocabulary.reserve(260);
    for (size_t byte = 0; byte < 256; byte++)
        vocabulary.emplace_back(1, char(byte));
    for (const char *token : {"he", "hel", "hell", "hello"})
        vocabulary.emplace_back(token);
    return Tokenizer(
        std::move(vocabulary),
        {{"h", "e"}, {"he", "l"}, {"hel", "l"}, {"hell", "o"}});
}

bool tokenizer_round_trips() {
    const std::string text = "hello, UTF-8: \xe4\xb8\x96\xe7\x95\x8c";
    const Tokenizer tokenizer = make_tokenizer();
    const std::vector<int> encoded = tokenizer.encode(text);
    bool passed = tokenizer.decode(encoded) == text &&
                  encoded.size() < text.size();

    std::vector<std::string> vocabulary;
    vocabulary.reserve(260);
    for (size_t byte = 0; byte < 256; byte++)
        vocabulary.emplace_back(1, char(byte));
    for (const char *token : {"he", "hel", "hell", "hello"})
        vocabulary.emplace_back(token);
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"he", "l"}, {"hel", "l"}, {"hell", "o"},
    };
    {
        std::ofstream vocab_file("build/tokenizer-test.vocab");
        for (size_t id = 0; id < vocabulary.size(); id++)
            vocab_file << id << ' ' << hex_encode(vocabulary[id]) << '\n';
        std::ofstream merges_file("build/tokenizer-test.merges");
        for (const auto &merge : merges) {
            merges_file << hex_encode(merge.first) << ' '
                        << hex_encode(merge.second) << '\n';
        }
    }
    try {
        const Tokenizer loaded =
            Tokenizer::load("build/tokenizer-test.vocab",
                            "build/tokenizer-test.merges");
        passed = passed && loaded.decode(loaded.encode(text)) == text;
    } catch (const std::exception &) {
        passed = false;
    }
    std::remove("build/tokenizer-test.vocab");
    std::remove("build/tokenizer-test.merges");
    return report(passed, "BPE ASCII/UTF-8 roundtrip");
}

} // namespace

bool run_transformer_selftests() {
    std::printf("\ntransformer + tokenizer (layout [batch=1][head][seq][dim])\n");
    bool passed = true;
    passed = attention_fixture_matches() && passed;
    passed = causal_mask_blocks_future() && passed;
    passed = decoder_matches_reference() && passed;
    passed = tokenizer_round_trips() && passed;
    return passed;
}
