#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "loader.hpp"
#include "quant_model.hpp"

class Llama2Tokenizer {
public:
    static Llama2Tokenizer load(const std::string &path, size_t vocab_size);

    std::vector<int> encode(const std::string &text, bool bos = true,
                            bool eos = false) const;
    std::string decode(const std::vector<int> &tokens) const;
    size_t vocabulary_size() const { return vocabulary_.size(); }

private:
    std::vector<std::string> vocabulary_;
    std::vector<float> scores_;
    std::map<std::string, int> token_to_id_;
    size_t max_token_length_ = 0;
};

ModelWeights load_llama2c_model(const std::string &path);
QuantizedModelWeights load_llama2c_quantized_model(
    const std::string &path, QuantType type);
