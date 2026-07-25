#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

class Tokenizer {
public:
    Tokenizer(std::vector<std::string> vocabulary,
              std::vector<std::pair<std::string, std::string>> merges);

    static Tokenizer load(const std::string &vocabulary_path,
                          const std::string &merges_path);

    std::vector<int> encode(const std::string &text) const;
    std::string decode(const std::vector<int> &tokens) const;
    size_t vocabulary_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::map<std::string, int> token_to_id_;
    std::map<std::pair<std::string, std::string>, size_t> merge_rank_;
};
