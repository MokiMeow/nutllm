#pragma once

#include <functional>
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
    // Transparent string-key lookup avoids allocating/copying a pair of
    // adjacent symbols on every BPE merge-rank probe.
    std::map<std::string, std::map<std::string, size_t, std::less<>>,
             std::less<>> merge_rank_;
};
