/* Byte-level BPE tokenizer.
 *
 * File tokens are hex-encoded, so every byte (including whitespace and NUL)
 * has an unambiguous representation. Encoding starts from individual bytes and
 * repeatedly applies the lowest-ranked adjacent merge. */

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "tokenizer.hpp"

namespace {

int hex_digit(char value) {
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'a' && value <= 'f')
        return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
        return value - 'A' + 10;
    return -1;
}

std::string decode_hex(const std::string &encoded) {
    if (encoded.empty() || encoded.size() % 2 != 0)
        throw std::runtime_error("tokenizer: malformed hex token");
    std::string decoded;
    decoded.reserve(encoded.size() / 2);
    for (size_t index = 0; index < encoded.size(); index += 2) {
        const int high = hex_digit(encoded[index]);
        const int low = hex_digit(encoded[index + 1]);
        if (high < 0 || low < 0)
            throw std::runtime_error("tokenizer: malformed hex token");
        decoded.push_back(char((high << 4) | low));
    }
    return decoded;
}

} // namespace

Tokenizer::Tokenizer(
    std::vector<std::string> vocabulary,
    std::vector<std::pair<std::string, std::string>> merges)
    : id_to_token_(std::move(vocabulary)) {
    if (id_to_token_.empty())
        throw std::invalid_argument("tokenizer: empty vocabulary");
    for (size_t id = 0; id < id_to_token_.size(); id++) {
        if (id > size_t(std::numeric_limits<int>::max()) ||
            id_to_token_[id].empty() ||
            !token_to_id_.emplace(id_to_token_[id], int(id)).second)
            throw std::invalid_argument("tokenizer: empty or duplicate token");
    }
    for (size_t rank = 0; rank < merges.size(); rank++) {
        if (merges[rank].first.empty() || merges[rank].second.empty() ||
            !merge_rank_.emplace(merges[rank], rank).second)
            throw std::invalid_argument("tokenizer: invalid duplicate merge");
        if (token_to_id_.count(merges[rank].first +
                               merges[rank].second) == 0)
            throw std::invalid_argument(
                "tokenizer: merged token missing from vocabulary");
    }
}

Tokenizer Tokenizer::load(const std::string &vocabulary_path,
                          const std::string &merges_path) {
    std::ifstream vocabulary_file(vocabulary_path);
    if (!vocabulary_file)
        throw std::runtime_error("tokenizer: cannot open vocabulary");
    std::vector<std::string> vocabulary;
    std::string line;
    size_t expected_id = 0;
    while (std::getline(vocabulary_file, line)) {
        std::istringstream stream(line);
        size_t id = 0;
        std::string encoded;
        if (!(stream >> id >> encoded) || id != expected_id)
            throw std::runtime_error(
                "tokenizer: vocabulary ids must be contiguous");
        vocabulary.push_back(decode_hex(encoded));
        expected_id++;
    }
    if (!vocabulary_file.eof())
        throw std::runtime_error("tokenizer: failed reading vocabulary");

    std::ifstream merges_file(merges_path);
    if (!merges_file)
        throw std::runtime_error("tokenizer: cannot open merges");
    std::vector<std::pair<std::string, std::string>> merges;
    while (std::getline(merges_file, line)) {
        if (line.empty())
            continue;
        std::istringstream stream(line);
        std::string left;
        std::string right;
        std::string trailing;
        if (!(stream >> left >> right) || (stream >> trailing))
            throw std::runtime_error("tokenizer: malformed merge");
        merges.emplace_back(decode_hex(left), decode_hex(right));
    }
    if (!merges_file.eof())
        throw std::runtime_error("tokenizer: failed reading merges");
    return Tokenizer(std::move(vocabulary), std::move(merges));
}

std::vector<int> Tokenizer::encode(const std::string &text) const {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());
    for (unsigned char byte : text)
        symbols.emplace_back(1, char(byte));

    for (;;) {
        size_t best_position = symbols.size();
        size_t best_rank = std::numeric_limits<size_t>::max();
        for (size_t index = 0; index + 1 < symbols.size(); index++) {
            const auto found =
                merge_rank_.find({symbols[index], symbols[index + 1]});
            if (found != merge_rank_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_position = index;
            }
        }
        if (best_position == symbols.size())
            break;
        const std::string left = symbols[best_position];
        const std::string right = symbols[best_position + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() &&
                symbols[index] == left && symbols[index + 1] == right) {
                merged.push_back(left + right);
                index += 2;
            } else {
                merged.push_back(symbols[index]);
                index++;
            }
        }
        symbols = std::move(merged);
    }

    std::vector<int> tokens;
    tokens.reserve(symbols.size());
    for (const std::string &symbol : symbols) {
        const auto found = token_to_id_.find(symbol);
        if (found == token_to_id_.end())
            throw std::runtime_error(
                "tokenizer: input byte absent from vocabulary");
        tokens.push_back(found->second);
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int> &tokens) const {
    std::string text;
    for (int token : tokens) {
        if (token < 0 || size_t(token) >= id_to_token_.size())
            throw std::out_of_range("tokenizer: token id out of range");
        text += id_to_token_[size_t(token)];
    }
    return text;
}
