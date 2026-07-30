/* Optional loader for the dependency-free llama2.c checkpoint/tokenizer
 * interchange format. This makes a public TinyStories checkpoint useful as an
 * external end-to-end proof while keeping CI on the tiny safetensors fixture. */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "llama2.hpp"

namespace {

class ReadOnlyFile {
public:
    explicit ReadOnlyFile(const std::string &path) {
#ifndef _WIN32
        descriptor_ = open(path.c_str(), O_RDONLY);
        if (descriptor_ < 0)
            throw std::runtime_error("llama2 checkpoint: cannot open " + path +
                                     ": " + std::strerror(errno));
        struct stat status {};
        if (fstat(descriptor_, &status) != 0 || status.st_size < 0) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("llama2 checkpoint: cannot stat " + path);
        }
        if (uint64_t(status.st_size) >
            uint64_t(std::numeric_limits<size_t>::max())) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::length_error(
                "llama2 checkpoint: file exceeds address space");
        }
        size_ = size_t(status.st_size);
        if (size_ == 0) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("llama2 checkpoint: empty file");
        }
        mapping_ =
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
        if (mapping_ == MAP_FAILED) {
            close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("llama2 checkpoint: mmap failed");
        }
        data_ = static_cast<const unsigned char *>(mapping_);
#else
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("llama2 checkpoint: cannot open " + path);
        owned_.assign(std::istreambuf_iterator<char>(input),
                      std::istreambuf_iterator<char>());
        if (owned_.empty())
            throw std::runtime_error("llama2 checkpoint: empty file");
        size_ = owned_.size();
        data_ = owned_.data();
#endif
    }

    ~ReadOnlyFile() {
#ifndef _WIN32
        if (mapping_ != MAP_FAILED)
            munmap(mapping_, size_);
        if (descriptor_ >= 0)
            close(descriptor_);
#endif
    }

    ReadOnlyFile(const ReadOnlyFile &) = delete;
    ReadOnlyFile &operator=(const ReadOnlyFile &) = delete;

    const unsigned char *data() const { return data_; }
    size_t size() const { return size_; }
    void read_floats(float *destination, const float *source,
                     size_t count) const {
        if (destination == nullptr || source == nullptr)
            throw std::invalid_argument(
                "llama2 checkpoint: null read buffer");
        if (count > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::length_error("llama2 checkpoint: read overflows");
        const uintptr_t mapping = reinterpret_cast<uintptr_t>(data_);
        const uintptr_t first = reinterpret_cast<uintptr_t>(source);
        const size_t bytes = count * sizeof(float);
        if (first < mapping || first - mapping > size_ ||
            bytes > size_ - size_t(first - mapping))
            throw std::runtime_error(
                "llama2 checkpoint: read outside file");
#ifndef _WIN32
        size_t consumed = 0;
        while (consumed < bytes) {
            const ssize_t result = pread(
                descriptor_,
                reinterpret_cast<unsigned char *>(destination) + consumed,
                bytes - consumed, off_t(first - mapping + consumed));
            if (result < 0 && errno == EINTR)
                continue;
            if (result <= 0)
                throw std::runtime_error(
                    "llama2 checkpoint: read failed");
            consumed += size_t(result);
        }
#else
        std::memcpy(destination, source, bytes);
#endif
    }

private:
#ifndef _WIN32
    int descriptor_ = -1;
    void *mapping_ = MAP_FAILED;
#else
    std::vector<unsigned char> owned_;
#endif
    const unsigned char *data_ = nullptr;
    size_t size_ = 0;
};

size_t checked_product(size_t left, size_t right, const char *label) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::length_error(std::string("llama2 checkpoint: ") + label +
                                " overflows");
    return left * right;
}

void checked_add(size_t &total, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - total)
        throw std::length_error("llama2 checkpoint: size overflows");
    total += value;
}

size_t positive_dimension(int32_t value, const char *name) {
    if (value <= 0)
        throw std::runtime_error(std::string("llama2 checkpoint: invalid ") +
                                 name);
    return size_t(value);
}

void copy_floats(float *destination, const float *source, size_t count) {
    std::memcpy(destination, source, count * sizeof(float));
}

void transpose_external(const float *source, Matrix &destination) {
    for (size_t external_row = 0; external_row < destination.cols();
         external_row++) {
        for (size_t external_column = 0;
             external_column < destination.rows(); external_column++) {
            destination.at(external_column, external_row) =
                source[external_row * destination.rows() + external_column];
        }
    }
}

int hex_digit(char value) {
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'a' && value <= 'f')
        return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
        return value - 'A' + 10;
    return -1;
}

struct Llama2Layout {
    ModelConfig config;
    bool shared_classifier = false;
    size_t embedding_count = 0;
    size_t attention_matrix = 0;
    size_t kv_matrix = 0;
    size_t ffn_matrix = 0;
    const float *embeddings = nullptr;
    const float *attention_norms = nullptr;
    const float *queries = nullptr;
    const float *keys = nullptr;
    const float *values = nullptr;
    const float *outputs = nullptr;
    const float *ffn_norms = nullptr;
    const float *gates = nullptr;
    const float *downs = nullptr;
    const float *ups = nullptr;
    const float *final_norm = nullptr;
    const float *classifier = nullptr;
};

Llama2Layout parse_layout(const ReadOnlyFile &file) {
    if (file.size() < sizeof(int32_t) * 7)
        throw std::runtime_error("llama2 checkpoint: truncated header");
    int32_t header[7] = {};
    std::memcpy(header, file.data(), sizeof(header));

    const size_t dim = positive_dimension(header[0], "dimension");
    const size_t ffn_dim = positive_dimension(header[1], "ffn dimension");
    const size_t layers = positive_dimension(header[2], "layer count");
    const size_t heads = positive_dimension(header[3], "head count");
    const size_t kv_heads = positive_dimension(header[4], "kv head count");
    if (header[5] == 0 ||
        header[5] == std::numeric_limits<int32_t>::min())
        throw std::runtime_error("llama2 checkpoint: invalid vocabulary");
    const bool shared_classifier = header[5] > 0;
    const size_t vocabulary =
        size_t(header[5] > 0 ? header[5] : -header[5]);
    const size_t max_sequence =
        positive_dimension(header[6], "context length");
    if (dim % heads != 0)
        throw std::runtime_error(
            "llama2 checkpoint: heads do not divide dimension");
    const size_t head_dim = dim / heads;
    const ModelConfig config{layers, heads, dim, head_dim, ffn_dim,
                             vocabulary, max_sequence, 1, kv_heads};
    config.validate();

    Llama2Layout layout;
    layout.config = config;
    layout.shared_classifier = shared_classifier;
    layout.embedding_count =
        checked_product(vocabulary, dim, "embeddings");
    const size_t norm_count = checked_product(layers, dim, "norms");
    layout.attention_matrix =
        checked_product(dim, dim, "attention matrix");
    const size_t attention_count =
        checked_product(layers, layout.attention_matrix,
                        "attention weights");
    layout.kv_matrix =
        checked_product(dim, config.kv_dim(), "kv matrix");
    const size_t kv_count =
        checked_product(layers, layout.kv_matrix, "kv weights");
    layout.ffn_matrix = checked_product(dim, ffn_dim, "ffn matrix");
    const size_t ffn_count =
        checked_product(layers, layout.ffn_matrix, "ffn weights");
    const size_t frequency_count =
        checked_product(max_sequence, head_dim, "rope frequencies");
    size_t total_floats = 0;
    checked_add(total_floats, layout.embedding_count);
    checked_add(total_floats, norm_count);
    checked_add(total_floats, attention_count);
    checked_add(total_floats, kv_count);
    checked_add(total_floats, kv_count);
    checked_add(total_floats, attention_count);
    checked_add(total_floats, norm_count);
    checked_add(total_floats, ffn_count);
    checked_add(total_floats, ffn_count);
    checked_add(total_floats, ffn_count);
    checked_add(total_floats, dim);
    checked_add(total_floats, frequency_count);
    if (!shared_classifier)
        checked_add(total_floats, layout.embedding_count);
    if (total_floats >
        (std::numeric_limits<size_t>::max() - sizeof(header)) /
            sizeof(float))
        throw std::length_error("llama2 checkpoint: file size overflows");
    if (file.size() != sizeof(header) + total_floats * sizeof(float))
        throw std::runtime_error(
            "llama2 checkpoint: size does not match configuration");

    size_t offset = 0;
    const float *weights =
        reinterpret_cast<const float *>(file.data() + sizeof(header));
    auto take = [&](size_t count) {
        const float *result = weights + offset;
        offset += count;
        return result;
    };
    layout.embeddings = take(layout.embedding_count);
    layout.attention_norms = take(norm_count);
    layout.queries = take(attention_count);
    layout.keys = take(kv_count);
    layout.values = take(kv_count);
    layout.outputs = take(attention_count);
    layout.ffn_norms = take(norm_count);
    layout.gates = take(ffn_count);
    layout.downs = take(ffn_count);
    layout.ups = take(ffn_count);
    layout.final_norm = take(dim);
    (void)take(frequency_count);
    layout.classifier =
        shared_classifier ? layout.embeddings :
                            take(layout.embedding_count);
    if (offset != total_floats)
        throw std::logic_error(
            "llama2 checkpoint: internal layout mismatch");
    return layout;
}

} // namespace

ModelWeights load_llama2c_model(const std::string &path) {
    ReadOnlyFile file(path);
    const Llama2Layout layout = parse_layout(file);
    const ModelConfig &config = layout.config;
    ModelWeights model(config, !layout.shared_classifier);
    model.tied_embeddings = layout.shared_classifier;
    copy_floats(model.embeddings.data(), layout.embeddings,
                layout.embedding_count);
    copy_floats(model.final_norm.data(), layout.final_norm, config.dim);
    if (!layout.shared_classifier)
        copy_floats(model.lm_head.data(), layout.classifier,
                    layout.embedding_count);
    for (size_t layer = 0; layer < config.layers; layer++) {
        LayerWeights &destination = model.layers[layer];
        copy_floats(destination.attention_norm.data(),
                    layout.attention_norms + layer * config.dim,
                    config.dim);
        copy_floats(destination.ffn_norm.data(),
                    layout.ffn_norms + layer * config.dim, config.dim);
        transpose_external(
            layout.queries + layer * layout.attention_matrix,
                           destination.query);
        transpose_external(layout.keys + layer * layout.kv_matrix,
                           destination.key);
        transpose_external(layout.values + layer * layout.kv_matrix,
                           destination.value);
        transpose_external(
            layout.outputs + layer * layout.attention_matrix,
                           destination.output);
        transpose_external(layout.gates + layer * layout.ffn_matrix,
                           destination.gate);
        transpose_external(layout.downs + layer * layout.ffn_matrix,
                           destination.down);
        transpose_external(layout.ups + layer * layout.ffn_matrix,
                           destination.up);
    }
    return model;
}

QuantizedModelWeights load_llama2c_quantized_model(
    const std::string &path, QuantType type) {
    ReadOnlyFile file(path);
    const Llama2Layout layout = parse_layout(file);
    const ModelConfig &config = layout.config;
    QuantizedModelWeights model(config, type, !layout.shared_classifier);
    model.tied_embeddings = layout.shared_classifier;
    file.read_floats(model.embeddings.data(), layout.embeddings,
                     layout.embedding_count);

    const auto quantize_layers =
        [type, &config, &file](const float *source, size_t rows,
                              size_t cols) {
            std::vector<QuantizedMatrix> matrices;
            matrices.reserve(config.layers);
            const size_t elements = checked_product(
                rows, cols, "quantized projection");
            std::vector<float> buffer(elements);
            for (size_t layer = 0; layer < config.layers; layer++) {
                file.read_floats(buffer.data(),
                                 source + layer * elements, elements);
                matrices.emplace_back(QuantizedMatrix::quantize(
                    buffer.data(), rows, cols, type));
            }
            return matrices;
        };

    std::vector<std::vector<float>> attention_norms;
    attention_norms.reserve(config.layers);
    for (size_t layer = 0; layer < config.layers; layer++) {
        attention_norms.emplace_back(config.dim);
        file.read_floats(attention_norms.back().data(),
                         layout.attention_norms + layer * config.dim,
                         config.dim);
    }
    std::vector<QuantizedMatrix> queries = quantize_layers(
        layout.queries, config.dim, config.dim);
    std::vector<QuantizedMatrix> keys = quantize_layers(
        layout.keys, config.kv_dim(), config.dim);
    std::vector<QuantizedMatrix> values = quantize_layers(
        layout.values, config.kv_dim(), config.dim);
    std::vector<QuantizedMatrix> outputs = quantize_layers(
        layout.outputs, config.dim, config.dim);

    std::vector<std::vector<float>> ffn_norms;
    ffn_norms.reserve(config.layers);
    for (size_t layer = 0; layer < config.layers; layer++) {
        ffn_norms.emplace_back(config.dim);
        file.read_floats(ffn_norms.back().data(),
                         layout.ffn_norms + layer * config.dim,
                         config.dim);
    }
    std::vector<QuantizedMatrix> gates = quantize_layers(
        layout.gates, config.ffn_dim, config.dim);
    std::vector<QuantizedMatrix> downs = quantize_layers(
        layout.downs, config.dim, config.ffn_dim);
    std::vector<QuantizedMatrix> ups = quantize_layers(
        layout.ups, config.ffn_dim, config.dim);

    file.read_floats(model.final_norm.data(), layout.final_norm,
                     config.dim);
    if (!layout.shared_classifier)
        file.read_floats(model.lm_head.data(), layout.classifier,
                         layout.embedding_count);
    for (size_t layer = 0; layer < config.layers; layer++) {
        model.layers.emplace_back(
            std::move(queries[layer]), std::move(keys[layer]),
            std::move(values[layer]), std::move(outputs[layer]),
            std::move(gates[layer]), std::move(ups[layer]),
            std::move(downs[layer]), std::move(attention_norms[layer]),
            std::move(ffn_norms[layer]));
    }
    return model;
}

Llama2Tokenizer Llama2Tokenizer::load(const std::string &path,
                                      size_t vocab_size) {
    if (vocab_size == 0 ||
        vocab_size > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("llama2 tokenizer: invalid vocabulary");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("llama2 tokenizer: cannot open " + path);
    uint32_t maximum_length = 0;
    input.read(reinterpret_cast<char *>(&maximum_length),
               sizeof(maximum_length));
    if (!input || maximum_length == 0)
        throw std::runtime_error("llama2 tokenizer: invalid header");

    Llama2Tokenizer tokenizer;
    tokenizer.max_token_length_ = maximum_length;
    tokenizer.vocabulary_.reserve(vocab_size);
    tokenizer.scores_.reserve(vocab_size);
    for (size_t id = 0; id < vocab_size; id++) {
        float score = 0.0f;
        int32_t length = 0;
        input.read(reinterpret_cast<char *>(&score), sizeof(score));
        input.read(reinterpret_cast<char *>(&length), sizeof(length));
        if (!input || !std::isfinite(score) || length <= 0 ||
            size_t(length) > tokenizer.max_token_length_)
            throw std::runtime_error(
                "llama2 tokenizer: malformed vocabulary entry");
        std::string token(size_t(length), '\0');
        input.read(token.data(), length);
        if (!input)
            throw std::runtime_error("llama2 tokenizer: truncated token");
        tokenizer.scores_.push_back(score);
        tokenizer.vocabulary_.push_back(std::move(token));
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("llama2 tokenizer: trailing data");
    for (size_t id = 0; id < tokenizer.vocabulary_.size(); id++)
        tokenizer.token_to_id_.emplace(tokenizer.vocabulary_[id], int(id));
    return tokenizer;
}

std::vector<int> Llama2Tokenizer::encode(const std::string &text, bool bos,
                                         bool eos) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() + 2);
    if (bos)
        tokens.push_back(1);
    if (!text.empty()) {
        const auto prefix = token_to_id_.find(" ");
        if (prefix == token_to_id_.end())
            throw std::runtime_error(
                "llama2 tokenizer: dummy prefix token missing");
        tokens.push_back(prefix->second);
    }
    for (size_t begin = 0; begin < text.size();) {
        size_t end = begin + 1;
        while (end < text.size() && end - begin < 4 &&
               (static_cast<unsigned char>(text[end]) & 0xc0) == 0x80)
            end++;
        const std::string piece = text.substr(begin, end - begin);
        const auto found = token_to_id_.find(piece);
        if (found != token_to_id_.end()) {
            tokens.push_back(found->second);
        } else {
            for (unsigned char byte : piece) {
                const size_t fallback = size_t(byte) + 3;
                if (fallback >= vocabulary_.size())
                    throw std::runtime_error(
                        "llama2 tokenizer: byte fallback unavailable");
                tokens.push_back(int(fallback));
            }
        }
        begin = end;
    }

    for (;;) {
        float best_score = -std::numeric_limits<float>::infinity();
        size_t best_position = tokens.size();
        int best_token = -1;
        for (size_t position = 0; position + 1 < tokens.size(); position++) {
            const std::string merged =
                vocabulary_[size_t(tokens[position])] +
                vocabulary_[size_t(tokens[position + 1])];
            const auto found = token_to_id_.find(merged);
            if (found != token_to_id_.end() &&
                scores_[size_t(found->second)] > best_score) {
                best_score = scores_[size_t(found->second)];
                best_position = position;
                best_token = found->second;
            }
        }
        if (best_position == tokens.size())
            break;
        tokens[best_position] = best_token;
        tokens.erase(tokens.begin() +
                     std::vector<int>::difference_type(best_position + 1));
    }
    if (eos)
        tokens.push_back(2);
    return tokens;
}

std::string Llama2Tokenizer::decode(const std::vector<int> &tokens) const {
    std::string text;
    size_t index = 0;
    int previous = -1;
    if (!tokens.empty() && tokens.front() == 1) {
        previous = 1;
        index = 1;
    }
    for (; index < tokens.size(); index++) {
        const int token = tokens[index];
        if (token < 0 || size_t(token) >= vocabulary_.size())
            throw std::out_of_range(
                "llama2 tokenizer: token id out of range");
        std::string piece = vocabulary_[size_t(token)];
        if (previous == 1 && !piece.empty() && piece[0] == ' ')
            piece.erase(piece.begin());
        if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
            piece[2] == 'x' && piece[5] == '>') {
            const int high = hex_digit(piece[3]);
            const int low = hex_digit(piece[4]);
            if (high >= 0 && low >= 0)
                piece.assign(1, char((high << 4) | low));
        }
        text += piece;
        previous = token;
    }
    return text;
}
