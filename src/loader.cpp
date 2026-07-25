/* Safetensors loader for float32 Llama-style checkpoints.
 *
 * The tensor payload stays memory-mapped for the lifetime of SafeTensorFile.
 * ModelWeights then transposes external [out, in] matrices into nutllm's
 * internal [in, out] layout, making the conversion explicit and testable. */

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "loader.hpp"

namespace {

struct Json {
    enum class Kind { string, number, array, object };
    Kind kind = Kind::object;
    std::string string;
    uint64_t number = 0;
    std::vector<Json> array;
    std::map<std::string, Json> object;
};

class JsonParser {
public:
    JsonParser(const char *data, size_t size) : data_(data), size_(size) {}

    Json parse() {
        Json value = parse_value();
        whitespace();
        if (position_ != size_)
            fail("trailing content");
        return value;
    }

private:
    [[noreturn]] void fail(const char *detail) const {
        throw std::runtime_error(std::string("safetensors: malformed JSON: ") +
                                 detail);
    }

    void whitespace() {
        while (position_ < size_ &&
               (data_[position_] == ' ' || data_[position_] == '\n' ||
                data_[position_] == '\r' || data_[position_] == '\t'))
            position_++;
    }

    char take() {
        if (position_ == size_)
            fail("unexpected end");
        return data_[position_++];
    }

    Json parse_value() {
        whitespace();
        if (position_ == size_)
            fail("missing value");
        if (data_[position_] == '"') {
            Json value;
            value.kind = Json::Kind::string;
            value.string = parse_string();
            return value;
        }
        if (data_[position_] == '{')
            return parse_object();
        if (data_[position_] == '[')
            return parse_array();
        if (data_[position_] >= '0' && data_[position_] <= '9')
            return parse_number();
        fail("unsupported value");
    }

    std::string parse_string() {
        if (take() != '"')
            fail("expected string");
        std::string result;
        while (position_ < size_) {
            const char value = take();
            if (value == '"')
                return result;
            if (static_cast<unsigned char>(value) < 0x20)
                fail("control character in string");
            if (value != '\\') {
                result.push_back(value);
                continue;
            }
            const char escape = take();
            switch (escape) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            default: fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }

    Json parse_number() {
        Json value;
        value.kind = Json::Kind::number;
        while (position_ < size_ && data_[position_] >= '0' &&
               data_[position_] <= '9') {
            const unsigned digit = unsigned(data_[position_] - '0');
            if (value.number >
                (std::numeric_limits<uint64_t>::max() - digit) / 10)
                fail("integer overflow");
            value.number = value.number * 10 + digit;
            position_++;
        }
        return value;
    }

    Json parse_array() {
        Json value;
        value.kind = Json::Kind::array;
        take();
        whitespace();
        if (position_ < size_ && data_[position_] == ']') {
            position_++;
            return value;
        }
        for (;;) {
            value.array.push_back(parse_value());
            whitespace();
            const char delimiter = take();
            if (delimiter == ']')
                return value;
            if (delimiter != ',')
                fail("expected array delimiter");
        }
    }

    Json parse_object() {
        Json value;
        value.kind = Json::Kind::object;
        take();
        whitespace();
        if (position_ < size_ && data_[position_] == '}') {
            position_++;
            return value;
        }
        for (;;) {
            whitespace();
            if (position_ == size_ || data_[position_] != '"')
                fail("expected object key");
            std::string key = parse_string();
            whitespace();
            if (take() != ':')
                fail("expected colon");
            if (!value.object.emplace(std::move(key), parse_value()).second)
                fail("duplicate object key");
            whitespace();
            const char delimiter = take();
            if (delimiter == '}')
                return value;
            if (delimiter != ',')
                fail("expected object delimiter");
        }
    }

    const char *data_;
    size_t size_;
    size_t position_ = 0;
};

const Json &member(const Json &object, const char *name, Json::Kind kind) {
    if (object.kind != Json::Kind::object)
        throw std::runtime_error("safetensors: expected object");
    const auto found = object.object.find(name);
    if (found == object.object.end() || found->second.kind != kind)
        throw std::runtime_error(std::string("safetensors: missing or invalid ") +
                                 name);
    return found->second;
}

size_t checked_size(uint64_t value, const char *label) {
    if (value > std::numeric_limits<size_t>::max())
        throw std::runtime_error(std::string("safetensors: ") + label +
                                 " exceeds address space");
    return size_t(value);
}

size_t element_count(const std::vector<size_t> &shape) {
    if (shape.empty())
        throw std::runtime_error("safetensors: scalar tensors unsupported");
    size_t count = 1;
    for (size_t dimension : shape) {
        if (dimension == 0 ||
            count > std::numeric_limits<size_t>::max() / dimension)
            throw std::runtime_error("safetensors: invalid tensor shape");
        count *= dimension;
    }
    return count;
}

size_t parse_decimal(const std::string &text, const std::string &name) {
    if (text.empty())
        throw std::runtime_error("model metadata: empty " + name);
    size_t result = 0;
    for (char value : text) {
        if (value < '0' || value > '9')
            throw std::runtime_error("model metadata: invalid " + name);
        const size_t digit = size_t(value - '0');
        if (result > (std::numeric_limits<size_t>::max() - digit) / 10)
            throw std::runtime_error("model metadata: overflow in " + name);
        result = result * 10 + digit;
    }
    return result;
}

std::string layer_name(size_t layer, const char *suffix) {
    return "model.layers." + std::to_string(layer) + suffix;
}

void copy_vector(const TensorView &source, std::vector<float> &destination,
                 const std::string &name) {
    if (source.shape != std::vector<size_t>{destination.size()})
        throw std::runtime_error("model: wrong shape for " + name);
    std::memcpy(destination.data(), source.data,
                destination.size() * sizeof(float));
}

void transpose_weight(const TensorView &source, Matrix &destination,
                      const std::string &name) {
    const std::vector<size_t> expected{destination.cols(), destination.rows()};
    if (source.shape != expected)
        throw std::runtime_error("model: wrong shape for " + name);
    for (size_t external_row = 0; external_row < source.shape[0];
         external_row++) {
        for (size_t external_column = 0;
             external_column < source.shape[1]; external_column++) {
            destination.at(external_column, external_row) =
                source.data[external_row * source.shape[1] + external_column];
        }
    }
}

void copy_matrix(const TensorView &source, Matrix &destination,
                 const std::string &name) {
    if (source.shape !=
        std::vector<size_t>{destination.rows(), destination.cols()})
        throw std::runtime_error("model: wrong shape for " + name);
    std::memcpy(destination.data(), source.data,
                destination.size() * sizeof(float));
}

} // namespace

struct SafeTensorFile::Storage {
#ifndef _WIN32
    int descriptor = -1;
    void *mapping = MAP_FAILED;
#else
    std::vector<unsigned char> owned;
#endif
    const unsigned char *data = nullptr;
    size_t size = 0;

    ~Storage() {
#ifndef _WIN32
        if (mapping != MAP_FAILED)
            munmap(mapping, size);
        if (descriptor >= 0)
            close(descriptor);
#endif
    }
};

SafeTensorFile::SafeTensorFile() = default;
SafeTensorFile::~SafeTensorFile() = default;
SafeTensorFile::SafeTensorFile(SafeTensorFile &&) noexcept = default;
SafeTensorFile &SafeTensorFile::operator=(SafeTensorFile &&) noexcept = default;

SafeTensorFile SafeTensorFile::load(const std::string &path) {
    SafeTensorFile file;
    file.storage_ = std::make_shared<Storage>();
#ifndef _WIN32
    file.storage_->descriptor = open(path.c_str(), O_RDONLY);
    if (file.storage_->descriptor < 0)
        throw std::runtime_error("safetensors: cannot open " + path + ": " +
                                 std::strerror(errno));
    struct stat status {};
    if (fstat(file.storage_->descriptor, &status) != 0 || status.st_size < 0)
        throw std::runtime_error("safetensors: cannot stat " + path);
    file.storage_->size = size_t(status.st_size);
    if (file.storage_->size == 0)
        throw std::runtime_error("safetensors: empty file");
    file.storage_->mapping =
        mmap(nullptr, file.storage_->size, PROT_READ, MAP_PRIVATE,
             file.storage_->descriptor, 0);
    if (file.storage_->mapping == MAP_FAILED)
        throw std::runtime_error("safetensors: mmap failed");
    file.storage_->data =
        static_cast<const unsigned char *>(file.storage_->mapping);
#else
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("safetensors: cannot open " + path);
    file.storage_->owned.assign(std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>());
    file.storage_->size = file.storage_->owned.size();
    file.storage_->data = file.storage_->owned.data();
#endif

    if (file.storage_->size < sizeof(uint64_t))
        throw std::runtime_error("safetensors: truncated length prefix");
    uint64_t header_length = 0;
    for (size_t byte = 0; byte < sizeof(uint64_t); byte++)
        header_length |= uint64_t(file.storage_->data[byte]) << (byte * 8);
    const size_t header_size = checked_size(header_length, "header");
    if (header_size == 0 ||
        header_size > file.storage_->size - sizeof(uint64_t))
        throw std::runtime_error("safetensors: truncated header");
    file.data_offset_ = sizeof(uint64_t) + header_size;

    const Json root =
        JsonParser(reinterpret_cast<const char *>(file.storage_->data + 8),
                   header_size).parse();
    if (root.kind != Json::Kind::object)
        throw std::runtime_error("safetensors: root must be an object");
    for (const auto &entry : root.object) {
        if (entry.first == "__metadata__") {
            if (entry.second.kind != Json::Kind::object)
                throw std::runtime_error("safetensors: invalid metadata");
            for (const auto &metadata : entry.second.object) {
                if (metadata.second.kind != Json::Kind::string)
                    throw std::runtime_error(
                        "safetensors: metadata values must be strings");
                file.metadata_.emplace(metadata.first, metadata.second.string);
            }
            continue;
        }
        const Json &dtype = member(entry.second, "dtype", Json::Kind::string);
        const Json &shape = member(entry.second, "shape", Json::Kind::array);
        const Json &offsets =
            member(entry.second, "data_offsets", Json::Kind::array);
        if (dtype.string != "F32")
            throw std::runtime_error("safetensors: only F32 is supported");
        Descriptor descriptor;
        for (const Json &dimension : shape.array) {
            if (dimension.kind != Json::Kind::number)
                throw std::runtime_error("safetensors: invalid shape");
            descriptor.shape.push_back(
                checked_size(dimension.number, "shape dimension"));
        }
        if (offsets.array.size() != 2 ||
            offsets.array[0].kind != Json::Kind::number ||
            offsets.array[1].kind != Json::Kind::number)
            throw std::runtime_error("safetensors: invalid data offsets");
        descriptor.begin =
            checked_size(offsets.array[0].number, "data offset");
        descriptor.end = checked_size(offsets.array[1].number, "data offset");
        const size_t elements = element_count(descriptor.shape);
        if (elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
            descriptor.begin > descriptor.end ||
            descriptor.end - descriptor.begin != elements * sizeof(float) ||
            descriptor.end > file.storage_->size - file.data_offset_ ||
            (file.data_offset_ + descriptor.begin) % alignof(float) != 0)
            throw std::runtime_error("safetensors: invalid tensor extent");
        file.tensors_.emplace(entry.first, std::move(descriptor));
    }
    if (file.tensors_.empty())
        throw std::runtime_error("safetensors: no tensors");
    return file;
}

TensorView SafeTensorFile::tensor(const std::string &name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end())
        throw std::runtime_error("model: missing tensor " + name);
    const Descriptor &descriptor = found->second;
    return {reinterpret_cast<const float *>(
                storage_->data + data_offset_ + descriptor.begin),
            descriptor.shape, element_count(descriptor.shape)};
}

const std::string &SafeTensorFile::metadata(const std::string &name) const {
    const auto found = metadata_.find(name);
    if (found == metadata_.end())
        throw std::runtime_error("model metadata: missing " + name);
    return found->second;
}

ModelWeights::ModelWeights(const ModelConfig &model_config)
    : config(model_config),
      embeddings(config.vocab_size, config.dim),
      final_norm(config.dim, 1.0f),
      lm_head(config.vocab_size, config.dim) {
    config.validate();
    layers.reserve(config.layers);
    for (size_t layer = 0; layer < config.layers; layer++)
        layers.emplace_back(config);
}

ModelWeights load_model(const std::string &path) {
    SafeTensorFile file = SafeTensorFile::load(path);
    ModelConfig config{
        parse_decimal(file.metadata("nutllm.layers"), "layers"),
        parse_decimal(file.metadata("nutllm.heads"), "heads"),
        parse_decimal(file.metadata("nutllm.dim"), "dim"),
        parse_decimal(file.metadata("nutllm.head_dim"), "head_dim"),
        parse_decimal(file.metadata("nutllm.ffn_dim"), "ffn_dim"),
        parse_decimal(file.metadata("nutllm.vocab_size"), "vocab_size"),
        parse_decimal(file.metadata("nutllm.max_seq"), "max_seq"),
        parse_decimal(file.metadata("nutllm.eos_token"), "eos_token")};
    config.validate();
    ModelWeights model(config);

    copy_matrix(file.tensor("model.embed_tokens.weight"), model.embeddings,
                "model.embed_tokens.weight");
    for (size_t layer = 0; layer < config.layers; layer++) {
        LayerWeights &weights = model.layers[layer];
        const std::string prefix = "model.layers." + std::to_string(layer);
        transpose_weight(
            file.tensor(layer_name(
                layer, ".self_attn.q_proj.weight")),
            weights.query, prefix + ".self_attn.q_proj.weight");
        transpose_weight(
            file.tensor(layer_name(
                layer, ".self_attn.k_proj.weight")),
            weights.key, prefix + ".self_attn.k_proj.weight");
        transpose_weight(
            file.tensor(layer_name(
                layer, ".self_attn.v_proj.weight")),
            weights.value, prefix + ".self_attn.v_proj.weight");
        transpose_weight(
            file.tensor(layer_name(
                layer, ".self_attn.o_proj.weight")),
            weights.output, prefix + ".self_attn.o_proj.weight");
        transpose_weight(
            file.tensor(layer_name(layer, ".mlp.gate_proj.weight")),
            weights.gate, prefix + ".mlp.gate_proj.weight");
        transpose_weight(
            file.tensor(layer_name(layer, ".mlp.up_proj.weight")),
            weights.up, prefix + ".mlp.up_proj.weight");
        transpose_weight(
            file.tensor(layer_name(layer, ".mlp.down_proj.weight")),
            weights.down, prefix + ".mlp.down_proj.weight");
        copy_vector(
            file.tensor(layer_name(layer, ".input_layernorm.weight")),
            weights.attention_norm, prefix + ".input_layernorm.weight");
        copy_vector(
            file.tensor(
                layer_name(layer, ".post_attention_layernorm.weight")),
            weights.ffn_norm,
            prefix + ".post_attention_layernorm.weight");
    }
    copy_vector(file.tensor("model.norm.weight"), model.final_norm,
                "model.norm.weight");
    copy_matrix(file.tensor("lm_head.weight"), model.lm_head,
                "lm_head.weight");
    return model;
}
