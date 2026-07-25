#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model.hpp"

struct TensorView {
    const float *data = nullptr;
    std::vector<size_t> shape;
    size_t elements = 0;
};

class SafeTensorFile {
public:
    static SafeTensorFile load(const std::string &path);

    SafeTensorFile();
    ~SafeTensorFile();
    SafeTensorFile(SafeTensorFile &&) noexcept;
    SafeTensorFile &operator=(SafeTensorFile &&) noexcept;
    SafeTensorFile(const SafeTensorFile &) = delete;
    SafeTensorFile &operator=(const SafeTensorFile &) = delete;

    TensorView tensor(const std::string &name) const;
    const std::string &metadata(const std::string &name) const;

private:
    struct Storage;
    struct Descriptor {
        std::vector<size_t> shape;
        size_t begin = 0;
        size_t end = 0;
    };

    std::shared_ptr<Storage> storage_;
    std::map<std::string, Descriptor> tensors_;
    std::map<std::string, std::string> metadata_;
    size_t data_offset_ = 0;

};

struct ModelWeights {
    ModelConfig config;
    Matrix embeddings;
    std::vector<LayerWeights> layers;
    std::vector<float> final_norm;
    Matrix lm_head;

    explicit ModelWeights(const ModelConfig &model_config);
};

ModelWeights load_model(const std::string &path);
