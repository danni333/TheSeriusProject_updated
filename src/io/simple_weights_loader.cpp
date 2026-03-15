#include "simple_weights_loader.h"

#include <fstream>

#include "../transformer/model.h"

namespace serius {
namespace io {

namespace {

constexpr uint32_t SIMPLE_WEIGHTS_MAGIC = 0x57524553;  // "SERW" little-endian

}  // namespace

bool save_simple_weights(const std::string &path, const model::TransformerModel &model)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(&SIMPLE_WEIGHTS_MAGIC), sizeof(SIMPLE_WEIGHTS_MAGIC));
    if (!f) {
        return false;
    }
    std::vector<core::autograd::Node *> params = model.parameters();
    for (core::autograd::Node *p : params) {
        const core::Tensor &t = p->data;
        size_t n = t.size();
        const float *ptr = t.data();
        f.write(reinterpret_cast<const char *>(ptr), static_cast<std::streamsize>(n * sizeof(float)));
        if (!f) {
            return false;
        }
    }
    return true;
}

bool load_simple_weights(const std::string &path, model::TransformerModel *model)
{
    if (!model) {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    uint32_t magic = 0;
    f.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (!f || magic != SIMPLE_WEIGHTS_MAGIC) {
        return false;
    }
    std::vector<core::autograd::Node *> params = model->parameters();
    for (core::autograd::Node *p : params) {
        core::Tensor &t = p->data;
        size_t n = t.size();
        float *ptr = t.data();
        f.read(reinterpret_cast<char *>(ptr), static_cast<std::streamsize>(n * sizeof(float)));
        if (!f || f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
            return false;
        }
    }
    return true;
}

}  // namespace io
}  // namespace serius
