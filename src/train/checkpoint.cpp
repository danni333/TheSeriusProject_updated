#include "checkpoint.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

namespace serius {
namespace training {

void clip_grad_norm(const std::vector<autograd::Node *> &params, float max_norm)
{
    float total_norm = 0.0f;
    for (auto *p : params) {
        const Tensor &g = p->grad;
        for (size_t i = 0; i < g.size(); ++i) {
            total_norm += g[i] * g[i];
        }
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto *p : params) {
            Tensor &g = p->grad;
            for (size_t i = 0; i < g.size(); ++i) {
                g[i] *= scale;
            }
        }
    }
}

void save_checkpoint(const std::string &path, const transformer::TransformerModel &model, const optim::AdamW &optimizer,
                     int epoch, int step)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("Failed to open checkpoint for writing: " + path);

    const char magic[4] = {'T', 'S', 'C', 'P'};
    os.write(magic, 4);
    int version = 1;
    os.write((char *)&version, 4);
    os.write((char *)&epoch, 4);
    os.write((char *)&step, 4);

    auto params = model.parameters();
    int num_params = static_cast<int>(params.size());
    os.write((char *)&num_params, 4);

    for (auto *p : params) {
        int name_len = static_cast<int>(p->name.length());
        os.write((char *)&name_len, 4);
        os.write(p->name.c_str(), name_len);

        int ndim = static_cast<int>(p->data.ndim());
        os.write((char *)&ndim, 4);
        for (size_t d : p->data.shape()) {
            size_t val = d;
            os.write((char *)&val, sizeof(size_t));
        }

        os.write((char *)p->data.data(), p->data.size() * sizeof(float));

        auto &state = optimizer.state().at(p);
        os.write((char *)state.m.data(), state.m.size() * sizeof(float));
        os.write((char *)state.v.data(), state.v.size() * sizeof(float));
        os.write((char *)&state.step, 4);
    }
}

void load_checkpoint(const std::string &path, transformer::TransformerModel &model, optim::AdamW &optimizer, int &epoch,
                     int &step)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Failed to open checkpoint for reading: " + path);

    char magic[4];
    is.read(magic, 4);
    if (magic[0] != 'T' || magic[1] != 'S' || magic[2] != 'C' || magic[3] != 'P') {
        throw std::runtime_error("Invalid checkpoint magic number");
    }

    int version;
    is.read((char *)&version, 4);
    is.read((char *)&epoch, 4);
    is.read((char *)&step, 4);

    int num_params;
    is.read((char *)&num_params, 4);

    std::map<std::string, autograd::Node *> param_map;
    for (auto *p : model.parameters()) {
        param_map[p->name] = p;
    }

    for (int i = 0; i < num_params; ++i) {
        int name_len;
        is.read((char *)&name_len, 4);
        std::string name(name_len, ' ');
        is.read(&name[0], name_len);

        int ndim;
        is.read((char *)&ndim, 4);
        std::vector<size_t> shape(ndim);
        size_t total_size = 1;
        for (int d = 0; d < ndim; ++d) {
            is.read((char *)&shape[d], sizeof(size_t));
            total_size *= shape[d];
        }

        autograd::Node *p = nullptr;
        if (param_map.count(name)) {
            p = param_map[name];
        }

        if (p) {
            is.read((char *)p->data.data(), total_size * sizeof(float));
            auto &state = optimizer.state().at(p);
            is.read((char *)state.m.data(), total_size * sizeof(float));
            is.read((char *)state.v.data(), total_size * sizeof(float));
            is.read((char *)&state.step, 4);
        } else {
            is.seekg(total_size * sizeof(float) * 3 + 4, std::ios::cur);
        }
    }
}

}  // namespace training
}  // namespace serius
