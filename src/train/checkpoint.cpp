#include "checkpoint.h"

#include <fstream>
#include <limits>
#include <map>

#include "../stability/grad_clip.h"

namespace serius {
namespace training {

void clip_grad_norm(const std::vector<core::autograd::Node *> &params, float max_norm)
{
    stability::clip_grad_norm(params, max_norm);
}

void save_checkpoint(const std::string &path, const model::TransformerModel &model, const optim::AdamW &optimizer,
                     int epoch, int step)
{
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("Failed to open checkpoint for writing: " + path);

    const char magic[4] = {'T', 'S', 'C', 'P'};
    os.write(magic, 4);
    int version = 1;
    os.write(reinterpret_cast<char *>(&version), 4);
    os.write(reinterpret_cast<char *>(&epoch), 4);
    os.write(reinterpret_cast<char *>(&step), 4);

    auto params = model.parameters();
    int num_params = static_cast<int>(params.size());
    os.write(reinterpret_cast<char *>(&num_params), 4);

    for (auto *p : params) {
        int name_len = static_cast<int>(p->name.length());
        os.write(reinterpret_cast<char *>(&name_len), 4);
        os.write(p->name.c_str(), name_len);

        int ndim = static_cast<int>(p->data.ndim());
        os.write(reinterpret_cast<char *>(&ndim), 4);
        for (size_t d : p->data.shape()) {
            size_t val = d;
            os.write(reinterpret_cast<char *>(&val), sizeof(size_t));
        }

        os.write(reinterpret_cast<char *>(p->data.data()), p->data.size() * sizeof(float));

        // A parameter may have no optimizer state yet (e.g. frozen or never
        // stepped). Write zero moments + step 0 instead of throwing via .at(),
        // which would abort the save mid-file and leave a truncated checkpoint.
        auto state_it = optimizer.state().find(p);
        if (state_it == optimizer.state().end()) {
            const size_t n = p->data.size();
            core::Tensor zeros({n});
            zeros.zeros();
            os.write(reinterpret_cast<const char *>(zeros.data()), n * sizeof(float));
            os.write(reinterpret_cast<const char *>(zeros.data()), n * sizeof(float));
            int zero_step = 0;
            os.write(reinterpret_cast<const char *>(&zero_step), 4);
        } else {
            auto &state = state_it->second;
            os.write(reinterpret_cast<const char *>(state.m.data()), state.m.size() * sizeof(float));
            os.write(reinterpret_cast<const char *>(state.v.data()), state.v.size() * sizeof(float));
            os.write(reinterpret_cast<const char *>(&state.step), 4);
        }
    }
}

void load_checkpoint(const std::string &path, model::TransformerModel &model, optim::AdamW &optimizer, int &epoch,
                     int &step)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Failed to open checkpoint for reading: " + path);

    auto fail = [&]() -> void {
        throw std::runtime_error("Truncated or corrupt checkpoint: " + path);
    };

    char magic[4];
    is.read(magic, 4);
    if (!is || magic[0] != 'T' || magic[1] != 'S' || magic[2] != 'C' || magic[3] != 'P') {
        throw std::runtime_error("Invalid checkpoint magic number");
    }

    int version;
    is.read(reinterpret_cast<char *>(&version), 4);
    is.read(reinterpret_cast<char *>(&epoch), 4);
    is.read(reinterpret_cast<char *>(&step), 4);
    if (!is || version != 1) fail();

    int num_params;
    is.read(reinterpret_cast<char *>(&num_params), 4);
    if (!is || num_params < 0 || num_params > 1000000) fail();

    std::map<std::string, core::autograd::Node *> param_map;
    for (auto *p : model.parameters()) {
        param_map[p->name] = p;
    }

    for (int i = 0; i < num_params; ++i) {
        int name_len;
        is.read(reinterpret_cast<char *>(&name_len), 4);
        if (!is || name_len < 0 || name_len > (1 << 20)) fail();
        std::string name(name_len, ' ');
        is.read(name.data(), name_len);

        int ndim;
        is.read(reinterpret_cast<char *>(&ndim), 4);
        if (!is || ndim < 0 || ndim > 8) fail();
        std::vector<size_t> shape(ndim);
        size_t total_size = 1;
        for (int d = 0; d < ndim; ++d) {
            is.read(reinterpret_cast<char *>(&shape[d]), sizeof(size_t));
            if (!is || shape[d] == 0 || shape[d] > (size_t(1) << 32)) fail();
            if (total_size > std::numeric_limits<size_t>::max() / shape[d]) fail();
            total_size *= shape[d];
        }

        core::autograd::Node *p = nullptr;
        if (param_map.count(name)) {
            p = param_map[name];
        }

        if (p) {
            // The serialized size must match the parameter's actual storage;
            // otherwise the read would overflow the parameter buffer.
            if (total_size != p->data.size() || !p->data.is_contiguous()) fail();
            // A param may legitimately have no optimizer state (saved before
            // the first step, or a frozen param): materialize zero moments so
            // the read below has somewhere valid to land. A present-but-wrong
            // size is corruption and must fail, not be silently overwritten.
            auto state_it = optimizer.state().find(p);
            if (state_it == optimizer.state().end()) {
                core::Tensor m({total_size}), v({total_size});
                m.zeros();
                v.zeros();
                state_it = optimizer.state().emplace(p, optim::ParamState(std::move(m), std::move(v), 0)).first;
            }
            auto &state = state_it->second;
            if (state.m.size() != total_size || state.v.size() != total_size) fail();

            p->data.syncToHost();
            is.read(reinterpret_cast<char *>(p->data.data()), total_size * sizeof(float));
            is.read(reinterpret_cast<char *>(state.m.data()), total_size * sizeof(float));
            is.read(reinterpret_cast<char *>(state.v.data()), total_size * sizeof(float));
            is.read(reinterpret_cast<char *>(&state.step), 4);
            if (!is) fail();
            if (p->data.is_gpu()) p->data.mark_dirty(core::Tensor::DataLocation::CPU);
            if (state.m.is_gpu()) state.m.mark_dirty(core::Tensor::DataLocation::CPU);
            if (state.v.is_gpu()) state.v.mark_dirty(core::Tensor::DataLocation::CPU);
        } else {
            is.seekg(total_size * sizeof(float) * 3 + 4, std::ios::cur);
            if (!is) fail();
        }
    }
}

}  // namespace training
}  // namespace serius
