#include "config_loader.h"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serius {
namespace io {

namespace {

model::ModelConfig parse_config(const nlohmann::json &j)
{
    model::ModelConfig c;
    c.vocab_size = j.value("vocab_size", c.vocab_size);
    c.d_model = j.value("hidden_size", c.d_model);
    c.n_heads = j.value("num_attention_heads", c.n_heads);
    c.n_layers = j.value("num_layers", c.n_layers);
    c.d_ff = j.value("intermediate_size", c.d_ff);
    c.max_seq_len = j.value("max_position_embeddings", c.max_seq_len);
    return c;
}

model::QwenConfig parse_qwen_config(const nlohmann::json &j)
{
    model::QwenConfig c;
    c.vocab_size = j.value("vocab_size", c.vocab_size);
    c.hidden_size = j.value("hidden_size", c.hidden_size);
    c.num_attention_heads = j.value("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_key_value_heads);
    c.num_hidden_layers = j.value("num_hidden_layers", c.num_hidden_layers);
    c.intermediate_size = j.value("intermediate_size", c.intermediate_size);
    c.max_position_embeddings = j.value("max_position_embeddings", c.max_position_embeddings);
    c.rms_norm_eps = j.value("rms_norm_eps", c.rms_norm_eps);
    c.rope_theta = j.value("rope_theta", c.rope_theta);
    return c;
}

}  // namespace

model::ModelConfig load_model_config(const std::string &path)
{
    std::ifstream f(path);
    if (!f) {
        return model::ModelConfig{};
    }
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        nlohmann::json j = nlohmann::json::parse(ss.str());
        return parse_config(j);
    } catch (const std::exception &) {
        return model::ModelConfig{};
    }
}

model::ModelConfig load_model_config_from_string(const std::string &json_str)
{
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);
        return parse_config(j);
} catch (const std::exception &) {
        return model::ModelConfig{};
    }
}

model::QwenConfig load_qwen_config(const std::string &path)
{
    std::ifstream f(path);
    if (!f) {
        return model::QwenConfig{};
    }
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        nlohmann::json j = nlohmann::json::parse(ss.str());
        return parse_qwen_config(j);
    } catch (const std::exception &) {
        return model::QwenConfig{};
    }
}

model::QwenConfig load_qwen_config_from_string(const std::string &json_str)
{
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);
        return parse_qwen_config(j);
    } catch (const std::exception &) {
        return model::QwenConfig{};
    }
}

}  // namespace io
}  // namespace serius
