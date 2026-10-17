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

    if (j.contains("moe_enabled")) {
        c.moe_enabled = j["moe_enabled"].get<bool>();
    }
    if (j.contains("num_experts")) {
        c.num_experts = j["num_experts"].get<size_t>();
    }
    if (j.contains("top_k")) {
        c.top_k = j["top_k"].get<size_t>();
    }
    if (j.contains("shared_experts")) {
        c.shared_experts = j["shared_experts"].get<size_t>();
    }
    if (j.contains("load_balance_alpha")) {
        c.load_balance_alpha = j["load_balance_alpha"].get<float>();
    }
    if (j.contains("load_balance")) {
        c.load_balance = j["load_balance"].get<bool>();
    }

    const auto moe_json = j.value("moe", nlohmann::json::object());
    if (!moe_json.is_null() && moe_json.is_object()) {
        if (moe_json.contains("enabled")) {
            c.moe.enabled = moe_json["enabled"].get<bool>();
        }
        if (moe_json.contains("num_experts")) {
            c.moe.num_experts = moe_json["num_experts"].get<size_t>();
        }
        if (moe_json.contains("top_k")) {
            c.moe.top_k = moe_json["top_k"].get<size_t>();
        }
        if (moe_json.contains("shared_experts")) {
            c.moe.shared_experts = moe_json["shared_experts"].get<size_t>();
        }
        if (moe_json.contains("load_balance_alpha")) {
            c.moe.load_balance_alpha = moe_json["load_balance_alpha"].get<float>();
        }
        if (moe_json.contains("load_balance")) {
            c.moe.load_balance = moe_json["load_balance"].get<bool>();
        }
    }

    if (moe_json.is_object() && moe_json.contains("enabled") && !j.contains("moe_enabled")) {
        c.moe_enabled = moe_json["enabled"].get<bool>();
    }

    if (moe_json.is_object() && moe_json.contains("num_experts") && !j.contains("num_experts")) {
        c.num_experts = moe_json["num_experts"].get<size_t>();
    }
    if (moe_json.is_object() && moe_json.contains("top_k") && !j.contains("top_k")) {
        c.top_k = moe_json["top_k"].get<size_t>();
    }
    if (moe_json.is_object() && moe_json.contains("shared_experts") && !j.contains("shared_experts")) {
        c.shared_experts = moe_json["shared_experts"].get<size_t>();
    }
    if (moe_json.is_object() && moe_json.contains("load_balance_alpha") && !j.contains("load_balance_alpha")) {
        c.load_balance_alpha = moe_json["load_balance_alpha"].get<float>();
    }
    if (moe_json.is_object() && moe_json.contains("load_balance") && !j.contains("load_balance")) {
        c.load_balance = moe_json["load_balance"].get<bool>();
    }

    c.moe.enabled = c.moe_enabled;
    c.moe.num_experts = c.num_experts;
    c.moe.top_k = c.top_k;
    c.moe.shared_experts = c.shared_experts;
    c.moe.load_balance_alpha = c.load_balance_alpha;
    c.moe.load_balance = c.load_balance;
    c.normalize_moe();
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
