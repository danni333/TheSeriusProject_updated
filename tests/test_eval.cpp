#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "eval/eval_suite.h"
#include "eval/metrics.h"
#include "eval/perplexity.h"
#include "eval/perplexity_runner.h"
#include "eval/report.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"

using serius::eval::EvalResult;
using serius::eval::eval_multiple_choice;
using serius::eval::perplexity_from_logprobs;
using serius::eval::uniform_perplexity;
using serius::eval::TrainingMetrics;
using serius::eval::save_json_report;
using serius::eval::save_csv_report;

TEST(Eval, MultipleChoicePerfect)
{
    std::vector<int> pred = {0, 1, 2, 3};
    std::vector<int> truth = {0, 1, 2, 3};
    const auto r = eval_multiple_choice("MMLU", pred, truth);
    EXPECT_EQ(r.benchmark, "MMLU");
    EXPECT_EQ(r.correct, 4);
    EXPECT_EQ(r.total, 4);
    EXPECT_FLOAT_EQ(r.accuracy(), 1.0f);
}

TEST(Eval, MultipleChoicePartial)
{
    std::vector<int> pred = {0, 1, 2, 3};
    std::vector<int> truth = {0, 2, 2, 1};
    const auto r = eval_multiple_choice("MMLU", pred, truth);
    EXPECT_EQ(r.correct, 2);
    EXPECT_EQ(r.total, 4);
    EXPECT_FLOAT_EQ(r.accuracy(), 0.5f);
}

TEST(Eval, MultipleChoiceMismatchedLengths)
{
    std::vector<int> pred = {0, 1, 2};
    std::vector<int> truth = {0, 1};
    const auto r = eval_multiple_choice("MMLU", pred, truth);
    EXPECT_EQ(r.total, 2);
    EXPECT_EQ(r.correct, 2);
}

TEST(Eval, PerplexityUniform)
{
    const size_t vocab_size = 1000;
    const float log_p = -std::log(static_cast<float>(vocab_size));
    std::vector<float> log_probs(50, log_p);
    const float ppl = perplexity_from_logprobs(log_probs);
    EXPECT_NEAR(ppl, static_cast<float>(vocab_size), 1e-3f);
}

TEST(Eval, PerplexityEqualsUniformPerplexity)
{
    const size_t vocab_size = 500;
    std::vector<float> log_probs(10, -std::log(static_cast<float>(vocab_size)));
    const float ppl = perplexity_from_logprobs(log_probs);
    EXPECT_NEAR(ppl, uniform_perplexity(vocab_size), 1e-3f);
}

TEST(Eval, PerplexityEmpty)
{
    EXPECT_FLOAT_EQ(perplexity_from_logprobs({}), 1.0f);
}

TEST(Eval, OverfittingDetection)
{
    TrainingMetrics m;
    for (int i = 0; i < 200; ++i) {
        m.steps.push_back(i);
        m.train_loss.push_back(2.0f - static_cast<float>(i) * 0.02f);  // decreasing fast
        m.val_loss.push_back(1.5f + static_cast<float>(i) * 0.02f);   // increasing fast
    }
    EXPECT_TRUE(m.is_overfitting(100));
}

TEST(Eval, NoOverfitting)
{
    TrainingMetrics m;
    for (int i = 0; i < 200; ++i) {
        m.steps.push_back(i);
        m.train_loss.push_back(2.0f - static_cast<float>(i) * 0.005f);
        m.val_loss.push_back(2.0f - static_cast<float>(i) * 0.004f);  // also decreasing
    }
    EXPECT_FALSE(m.is_overfitting(100));
}

TEST(Eval, OverfittingTooShort)
{
    TrainingMetrics m;
    m.steps = {0, 1, 2};
    m.train_loss = {2.0f, 1.0f, 0.5f};
    m.val_loss = {2.0f, 3.0f, 4.0f};
    EXPECT_FALSE(m.is_overfitting(100));
}

TEST(Eval, ExportCsv)
{
    TrainingMetrics m;
    m.steps = {0, 1, 2};
    m.train_loss = {2.0f, 1.0f, 0.5f};
    m.val_loss = {2.0f, 2.5f, 3.0f};
    const std::string path = "test_eval_metrics.csv";
    m.export_csv(path);
    std::ifstream f(path);
    ASSERT_TRUE(f);
    std::string line;
    std::getline(f, line);
    EXPECT_EQ(line, "step,train_loss,val_loss");
    std::getline(f, line);
    EXPECT_EQ(line, "0,2,2");
    f.close();
    std::remove(path.c_str());
}

TEST(Eval, SaveJsonReport)
{
    std::vector<EvalResult> results = {
        {"MMLU", 80, 100},
        {"GSM8K", 45, 50},
    };
    const std::string path = "test_eval_report.json";
    save_json_report(results, path);
    std::ifstream f(path);
    ASSERT_TRUE(f);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    EXPECT_NE(content.find("MMLU"), std::string::npos);
    EXPECT_NE(content.find("GSM8K"), std::string::npos);
    EXPECT_NE(content.find("0.8"), std::string::npos);
    EXPECT_NE(content.find("0.9"), std::string::npos);
    std::remove(path.c_str());
}

// ---- PerplexityRunner: real model scoring ----

namespace {

serius::model::LLaMAModel make_tiny_model()
{
    serius::model::LLamaConfig cfg;
    cfg.vocab_size = 320;  // >= basic tokenizer vocab (260)
    cfg.hidden_size = 64;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 128;
    cfg.max_position_embeddings = 128;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    return serius::model::LLaMAModel(cfg);
}

serius::BPETokenizer make_basic_tokenizer()
{
    serius::BPETokenizer tok;
    tok.init_basic_vocab();
    return tok;
}

}  // namespace

TEST(Eval, PerplexityRunnerScoresRealTokens)
{
    auto model = make_tiny_model();
    auto tok = make_basic_tokenizer();

    const std::string text = "hello world this is a test of perplexity scoring";
    const auto r = serius::eval::evaluate_perplexity(model, tok, text);

    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_GT(r.total_tokens, 2u);
    // Every token except the very first stream token is scored.
    EXPECT_EQ(r.scored_tokens, r.total_tokens - 1);
    EXPECT_EQ(r.chunks, 1u);
    EXPECT_TRUE(std::isfinite(r.mean_nll));
    EXPECT_GT(r.perplexity, 0.0);
    EXPECT_GT(r.elapsed_seconds, 0.0);
}

TEST(Eval, PerplexityRunnerDeterministic)
{
    auto model = make_tiny_model();
    auto tok = make_basic_tokenizer();
    const std::string text = "deterministic scoring check";

    const auto a = serius::eval::evaluate_perplexity(model, tok, text);
    const auto b = serius::eval::evaluate_perplexity(model, tok, text);
    ASSERT_TRUE(a.ok && b.ok);
    EXPECT_DOUBLE_EQ(a.mean_nll, b.mean_nll);
    EXPECT_DOUBLE_EQ(a.perplexity, b.perplexity);
}

TEST(Eval, PerplexityRunnerEmptyText)
{
    auto model = make_tiny_model();
    auto tok = make_basic_tokenizer();
    const auto r = serius::eval::evaluate_perplexity(model, tok, "");
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.scored_tokens, 0u);
}

TEST(Eval, PerplexityRunnerMaxTokens)
{
    auto model = make_tiny_model();
    auto tok = make_basic_tokenizer();
    const std::string text = "one two three four five six seven eight nine ten";

    serius::eval::PerplexityOptions opts;
    opts.max_tokens = 3;
    const auto r = serius::eval::evaluate_perplexity(model, tok, text, opts);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.scored_tokens, 3u);
}

TEST(Eval, PerplexityRunnerMultiChunkCarryOver)
{
    auto model = make_tiny_model();
    auto tok = make_basic_tokenizer();
    // Long enough text to span several tiny context windows.
    const std::string text =
        "the quick brown fox jumps over the lazy dog and runs through the forest "
        "while the wind blows softly across the riverbank near the old stone bridge";

    serius::eval::PerplexityOptions opts;
    opts.context_tokens = 8;
    const auto r = serius::eval::evaluate_perplexity(model, tok, text, opts);
    ASSERT_TRUE(r.ok) << r.error;
    EXPECT_GT(r.chunks, 1u);
    // Carry-over context keeps continuity: still every token but the first scored.
    EXPECT_EQ(r.scored_tokens, r.total_tokens - 1);
    EXPECT_TRUE(std::isfinite(r.mean_nll));
    EXPECT_GT(r.perplexity, 0.0);
}

TEST(Eval, SaveCsvReport)
{
    std::vector<EvalResult> results = {
        {"MMLU", 80, 100},
    };
    const std::string path = "test_eval_report.csv";
    save_csv_report(results, path);
    std::ifstream f(path);
    ASSERT_TRUE(f);
    std::string line;
    std::getline(f, line);
    EXPECT_EQ(line, "benchmark,correct,total,accuracy");
    std::getline(f, line);
    EXPECT_NE(line.find("MMLU"), std::string::npos);
    EXPECT_NE(line.find("0.8"), std::string::npos);
    f.close();
    std::remove(path.c_str());
}
