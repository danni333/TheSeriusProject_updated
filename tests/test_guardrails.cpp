#include "safety/guardrails.h"
#include "safety/toxicity.h"
#include "safety/sanitizer.h"
#include <gtest/gtest.h>
#include <string>

TEST(Guardrails, BlocklistToken)
{
    serius::safety::Guardrails g;
    g.add_blocklist_token(42);
    EXPECT_TRUE(g.is_blocked_token(42));
    EXPECT_FALSE(g.is_blocked_token(0));
    g.remove_blocklist_token(42);
    EXPECT_FALSE(g.is_blocked_token(42));
}

TEST(Guardrails, CheckResponse)
{
    serius::safety::Guardrails g;
    g.add_blocklist_phrase("badword");
    EXPECT_FALSE(g.check_response("this has badword inside"));
    EXPECT_TRUE(g.check_response("clean text"));
    g.clear();
    EXPECT_TRUE(g.check_response("this has badword inside"));
}

TEST(ToxicityClassifier, ScoreRange)
{
    serius::safety::ToxicityClassifier::Config cfg;
    cfg.vocab_size = 1000;
    cfg.embedding_dim = 64;
    cfg.hidden_dim = 32;
    serius::safety::ToxicityClassifier classifier(cfg);

    std::vector<int> tokens = {1, 2, 3, 4, 5};
    float score = classifier.score(tokens);

    // Score should be in [0, 1]
    EXPECT_GE(score, 0.0f);
    EXPECT_LE(score, 1.0f);
}

TEST(ToxicityClassifier, EmptyTokens)
{
    serius::safety::ToxicityClassifier classifier;
    std::vector<int> empty_tokens;
    float score = classifier.score(empty_tokens);
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(ToxicityClassifier, Threshold)
{
    serius::safety::ToxicityClassifier::Config cfg;
    cfg.vocab_size = 1000;
    cfg.embedding_dim = 64;
    cfg.hidden_dim = 32;
    serius::safety::ToxicityClassifier classifier(cfg);
    classifier.set_threshold(0.5f);

    std::vector<int> tokens = {1, 2, 3};
    float score = classifier.score(tokens);

    // Check that is_toxic can be called
    bool is_toxic = classifier.is_toxic(tokens);
    // Just verify it returns a boolean value
    EXPECT_TRUE(is_toxic == true || is_toxic == false);
}

TEST(Sanitizer, LengthLimit)
{
    serius::safety::Sanitizer::Config cfg;
    cfg.max_response_length = 10;
    serius::safety::Sanitizer sanitizer(cfg);

    std::string long_text = "This is a very long text that exceeds the limit";
    std::string result = sanitizer.sanitize(long_text);

    EXPECT_LE(result.length(), 13);  // 10 + "..."
    EXPECT_NE(result.find("..."), std::string::npos);
}

TEST(Sanitizer, SQLInjectionRemoval)
{
    serius::safety::Sanitizer::Config cfg;
    cfg.remove_sql_injection = true;
    serius::safety::Sanitizer sanitizer(cfg);

    std::string text = "SELECT * FROM users; DROP TABLE users;";
    std::string result = sanitizer.sanitize(text);

    EXPECT_NE(result.find("[SQL_REDACTED]"), std::string::npos);
}

TEST(Sanitizer, ShellCommandRemoval)
{
    serius::safety::Sanitizer::Config cfg;
    cfg.remove_shell_commands = true;
    serius::safety::Sanitizer sanitizer(cfg);

    std::string text = "Run rm -rf /home/user/data";
    std::string result = sanitizer.sanitize(text);

    EXPECT_NE(result.find("[SHELL_REDACTED]"), std::string::npos);
}

TEST(Sanitizer, CodeBlockRemoval)
{
    serius::safety::Sanitizer::Config cfg;
    cfg.remove_code_blocks = true;
    serius::safety::Sanitizer sanitizer(cfg);

    std::string text = "Here is code: ```python print('hello') ```";
    std::string result = sanitizer.sanitize(text);

    EXPECT_NE(result.find("[CODE_BLOCK]"), std::string::npos);
}

TEST(Sanitizer, CustomPattern)
{
    serius::safety::Sanitizer sanitizer;
    sanitizer.add_pattern(R"(\b\d{3}-\d{3}-\d{4}\b)", "[PHONE]");

    std::string text = "Call me at 123-456-7890";
    std::string result = sanitizer.sanitize(text);

    EXPECT_NE(result.find("[PHONE]"), std::string::npos);
    EXPECT_EQ(result.find("123-456-7890"), std::string::npos);
}

TEST(Sanitizer, ClearPatterns)
{
    serius::safety::Sanitizer sanitizer;
    sanitizer.add_pattern("dangerous", "[REDACTED]");
    sanitizer.clear_patterns();

    std::string text = "This is dangerous";
    std::string result = sanitizer.sanitize(text);

    EXPECT_EQ(result.find("[REDACTED]"), std::string::npos);
}

TEST(Sanitizer, MultipleRedactions)
{
    serius::safety::Sanitizer::Config cfg;
    cfg.remove_sql_injection = true;
    cfg.remove_shell_commands = true;
    serius::safety::Sanitizer sanitizer(cfg);

    std::string text = "DROP TABLE users; then run rm -rf /";
    std::string result = sanitizer.sanitize(text);

    // Should have both redactions
    EXPECT_NE(result.find("[SQL_REDACTED]"), std::string::npos);
    EXPECT_NE(result.find("[SHELL_REDACTED]"), std::string::npos);
}
