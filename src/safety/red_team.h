#pragma once

#include <string>
#include <vector>

namespace serius {
namespace safety {

enum class AttackCategory
{
    Jailbreak,
    PromptInjection,
    DataExtraction,
    MaliciousCode
};

struct AttackResult
{
    std::string prompt;
    std::string response;
    AttackCategory category;
    float severity;   // 0-1
    bool success;     // true = attack succeeded (unsafe)
};

/// Simple keyword-based toxicity scorer (0 = safe, 1 = toxic).
class ToxicityClassifier
{
   public:
    float score(const std::string &text) const;
};

/// Target model interface for red teaming.
class RedTeamTarget
{
   public:
    virtual ~RedTeamTarget() = default;
    virtual std::string generate(const std::string &prompt, float temperature = 0.3f) = 0;
};

/// Adversarial testing suite (red teaming).
class RedTeamSuite
{
   public:
    RedTeamSuite(RedTeamTarget &target, ToxicityClassifier &toxicity);

    std::vector<AttackResult> run_all();

   private:
    RedTeamTarget &target_;
    ToxicityClassifier &toxicity_;

    AttackResult test_one(const std::string &prompt, AttackCategory category);

    static std::vector<std::string> jailbreak_templates();
    static std::vector<std::string> injection_templates();
    static std::vector<std::string> extraction_templates();
    static std::vector<std::string> malicious_code_templates();
};

}  // namespace safety
}  // namespace serius
