#pragma once

#include <functional>
#include <string>
#include <vector>

namespace serius {
namespace data {

/// Synthetic data sample: instruction + output pair.
struct SyntheticSample
{
    std::string instruction;
    std::string output;
    float quality = 0.0f;
};

/// Quality judge interface: returns a score in [0, 1] for an instruction+output pair.
class QualityJudge
{
   public:
    virtual ~QualityJudge() = default;
    virtual float score(const std::string &instruction, const std::string &output) = 0;
};

/// Simple heuristic quality judge: longer, well-formed outputs score higher.
class HeuristicJudge : public QualityJudge
{
   public:
    float score(const std::string &instruction, const std::string &output) override;
};

/// Generate a synthetic dataset using a teacher model and a quality judge.
/// The teacher is expected to provide: generate_instruction() and generate_output().
class SyntheticGenerator
{
   public:
    using InstructionGen = std::function<std::string()>;
    using OutputGen = std::function<std::string(const std::string &)>;

    SyntheticGenerator(InstructionGen gen_instruction,
                       OutputGen gen_output,
                       QualityJudge &judge,
                       float quality_threshold = 0.7f);

    /// Generate up to n_samples accepted samples.
    std::vector<SyntheticSample> generate(int n_samples);

   private:
    InstructionGen gen_instruction_;
    OutputGen gen_output_;
    QualityJudge &judge_;
    float quality_threshold_;
};

}  // namespace data
}  // namespace serius
