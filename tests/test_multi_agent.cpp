#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "agent/consensus.h"
#include "agent/model_interface.h"
#include "agent/planner.h"
#include "agent/specialist.h"
#include "agent/supervisor.h"

using serius::agent::ConsensusResult;
using serius::agent::DelphiConsensus;
using serius::agent::Model;
using serius::agent::Planner;
using serius::agent::PlanResult;
using serius::agent::SamplingParams;
using serius::agent::Specialist;
using serius::agent::SubTask;
using serius::agent::Supervisor;

// ---- Mock model ----

class ScriptedModel : public Model
{
   public:
    void add_response(std::string response) { responses_.push_back(std::move(response)); }

    std::string generate(const std::string &, const SamplingParams &) override
    {
        if (index_ < responses_.size()) {
            return responses_[index_++];
        }
        return "[ANSWER] fallback.\n";
    }

    size_t calls() const { return index_; }

   private:
    std::vector<std::string> responses_;
    size_t index_ = 0;
};

// ---- Specialist tests ----

TEST(MultiAgent, SpecialistExecutes)
{
    ScriptedModel model;
    model.add_response("Translation: Bonjour le monde.");
    Specialist specialist("Translator", "translation", &model);
    std::string result = specialist.execute("Translate hello world");
    EXPECT_FALSE(result.empty());
}

TEST(MultiAgent, SpecialistCanHandle)
{
    ScriptedModel model;
    model.add_response("ok");
    Specialist specialist("Translator", "translation language", &model);
    EXPECT_TRUE(specialist.can_handle("translation"));
    EXPECT_TRUE(specialist.can_handle("language"));
    EXPECT_FALSE(specialist.can_handle("cooking recipe"));
}

// ---- Planner tests ----

TEST(MultiAgent, PlannerDecomposes)
{
    ScriptedModel model;
    model.add_response("- Research the topic\n- Write the summary\n- Review the draft");
    Planner planner(model);
    PlanResult result = planner.decompose("Write a report");
    EXPECT_EQ(result.tasks.size(), 3u);
    EXPECT_EQ(result.tasks[0].description, "Research the topic");
    EXPECT_EQ(result.tasks[1].description, "Write the summary");
}

TEST(MultiAgent, PlannerFallback)
{
    ScriptedModel model;
    model.add_response("Just do it");
    Planner planner(model);
    PlanResult result = planner.decompose("Do something simple");
    // Should have at least one task.
    EXPECT_GE(result.tasks.size(), 1u);
}

// ---- Supervisor tests ----

TEST(MultiAgent, SupervisorDirectAnswer)
{
    ScriptedModel model;
    model.add_response("Direct answer here.");
    Supervisor supervisor(model);
    std::string answer = supervisor.process("Simple question");
    EXPECT_NE(answer.find("Direct answer"), std::string::npos);
}

TEST(MultiAgent, SupervisorDelegates)
{
    ScriptedModel core;
    core.add_response(
        "[{\"agent\":\"Translator\",\"prompt\":\"Translate 'hello' to French\"}]");
    core.add_response("The translation is: Bonjour.");

    ScriptedModel spec;
    spec.add_response("Bonjour means hello in French.");

    Supervisor supervisor(core);
    supervisor.register_specialist(
        new Specialist("Translator", "translation", &spec));

    std::string answer = supervisor.process("Translate hello to French");
    EXPECT_FALSE(answer.empty());
}

// ---- Consensus tests ----

TEST(MultiAgent, ConsensusConverges)
{
    ScriptedModel a, b, c;
    a.add_response("The answer is 42.");
    b.add_response("The answer is 42.");
    c.add_response("The answer is 42.");

    // Second round responses (identical → converges).
    a.add_response("The answer is 42.");
    b.add_response("The answer is 42.");
    c.add_response("The answer is 42.");

    std::vector<Model *> agents = {&a, &b, &c};
    DelphiConsensus consensus(agents, 3);
    ConsensusResult result = consensus.debate("What is the answer?");
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.rounds, 1);
    EXPECT_NE(result.final_answer.find("42"), std::string::npos);
}

TEST(MultiAgent, ConsensusMajorityVote)
{
    ScriptedModel a, b, c;
    // Round 1: 2 similar, 1 different.
    a.add_response("The sky is blue.");
    b.add_response("The sky is blue.");
    c.add_response("The sky is green.");

    // Round 2: all converge to majority.
    a.add_response("The sky is blue.");
    b.add_response("The sky is blue.");
    c.add_response("The sky is blue.");

    std::vector<Model *> agents = {&a, &b, &c};
    DelphiConsensus consensus(agents, 3);
    ConsensusResult result = consensus.debate("What color is the sky?");
    EXPECT_TRUE(result.converged);
    EXPECT_NE(result.final_answer.find("blue"), std::string::npos);
}

TEST(MultiAgent, ConsensusEmptyAgents)
{
    std::vector<Model *> agents;
    DelphiConsensus consensus(agents, 3);
    ConsensusResult result = consensus.debate("Question");
    EXPECT_TRUE(result.final_answer.empty());
}