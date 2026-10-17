#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "autograd/engine.h"
#include "autograd/ops.h"
#include "finetune/sft_loader.h"
#include "finetune/sft_trainer.h"
#include "optim/adamw.h"
#include "transformer/model.h"

using namespace serius;
using namespace serius::core;

TEST(SFTDataLoader, ShiftsTargetsAndMasksOnlyResponseAndEos)
{
    BPETokenizer tokenizer;
    finetune::SFTDataLoader loader(tokenizer);
    finetune::SFTExample example{"Translate to English:", "Buongiorno.", "Good morning."};

    const std::string prompt = "[SYSTEM] Sei un assistente utile.\n[USER] Translate to English: Buongiorno.\n[ASSISTANT] ";
    const std::vector<TokenId> prompt_ids = tokenizer.encode(prompt);
    const std::vector<TokenId> response_ids = tokenizer.encode(example.output);

    const finetune::SFTSample sample = loader.prepare(example);
    ASSERT_EQ(sample.input_ids.size(), prompt_ids.size() + response_ids.size());
    ASSERT_EQ(sample.target_ids.size(), sample.input_ids.size());
    ASSERT_EQ(sample.loss_mask.size(), sample.target_ids.size());

    for (size_t i = 0; i < sample.input_ids.size(); ++i) {
        EXPECT_EQ(sample.target_ids[i], i + 1 < sample.input_ids.size() ? sample.input_ids[i + 1]
                                                                         : BPETokenizer::EOS_ID);
        EXPECT_EQ(sample.loss_mask[i], i + 1 >= prompt_ids.size());
    }
    EXPECT_EQ(sample.target_ids.back(), BPETokenizer::EOS_ID);
    EXPECT_EQ(std::count(sample.loss_mask.begin(), sample.loss_mask.end(), true), response_ids.size() + 1);
}

TEST(SFTDataLoader, RejectsMissingInstructionOrResponse)
{
    BPETokenizer tokenizer;
    finetune::SFTDataLoader loader(tokenizer);

    EXPECT_THROW(loader.prepare({"", "context", "response"}), std::invalid_argument);
    EXPECT_THROW(loader.prepare({"instruction", "context", ""}), std::invalid_argument);
}

TEST(SFTLoss, MaskedCrossEntropyIgnoresPromptRowsAndGradient)
{
    autograd::Node logits(Tensor({2, 3}));
    logits.data.data()[0] = 20.0f;
    logits.data.data()[1] = -5.0f;
    logits.data.data()[2] = 0.0f;
    logits.data.data()[3] = 0.0f;
    logits.data.data()[4] = 2.0f;
    logits.data.data()[5] = 1.0f;

    autograd::Node targets(Tensor({2}), false);
    targets.data.data()[0] = 0.0f;
    targets.data.data()[1] = 1.0f;

    Tensor mask({2});
    mask.data()[0] = 0.0f;
    mask.data()[1] = 1.0f;

    autograd::Node *loss = autograd::ops::masked_cross_entropy(&logits, &targets, mask);
    const float expected_loss = std::log(std::exp(0.0f) + std::exp(2.0f) + std::exp(1.0f)) - 2.0f;
    EXPECT_NEAR(loss->data.data()[0], expected_loss, 1e-6f);

    autograd::backward(loss);
    for (size_t token = 0; token < 3; ++token) {
        EXPECT_FLOAT_EQ(logits.grad.data()[token], 0.0f);
    }
    EXPECT_NEAR(logits.grad.data()[3], 0.09003057f, 1e-6f);
    EXPECT_NEAR(logits.grad.data()[4], -0.33475906f, 1e-6f);
    EXPECT_NEAR(logits.grad.data()[5], 0.24472847f, 1e-6f);
}

TEST(SFTLoss, RejectsEmptyMaskAndMismatchedShapes)
{
    autograd::Node logits(Tensor({2, 3}));
    autograd::Node targets(Tensor({2}), false);
    Tensor empty_mask({2});
    EXPECT_THROW(autograd::ops::masked_cross_entropy(&logits, &targets, empty_mask), std::invalid_argument);

    Tensor mismatched_mask({1});
    EXPECT_THROW(autograd::ops::masked_cross_entropy(&logits, &targets, mismatched_mask), std::invalid_argument);
}

TEST(SFTTrainer, SingleStepComputesFiniteLossAndUpdatesStep)
{
    model::ModelConfig config;
    config.vocab_size = 260;
    config.d_model = 16;
    config.n_heads = 2;
    config.n_layers = 1;
    config.d_ff = 32;
    config.max_seq_len = 64;

    model::TransformerModel model(config);
    optim::AdamWConfig adam_cfg;
    adam_cfg.lr = 1e-3f;
    optim::AdamW optimizer(model.parameters(), adam_cfg);

    finetune::SFTTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-3f;
    finetune::SFTTrainer trainer(model, optimizer, trainer_cfg);

    BPETokenizer tokenizer;
    finetune::SFTDataLoader loader(tokenizer);
    finetune::SFTSample sample = loader.prepare({"Saluta l'utente", "", "Ciao!"});

    float loss = trainer.train_step(sample);
    EXPECT_GT(loss, 0.0f);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_EQ(trainer.current_step(), 1u);
    ASSERT_EQ(trainer.loss_history().size(), 1u);
    EXPECT_FLOAT_EQ(trainer.loss_history()[0], loss);
}

TEST(SFTTrainer, ConvergenceOnInstructionPairs)
{
    // Overfit on a single instruction-response pair.
    // Weight decay is disabled so the tiny gradient signal (only response
    // tokens are masked) is not overwhelmed by regularisation pulling all
    // ~24K parameters toward zero.
    model::ModelConfig config;
    config.vocab_size = 260;
    config.d_model = 32;
    config.n_heads = 4;
    config.n_layers = 1;
    config.d_ff = 64;
    config.max_seq_len = 128;

    model::TransformerModel model(config);
    optim::AdamWConfig adam_cfg;
    adam_cfg.lr = 0.01f;
    adam_cfg.weight_decay = 0.0f;          // critical: disable weight decay
    optim::AdamW optimizer(model.parameters(), adam_cfg);

    finetune::SFTTrainerConfig trainer_cfg;
    trainer_cfg.max_steps = 80;
    trainer_cfg.learning_rate = 0.01f;
    trainer_cfg.epochs = 80;               // 1 sample/epoch → 80 steps total
    trainer_cfg.grad_clip = 0.0f;          // let gradients flow unclipped
    trainer_cfg.verbose = true;
    trainer_cfg.log_interval = 10;
    finetune::SFTTrainer trainer(model, optimizer, trainer_cfg);

    BPETokenizer tokenizer;
    finetune::SFTDataLoader loader(tokenizer);

    // Single short example: maximises the gradient signal per step.
    finetune::SFTSample sample = loader.prepare({"Ciao", "", "Hi"});
    std::vector<finetune::SFTSample> dataset = {sample};

    const float initial_eval_loss = trainer.evaluate(dataset);
    EXPECT_GT(initial_eval_loss, 0.0f);

    auto step_losses = trainer.train(dataset);
    ASSERT_EQ(step_losses.size(), 80u);
    EXPECT_EQ(trainer.current_step(), 80u);

    const float final_eval_loss = trainer.evaluate(dataset);
    EXPECT_GT(final_eval_loss, 0.0f);
    EXPECT_LT(final_eval_loss, initial_eval_loss);
}

TEST(SFTTrainer, FreezeParametersPreventsGradUpdates)
{
    model::ModelConfig config;
    config.vocab_size = 260;
    config.d_model = 16;
    config.n_heads = 2;
    config.n_layers = 1;
    config.d_ff = 32;
    config.max_seq_len = 64;

    model::TransformerModel model(config);
    optim::AdamWConfig adam_cfg;
    adam_cfg.lr = 1e-3f;
    optim::AdamW optimizer(model.parameters(), adam_cfg);

    finetune::SFTTrainerConfig trainer_cfg;
    finetune::SFTTrainer trainer(model, optimizer, trainer_cfg);

    // Freeze embedding
    trainer.freeze_embedding();
    EXPECT_FALSE(model.embedding->requires_grad);

    Tensor initial_emb = model.embedding->data.copy();

    BPETokenizer tokenizer;
    finetune::SFTDataLoader loader(tokenizer);
    finetune::SFTSample sample = loader.prepare({"Domanda", "", "Risposta"});

    trainer.train_step(sample);

    // Verifichiamo che embedding non sia cambiato
    for (size_t i = 0; i < initial_emb.size(); ++i) {
        EXPECT_FLOAT_EQ(model.embedding->data.data()[i], initial_emb.data()[i]);
    }

    // Unfreeze all
    trainer.unfreeze_all();
    EXPECT_TRUE(model.embedding->requires_grad);
}
