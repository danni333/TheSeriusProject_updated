#include <gtest/gtest.h>

#include <filesystem>

#include "../src/autograd/engine.h"
#include "../src/autograd/ops.h"
#include "../src/data/dataloader.h"
#include "../src/optim/adamw.h"
#include "../src/optim/lr_schedule.h"
#include "../src/train/checkpoint.h"
#include "../src/train/trainer.h"
#include "../src/transformer/model.h"

using namespace serius;
using namespace serius::transformer;
using namespace serius::optim;
using namespace serius::training;

class TrainingTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        config.vocab_size = 100;
        config.d_model = 32;
        config.n_heads = 4;
        config.n_layers = 2;
        config.d_ff = 64;
        config.max_seq_len = 16;
    }

    ModelConfig config;
};

TEST_F(TrainingTest, FullTrainingStep)
{
    TransformerModel model(config);
    AdamWConfig adam_config;
    adam_config.lr = 1e-3f;

    auto params = model.parameters();
    AdamW optimizer(params, adam_config);
    WSDScheduler scheduler(1e-3f, 10, 10, 10);

    TrainerConfig trainer_config;
    trainer_config.max_epochs = 1;
    trainer_config.checkpoint_interval = 5;
    trainer_config.checkpoint_dir = "test_checkpoints";

    std::filesystem::create_directory(trainer_config.checkpoint_dir);

    Trainer trainer(model, optimizer, scheduler, trainer_config);

    // We need a dummy dataloader or mock
    // For now, let's just do a manual step to verify tensors don't crash and grads flow

    Tensor input_ids({1, 8});  // B=1, S=8
    input_ids.fill(1.0f);
    Tensor target_ids({1, 8});
    target_ids.fill(2.0f);

    autograd::Node *input_node = new autograd::Node(std::move(input_ids), false);

    // Manual step logic from Trainer to verify
    optimizer.zero_grad();
    autograd::Node *logits = model.forward(input_node);

    // logits is (1, 8, 100) -> CE needs (B, V) or (B*S, V)
    // Reshape logits to (8, 100)
    Tensor logits_reshaped = logits->data.copy();
    logits_reshaped.reshape({8, 100});
    autograd::Node *logits_node = new autograd::Node(std::move(logits_reshaped), true);
    logits_node->children = {logits};
    logits_node->backward_fn = [logits, logits_node]() {
        if (logits->requires_grad) {
            logits->grad += logits_node->grad.reshape(logits->grad.shape());
        }
    };

    // targets needs to be (8,)
    Tensor target_reshaped({8});
    target_reshaped.fill(5.0f);
    autograd::Node *target_node_reshaped = new autograd::Node(std::move(target_reshaped), false);

    autograd::Node *loss = autograd::ops::cross_entropy(logits_node, target_node_reshaped);

    float initial_loss = loss->data[0];
    EXPECT_GT(initial_loss, 0.0f);

    autograd::backward(loss);

    // Check if some weight has gradient
    bool has_grad = false;
    for (auto *p : params) {
        for (size_t i = 0; i < p->grad.size(); ++i) {
            if (std::abs(p->grad[i]) > 1e-8f) {
                has_grad = true;
                break;
            }
        }
        if (has_grad) break;
    }
    EXPECT_TRUE(has_grad);

    optimizer.step();

    // Save/Load check
    save_checkpoint("test_checkpoints/test.ckpt", model, optimizer, 0, 1);
    EXPECT_TRUE(std::filesystem::exists("test_checkpoints/test.ckpt"));

    int epoch, step;
    load_checkpoint("test_checkpoints/test.ckpt", model, optimizer, epoch, step);
    EXPECT_EQ(epoch, 0);
    EXPECT_EQ(step, 1);

    std::filesystem::remove_all("test_checkpoints");
}
