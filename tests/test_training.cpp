#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "autograd/engine.h"
#include "autograd/ops.h"
#include "data/dataloader.h"
#include "optim/adamw.h"
#include "optim/lr_schedule.h"
#include "tensor/tensor.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"
#include "transformer/config.h"
#include "transformer/model.h"

namespace serius {
namespace test {

using core::Tensor;
using core::autograd::Node;
using namespace model;
using namespace optim;
using namespace training;
namespace autograd = core::autograd;

class TrainingTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        const int VOCAB_SIZE = 100;
        const int D_MODEL = 32;
        const int N_HEADS = 4;
        const int N_LAYERS = 2;
        const int D_FF = 64;
        const int MAX_SEQ_LEN = 16;

        config.vocab_size = VOCAB_SIZE;
        config.d_model = D_MODEL;
        config.n_heads = N_HEADS;
        config.n_layers = N_LAYERS;
        config.d_ff = D_FF;
        config.max_seq_len = MAX_SEQ_LEN;
    }

    ModelConfig config;
};

TEST_F(TrainingTest, FullTrainingStep)
{
    TransformerModel model(config);
    AdamWConfig adam_config;

    const float LEARNING_RATE = 1e-3F;
    adam_config.lr = LEARNING_RATE;

    auto params = model.parameters();
    AdamW optimizer(params, adam_config);

    const int SCHEDULER_STEPS = 10;
    WSDScheduler scheduler(LEARNING_RATE, SCHEDULER_STEPS, SCHEDULER_STEPS, SCHEDULER_STEPS);

    TrainerConfig trainer_config;

    const int MAX_EPOCHS = 1;
    trainer_config.max_epochs = MAX_EPOCHS;

    Trainer trainer(model, optimizer, scheduler, trainer_config);

    const size_t BATCH_SIZE = 1;
    const size_t SEQ_LEN = 8;

    std::vector<size_t> shape_input = {BATCH_SIZE, SEQ_LEN};
    Tensor input_ids(shape_input);

    const float INPUT_VAL = 1.0F;
    input_ids.fill(INPUT_VAL);

    std::vector<size_t> shape_target = {SEQ_LEN};
    Tensor target_ids(shape_target);

    const float TARGET_VAL = 2.0F;
    target_ids.fill(TARGET_VAL);

    autograd::Node *input_node = new autograd::Node(input_ids.copy(), false);

    optimizer.zero_grad();
    autograd::Node *logits = model.forward(input_node);

    Tensor logits_reshaped = logits->data.copy();
    const size_t VOCAB_SIZE = 100;  // Match config
    logits_reshaped.reshape({SEQ_LEN, VOCAB_SIZE});
    autograd::Node *logits_node = new autograd::Node(logits_reshaped.copy(), true);
    logits_node->children = {logits};
    logits_node->backward_fn = [logits, logits_node]() {
        if (logits->requires_grad) {
            Tensor grad_copy = logits_node->grad.copy();
            grad_copy.reshape(logits->grad.shape());
            logits->grad = logits->grad + grad_copy;
        }
    };

    autograd::Node *target_node = new autograd::Node(target_ids.copy(), false);

    autograd::Node *loss = autograd::ops::cross_entropy(logits_node, target_node);

    const float ZERO_LOSS = 0.0F;
    float initial_loss = loss->data.at({0});
    EXPECT_GT(initial_loss, ZERO_LOSS);

    autograd::backward(loss);

    bool has_grad = false;
    for (auto *p : params) {
        if (p->grad.size() > 0) {
            has_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_grad);

    optimizer.step();
}

TEST_F(TrainingTest, MixedPrecisionTrainingKeepsParametersFinite)
{
    TransformerModel model(config);

    AdamWConfig adam_config;
    adam_config.lr = 1e-3F;
    auto params = model.parameters();
    AdamW optimizer(params, adam_config);

    WSDScheduler scheduler(adam_config.lr, 10, 10, 10);
    TrainerConfig trainer_config;
    trainer_config.max_epochs = 1;
    trainer_config.use_amp = true;
    Trainer trainer(model, optimizer, scheduler, trainer_config);
    std::vector<Tensor> initial_params;
    initial_params.reserve(params.size());
    for (const auto *param : params) {
        initial_params.push_back(param->data.copy());
    }

    const std::filesystem::path data_path =
        std::filesystem::temp_directory_path() / "serius_training_amp_test.txt";
    {
        std::ofstream data_file(data_path);
        ASSERT_TRUE(data_file.is_open());
        data_file << "ABCDEFGHIJKLMNOPQRSTUVWX";
    }

    DataLoader loader(std::make_shared<BPETokenizer>(), 1, 8, 1);
    loader.load_files({data_path.string()});
    ASSERT_TRUE(loader.next_batch().has_value());
    loader.reset();

    trainer.train(loader);
    std::filesystem::remove(data_path);

    bool parameters_updated = false;
    for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
        const auto *param = params[param_idx];
        for (size_t i = 0; i < param->data.size(); ++i) {
            const float value = param->data.item(i);
            EXPECT_TRUE(std::isfinite(value)) << "non-finite parameter element " << i;
            parameters_updated = parameters_updated || value != initial_params[param_idx].item(i);
        }
    }
    EXPECT_TRUE(parameters_updated);
}

}  // namespace test
}  // namespace serius
