#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/neuro/encoder.h"
#include "../src/neuro/event_engine.h"
#include "../src/neuro/lif_neuron.h"
#include "../src/neuro/spiking_layer.h"

using namespace serius::neuro;
using serius::core::Tensor;

// ---------------------------------------------------------------------------
// LIF Neuron Tests
// ---------------------------------------------------------------------------

TEST(LIFNeuronTest, ThresholdSpikeAndReset)
{
    LIFNeuron neuron(/*threshold=*/1.0f, /*decay=*/1.0f, /*reset_value=*/0.0f);

    // Sub-threshold input: no spike
    EXPECT_FALSE(neuron.step(0.4f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 0.4f);

    // Another sub-threshold step: 0.4 + 0.4 = 0.8 < 1.0 -> no spike
    EXPECT_FALSE(neuron.step(0.4f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 0.8f);

    // Reaching threshold: 0.8 + 0.3 = 1.1 >= 1.0 -> SPIKE and reset
    EXPECT_TRUE(neuron.step(0.3f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 0.0f);
}

TEST(LIFNeuronTest, MembranePotentialDecay)
{
    LIFNeuron neuron(/*threshold=*/2.0f, /*decay=*/0.5f, /*reset_value=*/0.0f);

    // Inject initial charge
    EXPECT_FALSE(neuron.step(1.0f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 1.0f);

    // Without input current, potential should decay by 50% each step
    EXPECT_FALSE(neuron.step(0.0f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 0.5f);

    EXPECT_FALSE(neuron.step(0.0f));
    EXPECT_FLOAT_EQ(neuron.membrane_potential(), 0.25f);
}

TEST(LIFNeuronTest, RefractoryPeriodEnforcement)
{
    LIFNeuron neuron(/*threshold=*/1.0f, /*decay=*/1.0f, /*reset_value=*/0.0f, /*refractory=*/2);

    // Spike triggered
    EXPECT_TRUE(neuron.step(1.5f));
    EXPECT_TRUE(neuron.is_refractory());

    // Refractory step 1: huge input ignored, no spike
    EXPECT_FALSE(neuron.step(10.0f));
    EXPECT_TRUE(neuron.is_refractory());

    // Refractory step 2: huge input ignored, no spike
    EXPECT_FALSE(neuron.step(10.0f));
    EXPECT_FALSE(neuron.is_refractory());

    // Recovered: now responds normally to input
    EXPECT_TRUE(neuron.step(1.2f));
}

// ---------------------------------------------------------------------------
// Spiking Linear Layer Tests
// ---------------------------------------------------------------------------

TEST(SpikingLayerTest, ForwardSpikeIntegration)
{
    const size_t in_dim = 4;
    const size_t out_dim = 2;
    SpikingLinear layer(in_dim, out_dim, /*threshold=*/1.0f, /*decay=*/1.0f);

    // Set synaptic weights
    // Neuron 0: responds strongly to input 0 and 1
    // Neuron 1: responds strongly to input 2 and 3
    Tensor &w = layer.weights();
    w[0 * in_dim + 0] = 0.6f;
    w[0 * in_dim + 1] = 0.6f;
    w[1 * in_dim + 2] = 0.6f;
    w[1 * in_dim + 3] = 0.6f;

    // Pattern 1: inputs 0 and 1 active
    std::vector<bool> in1 = {true, true, false, false};
    auto out1 = layer.forward(in1);
    EXPECT_TRUE(out1[0]);   // 0.6 + 0.6 = 1.2 >= 1.0 -> spikes
    EXPECT_FALSE(out1[1]);  // 0.0 < 1.0 -> silent

    layer.reset();

    // Pattern 2: inputs 2 and 3 active
    std::vector<bool> in2 = {false, false, true, true};
    auto out2 = layer.forward(in2);
    EXPECT_FALSE(out2[0]);
    EXPECT_TRUE(out2[1]);
}

TEST(SpikingLayerTest, STDPPlasticity)
{
    SpikingLinear layer(2, 1, /*threshold=*/1.0f);
    Tensor &w = layer.weights();
    w[0] = 0.5f;  // synapse 0
    w[1] = 0.5f;  // synapse 1

    // Pre 0 fired, Pre 1 silent, Post 0 fired
    std::vector<bool> pre = {true, false};
    std::vector<bool> post = {true};

    layer.apply_stdp(pre, post, /*lr_ltp=*/0.1f, /*lr_ltd=*/0.05f);

    // Synapse 0: LTP (pre + post) -> increases
    EXPECT_NEAR(w[0], 0.6f, 1e-5f);
    // Synapse 1: LTD (post without pre) -> depresses
    EXPECT_NEAR(w[1], 0.45f, 1e-5f);
}

// ---------------------------------------------------------------------------
// Event-Driven Execution Engine Tests
// ---------------------------------------------------------------------------

TEST(EventEngineTest, EventSparseExecutionWorkload)
{
    const size_t num_neurons = 1000;
    std::vector<LIFNeuron> neurons(num_neurons, LIFNeuron(/*threshold=*/1.0f, /*decay=*/0.9f));

    Tensor weights({num_neurons, num_neurons});
    weights.zeros();

    // Sparse connections: only 2% non-zero weights
    for (size_t i = 0; i < num_neurons; ++i) {
        weights[i * num_neurons + (i % 20)] = 0.5f;
        weights[i * num_neurons + ((i + 7) % 20)] = 0.6f;
    }

    EventEngine engine;

    // Only 4 active source neurons out of 1000 (< 0.5% input activity)
    std::vector<int> active_sources = {3, 15, 42, 178};

    std::vector<int> fired = engine.step(active_sources, weights, neurons);

    const EventStats &stats = engine.last_stats();
    // Verify sparsity: active neurons must be < 5% of total
    EXPECT_LT(stats.activity_ratio, 0.05f);
    EXPECT_GT(stats.sparsity_ratio, 0.95f);

    // Verify significant compute savings (> 95% ops saved)
    size_t total_possible_ops = num_neurons * num_neurons;
    EXPECT_GT(stats.operations_saved, total_possible_ops * 95 / 100);
}

TEST(EventEngineTest, SkipsNegligibleWeights)
{
    std::vector<LIFNeuron> neurons(4, LIFNeuron(1.0f, 1.0f));
    Tensor weights({4, 2});
    weights.zeros();
    weights[0 * 2 + 0] = 0.0000001f;  // Below threshold
    weights[1 * 2 + 0] = 1.5f;        // Above threshold

    EventEngine engine;
    std::vector<int> active_sources = {0};

    std::vector<int> fired = engine.step(active_sources, weights, neurons, /*weight_threshold=*/1e-5f);

    // Only 1 operation performed
    EXPECT_EQ(engine.last_stats().operations_performed, 1u);
    // Destination neuron 1 spiked because 1.5 >= 1.0
    ASSERT_EQ(fired.size(), 1u);
    EXPECT_EQ(fired[0], 1);
}

// ---------------------------------------------------------------------------
// Spike Encoder and Decoder Tests
// ---------------------------------------------------------------------------

TEST(SpikeEncoderTest, RateEncodeMonotonicityAndDecoding)
{
    const size_t steps = 100;
    auto sp_low = SpikeEncoder::rate_encode(0.2f, steps, true);
    auto sp_med = SpikeEncoder::rate_encode(0.5f, steps, true);
    auto sp_high = SpikeEncoder::rate_encode(0.8f, steps, true);

    float dec_low = SpikeEncoder::decode_rate(sp_low);
    float dec_med = SpikeEncoder::decode_rate(sp_med);
    float dec_high = SpikeEncoder::decode_rate(sp_high);

    EXPECT_NEAR(dec_low, 0.2f, 0.02f);
    EXPECT_NEAR(dec_med, 0.5f, 0.02f);
    EXPECT_NEAR(dec_high, 0.8f, 0.02f);

    EXPECT_LT(dec_low, dec_med);
    EXPECT_LT(dec_med, dec_high);
}

TEST(SpikeEncoderTest, LatencyEncodingEarliestSpike)
{
    const size_t steps = 20;
    auto sp_high = SpikeEncoder::latency_encode(0.95f, steps);
    auto sp_low = SpikeEncoder::latency_encode(0.1f, steps);

    // Find time of spikes
    size_t t_high = 999, t_low = 999;
    for (size_t t = 0; t < steps; ++t) {
        if (sp_high[t] && t_high == 999) t_high = t;
        if (sp_low[t] && t_low == 999) t_low = t;
    }

    // High input must spike earlier than low input
    EXPECT_LT(t_high, t_low);

    // Decode check
    float dec_high = SpikeEncoder::decode_latency(sp_high);
    float dec_low = SpikeEncoder::decode_latency(sp_low);
    EXPECT_GT(dec_high, dec_low);
}

// ---------------------------------------------------------------------------
// End-to-End Neuromorphic Pattern Logic (Spiking XOR)
// ---------------------------------------------------------------------------

TEST(NeuromorphicTest, SpikingXORLogic)
{
    // A 2-layer spiking network computing XOR(x0, x1)
    // Hidden layer: 2 neurons
    // H0 fires when either input is on (OR behavior): w=(1.0, 1.0), threshold=0.9
    // H1 fires when BOTH inputs are on (AND behavior): w=(1.0, 1.0), threshold=1.9
    // Output neuron: fires if H0 fires and H1 is inhibited:
    // w=(1.0, -1.5), threshold=0.9

    SpikingLinear hidden(2, 2, /*threshold=*/0.9f, /*decay=*/0.0f);
    hidden.neurons()[1].set_threshold(1.9f);  // AND detector

    Tensor &Wh = hidden.weights();
    Wh[0 * 2 + 0] = 1.0f;  Wh[0 * 2 + 1] = 1.0f;  // H0 (OR)
    Wh[1 * 2 + 0] = 1.0f;  Wh[1 * 2 + 1] = 1.0f;  // H1 (AND)

    SpikingLinear output(2, 1, /*threshold=*/0.8f, /*decay=*/0.0f);
    Tensor &Wo = output.weights();
    Wo[0] = 1.0f;   // Excitatory from H0
    Wo[1] = -1.5f;  // Inhibitory from H1

    auto eval_xor = [&](bool in0, bool in1) -> bool {
        hidden.reset();
        output.reset();
        auto h_out = hidden.forward({in0, in1});
        auto o_out = output.forward(h_out);
        return o_out[0];
    };

    EXPECT_FALSE(eval_xor(false, false)) << "XOR(0, 0) should be 0";
    EXPECT_TRUE(eval_xor(true, false))   << "XOR(1, 0) should be 1";
    EXPECT_TRUE(eval_xor(false, true))   << "XOR(0, 1) should be 1";
    EXPECT_FALSE(eval_xor(true, true))   << "XOR(1, 1) should be 0";
}
