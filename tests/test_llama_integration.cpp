// End-to-end tests against a real GGUF model. Skipped when the model is absent.
// Override the model with AVIS_TEST_MODEL=/path/to/model.gguf.

#include "avis/engine/engine.hpp"
#include "avis/llama/llama_executor.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

using namespace avis;
using namespace avis::engine;

namespace {

std::string model_path() {
    if (const char* env = std::getenv("AVIS_TEST_MODEL")) {
        return env;
    }
    return AVIS_SOURCE_DIR "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

class LlamaIntegration : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(model_path())) {
            return;
        }
        llama::LlamaConfig cfg{.model_path = model_path(), .n_ctx = 4096, .n_batch = 256, .n_seq_max = 8};
        auto loaded = llama::LlamaExecutor::load(cfg);
        ASSERT_TRUE(loaded) << loaded.error().message;
        exec_ = std::move(*loaded);
    }
    static void TearDownTestSuite() { exec_.reset(); }

    void SetUp() override {
        if (!exec_) {
            GTEST_SKIP() << "model not found: " << model_path();
        }
    }

    static EngineConfig config(std::size_t max_running, std::size_t num_blocks = 256) {
        return {.scheduler = {.max_running = max_running, .max_step_tokens = 256, .prefill_chunk = 64,
                              .max_seq_len = 4096},
                .kv = {.num_blocks = num_blocks, .block_size = 16, .num_slots = 8}};
    }

    static Request greedy(std::string_view user, std::size_t max_new = 24) {
        auto tokens = exec_->tokenize(exec_->format_chat("You are a terse assistant.", user));
        EXPECT_TRUE(tokens);
        Request r;
        r.prompt = *tokens;
        r.max_new_tokens = max_new;
        r.sampling.temperature = 0.0f; // greedy: output depends only on the KV contents
        return r;
    }

    static std::vector<Token> run_alone(const Request& request) {
        FixedClockPolicy policy;
        Engine engine(*exec_, config(1), policy);
        auto f = engine.submit(request);
        while (engine.step()) {
        }
        return f.get().tokens;
    }

    static std::string detok(std::span<const Token> tokens) {
        std::string text;
        for (const Token t : tokens) {
            text += exec_->token_to_piece(t);
        }
        return text;
    }

    static inline std::unique_ptr<llama::LlamaExecutor> exec_;
};

} // namespace

TEST_F(LlamaIntegration, TokenizeAndDetokenizeRoundTrip) {
    const auto tokens = exec_->tokenize("Energy aware inference", /*add_special=*/false);
    ASSERT_TRUE(tokens);
    std::string text;
    for (const Token t : *tokens) {
        text += exec_->token_to_piece(t);
    }
    EXPECT_EQ(text, "Energy aware inference");
    EXPECT_FALSE(exec_->format_chat("sys", "hi").empty());
    EXPECT_FALSE(exec_->description().empty());
}

TEST_F(LlamaIntegration, GeneratesText) {
    const auto tokens = run_alone(greedy("Name a primary colour.", 8));
    EXPECT_FALSE(tokens.empty());
}

static std::vector<std::vector<Token>> run_batched(llama::LlamaExecutor& exec, const std::vector<Request>& requests,
                                                             std::size_t* max_batch) {
    FixedClockPolicy policy;
    Engine engine(exec, {.scheduler = {.max_running = 4, .max_step_tokens = 256, .prefill_chunk = 64},
                         .kv = {.num_blocks = 256, .block_size = 16, .num_slots = 8}},
                  policy);
    std::vector<std::future<RequestOutput>> futures;
    for (const auto& r : requests) {
        futures.push_back(engine.submit(r));
    }
    while (engine.step()) {
    }
    *max_batch = engine.stats().max_batch_seqs;
    std::vector<std::vector<Token>> out;
    for (auto& f : futures) {
        out.push_back(f.get().tokens);
    }
    return out;
}

TEST_F(LlamaIntegration, BatchedSequencesKeepTheirOwnContext) {
    const std::vector<std::string> questions{
        "What is a GPU?", "Explain KV caching in one sentence.", "What is 12 times 12?", "Define latency."};
    std::vector<Request> requests;
    for (const auto& q : questions) {
        requests.push_back(greedy(q));
    }
    std::size_t max_batch = 0;
    const auto batched = run_batched(*exec_, requests, &max_batch);
    EXPECT_GT(max_batch, 1u);
    for (std::size_t i = 0; i < questions.size(); ++i) {
        const auto alone = run_alone(requests[i]);
        // Batched kernels round differently, so long greedy continuations may
        // legitimately fork at near-ties. Leaked KV from another sequence would
        // instead change the very first predictions.
        ASSERT_GE(alone.size(), 2u);
        ASSERT_GE(batched[i].size(), 2u);
        EXPECT_EQ(alone[0], batched[i][0]) << "request " << i << "\n  alone:   " << detok(alone)
                                           << "\n  batched: " << detok(batched[i]);
        EXPECT_EQ(alone[1], batched[i][1]) << "request " << i;
    }
}

TEST_F(LlamaIntegration, BatchedRunsAreDeterministic) {
    std::vector<Request> requests;
    for (const auto* q : {"Why is the sky blue?", "Name two sorting algorithms.", "What is RAM?"}) {
        requests.push_back(greedy(q, 32));
    }
    std::size_t max_batch = 0;
    const auto first = run_batched(*exec_, requests, &max_batch);
    const auto second = run_batched(*exec_, requests, &max_batch);
    EXPECT_EQ(first, second);
}

TEST_F(LlamaIntegration, PrefixCacheReuseKeepsOutputIdentical) {
    const auto request = greedy("List three uses of a hash map.");
    const auto reference = run_alone(request);

    FixedClockPolicy policy;
    Engine engine(*exec_, config(2), policy);
    auto first = engine.submit(request);
    while (engine.step()) {
    }
    auto second = engine.submit(request); // same prompt: served mostly from the prefix cache
    while (engine.step()) {
    }
    const auto out1 = first.get();
    const auto out2 = second.get();
    EXPECT_GT(out2.metrics.cached_prompt_tokens, 0u);
    EXPECT_EQ(out1.tokens, reference);
    const std::size_t n = std::min<std::size_t>(12, reference.size());
    EXPECT_TRUE(std::equal(reference.begin(), reference.begin() + static_cast<std::ptrdiff_t>(n), out2.tokens.begin()));
}

TEST_F(LlamaIntegration, CompletesUnderKvPressureWithPreemption) {
    FixedClockPolicy policy;
    // 12 blocks of 16 tokens: four ~40-token prompts growing by 48 tokens cannot all stay resident.
    Engine engine(*exec_, config(4, 12), policy);
    std::vector<std::future<RequestOutput>> futures;
    for (int i = 0; i < 4; ++i) {
        auto r = greedy(std::format("Count from {} upwards.", i * 10), 48);
        r.ignore_eos = true;
        futures.push_back(engine.submit(std::move(r)));
    }
    while (engine.step()) {
    }
    for (auto& f : futures) {
        EXPECT_EQ(f.get().tokens.size(), 48u);
    }
    EXPECT_GT(engine.stats().preemptions, 0u);
}
