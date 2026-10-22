#include "avis/engine/scheduler.hpp"

#include <gtest/gtest.h>

#include <map>
#include <numeric>

using namespace avis;
using namespace avis::engine;

namespace {

constexpr Token kEog = -1;

std::vector<Token> iota_tokens(std::size_t n, Token start = 0) {
    std::vector<Token> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

Request make_request(std::vector<Token> prompt, std::size_t max_new) {
    Request r;
    r.prompt = std::move(prompt);
    r.max_new_tokens = max_new;
    return r;
}

// Drives a scheduler with a deterministic fake model: the "next token" of a
// sequence is 1000 + its current length, so outputs are easy to predict.
struct Harness {
    explicit Harness(kv::BlockManagerConfig kv_cfg, SchedulerConfig cfg) : blocks(kv_cfg), sched(cfg, blocks) {}

    kv::BlockManager blocks;
    Scheduler sched;
    std::map<RequestId, std::unique_ptr<Sequence>> finished;
    std::vector<StepPlan> history;
    std::function<Token(const Sequence&)> model = [](const Sequence& s) {
        return static_cast<Token>(1000 + s.tokens.size());
    };

    void step() {
        auto plan = sched.schedule(Clock::now());
        std::vector<Token> sampled;
        for (const auto& c : plan.chunks) {
            if (c.sample) {
                sampled.push_back(model(*c.seq));
            }
        }
        auto outcome = sched.complete(plan, sampled, [](Token t) { return t == kEog; }, Clock::now());
        for (auto& seq : outcome.finished) {
            finished.emplace(seq->id, std::move(seq));
        }
        history.push_back(std::move(plan));
    }

    void run(std::size_t max_steps = 10'000) {
        for (std::size_t i = 0; i < max_steps && sched.has_work(); ++i) {
            step();
        }
        ASSERT_FALSE(sched.has_work()) << "scheduler did not drain";
    }
};

kv::BlockManagerConfig roomy_kv() { return {.num_blocks = 256, .block_size = 4, .num_slots = 8}; }

} // namespace

TEST(Scheduler, SingleRequestGeneratesUntilMaxTokens) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 64, .prefill_chunk = 64});
    ASSERT_TRUE(h.sched.add(1, make_request(iota_tokens(10), 5), Clock::now()));
    h.run();
    ASSERT_EQ(h.finished.size(), 1u);
    const auto& seq = *h.finished.at(1);
    EXPECT_EQ(seq.generated(), 5u);
    EXPECT_EQ(seq.finish_reason, FinishReason::length);
    // One prefill step that samples the first token, then four decode steps.
    ASSERT_EQ(h.history.size(), 5u);
    EXPECT_EQ(h.history[0].prefill_tokens, 10u);
    EXPECT_EQ(h.history[1].decode_tokens, 1u);
    // Generated tokens follow the fake model.
    EXPECT_EQ(seq.generated_tokens()[0], 1010);
    EXPECT_EQ(seq.generated_tokens()[4], 1014);
    EXPECT_TRUE(seq.metrics.first_token.has_value());
    EXPECT_TRUE(seq.metrics.finished.has_value());
}

TEST(Scheduler, ChunkedPrefillRespectsChunkSize) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 64, .prefill_chunk = 16});
    ASSERT_TRUE(h.sched.add(1, make_request(iota_tokens(40), 2), Clock::now()));
    h.run();
    ASSERT_GE(h.history.size(), 3u);
    EXPECT_EQ(h.history[0].prefill_tokens, 16u);
    EXPECT_FALSE(h.history[0].chunks[0].sample);
    EXPECT_EQ(h.history[1].prefill_tokens, 16u);
    EXPECT_EQ(h.history[2].prefill_tokens, 8u);
    EXPECT_TRUE(h.history[2].chunks[0].sample);
}

TEST(Scheduler, BatchesConcurrentRequestsIntoOneStep) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 256, .prefill_chunk = 64});
    for (RequestId id = 1; id <= 3; ++id) {
        ASSERT_TRUE(h.sched.add(id, make_request(iota_tokens(8, static_cast<Token>(id * 100)), 4), Clock::now()));
    }
    h.step();
    EXPECT_EQ(h.history[0].chunks.size(), 3u);
    h.step();
    EXPECT_EQ(h.history[1].decode_tokens, 3u);
    h.run();
    EXPECT_EQ(h.finished.size(), 3u);
}

TEST(Scheduler, DecodeIsScheduledBeforeNewPrefill) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 10, .prefill_chunk = 64});
    ASSERT_TRUE(h.sched.add(1, make_request(iota_tokens(6), 10), Clock::now()));
    h.step(); // prefill of request 1
    ASSERT_TRUE(h.sched.add(2, make_request(iota_tokens(30, 500), 2), Clock::now()));
    h.step();
    const auto& plan = h.history[1];
    ASSERT_EQ(plan.chunks.size(), 2u);
    EXPECT_EQ(plan.chunks[0].seq->id, 1u); // decode first
    EXPECT_EQ(plan.decode_tokens, 1u);
    EXPECT_EQ(plan.prefill_tokens, 9u); // the rest of the 10-token budget
}

TEST(Scheduler, RespectsMaxRunning) {
    Harness h(roomy_kv(), {.max_running = 2, .max_step_tokens = 256, .prefill_chunk = 64});
    for (RequestId id = 1; id <= 5; ++id) {
        ASSERT_TRUE(h.sched.add(id, make_request(iota_tokens(4, static_cast<Token>(id * 100)), 3), Clock::now()));
    }
    while (h.sched.has_work()) {
        h.step();
        EXPECT_LE(h.sched.num_running(), 2u);
    }
    EXPECT_EQ(h.finished.size(), 5u);
}

TEST(Scheduler, StopsOnEndOfGeneration) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 64, .prefill_chunk = 64});
    h.model = [](const Sequence& s) { return s.generated() == 3 ? kEog : Token{7}; };
    ASSERT_TRUE(h.sched.add(1, make_request(iota_tokens(5), 100), Clock::now()));
    h.run();
    const auto& seq = *h.finished.at(1);
    EXPECT_EQ(seq.finish_reason, FinishReason::stop);
    EXPECT_EQ(seq.generated(), 3u); // the end-of-generation token is not part of the output
}

TEST(Scheduler, IgnoreEosKeepsGenerating) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 64, .prefill_chunk = 64});
    h.model = [](const Sequence&) { return kEog; };
    auto req = make_request(iota_tokens(5), 4);
    req.ignore_eos = true;
    ASSERT_TRUE(h.sched.add(1, std::move(req), Clock::now()));
    h.run();
    EXPECT_EQ(h.finished.at(1)->generated(), 4u);
}

TEST(Scheduler, PreemptsAndRecomputesWhenKvIsExhausted) {
    // 12 blocks of 4 tokens: three sequences growing to 4 + 20 tokens cannot all fit.
    Harness h({.num_blocks = 12, .block_size = 4, .num_slots = 4},
              {.max_running = 3, .max_step_tokens = 64, .prefill_chunk = 64, .watermark_blocks = 1});
    for (RequestId id = 1; id <= 3; ++id) {
        ASSERT_TRUE(h.sched.add(id, make_request(iota_tokens(4, static_cast<Token>(id * 100)), 20), Clock::now()));
    }
    h.run();
    ASSERT_EQ(h.finished.size(), 3u);
    EXPECT_GT(h.sched.num_preemptions(), 0u);
    for (const auto& [id, seq] : h.finished) {
        EXPECT_EQ(seq->generated(), 20u) << "request " << id;
        // Recomputation must not change the output of the deterministic model.
        for (std::size_t i = 0; i < seq->generated(); ++i) {
            EXPECT_EQ(seq->generated_tokens()[i], static_cast<Token>(1000 + 4 + i));
        }
    }
}

TEST(Scheduler, ReusesSharedPromptPrefix) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 256, .prefill_chunk = 256});
    const auto system = iota_tokens(16); // shared "system prompt"
    auto a = system;
    a.push_back(900);
    auto b = system;
    b.push_back(901);
    ASSERT_TRUE(h.sched.add(1, make_request(a, 2), Clock::now()));
    h.run();
    ASSERT_TRUE(h.sched.add(2, make_request(b, 2), Clock::now()));
    h.step();
    const auto& plan = h.history.back();
    EXPECT_EQ(plan.prefill_tokens, 1u); // only the unshared tail is computed
    h.run();
    EXPECT_EQ(h.finished.at(2)->metrics.cached_prompt_tokens, 16u);
}

TEST(Scheduler, RejectsInvalidRequests) {
    Harness h(roomy_kv(), {.max_running = 4, .max_seq_len = 32});
    EXPECT_FALSE(h.sched.add(1, make_request({}, 4), Clock::now()));
    EXPECT_FALSE(h.sched.add(2, make_request(iota_tokens(40), 4), Clock::now()));
    EXPECT_FALSE(h.sched.has_work());
}

TEST(Scheduler, KvExhaustionPreemptsNewest) {
    Harness h(roomy_kv(), {.max_running = 4, .max_step_tokens = 64, .prefill_chunk = 64});
    ASSERT_TRUE(h.sched.add(1, make_request(iota_tokens(4), 5), Clock::now()));
    ASSERT_TRUE(h.sched.add(2, make_request(iota_tokens(4, 50), 5), Clock::now()));
    (void)h.sched.schedule(Clock::now()); // plan discarded, as if the backend failed
    h.sched.on_kv_exhausted();
    EXPECT_EQ(h.sched.num_running(), 1u);
    EXPECT_EQ(h.sched.num_waiting(), 1u);
    h.run();
    EXPECT_EQ(h.finished.size(), 2u);
}
