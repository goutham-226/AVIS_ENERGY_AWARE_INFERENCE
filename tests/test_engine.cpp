#include "avis/engine/engine.hpp"
#include "support/fake_gpu.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <format>
#include <numeric>
#include <set>
#include <thread>

using namespace avis;
using namespace avis::engine;

namespace {

// Deterministic executor: the next token of a sequence is 1000 + its length.
class FakeExecutor final : public Executor {
public:
    Result<std::vector<Token>> execute(const StepPlan& plan) override {
        ops.insert(ops.end(), plan.kv_ops.begin(), plan.kv_ops.end());
        ++calls;
        if (fail_kv_times > 0) {
            --fail_kv_times;
            return make_error("no KV slot", ErrorCode::kv_exhausted);
        }
        if (fail_hard) {
            return make_error("device lost");
        }
        max_batch = std::max(max_batch, plan.chunks.size());
        std::vector<Token> out;
        for (const auto& c : plan.chunks) {
            if (c.sample) {
                out.push_back(static_cast<Token>(1000 + c.seq->tokens.size()));
            }
        }
        return out;
    }
    void release(RequestId id) override { released.insert(id); }
    void reset() override { ++resets; }
    bool is_eog(Token t) const override { return t == eog; }
    std::string token_to_piece(Token t) const override { return std::format("<{}>", t); }

    std::vector<kv::KvOp> ops;
    std::set<RequestId> released;
    std::size_t max_batch = 0;
    int calls = 0;
    int resets = 0;
    int fail_kv_times = 0;
    bool fail_hard = false;
    Token eog = -1;
};

EngineConfig small_engine() {
    return {.scheduler = {.max_running = 4, .max_step_tokens = 128, .prefill_chunk = 64},
            .kv = {.num_blocks = 128, .block_size = 8, .num_slots = 8}};
}

Request make_request(std::size_t prompt_len, std::size_t max_new, Token start = 0) {
    Request r;
    r.prompt.resize(prompt_len);
    std::iota(r.prompt.begin(), r.prompt.end(), start);
    r.max_new_tokens = max_new;
    return r;
}

} // namespace

TEST(Engine, SynchronousStepCompletesRequest) {
    FakeExecutor exec;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    auto future = engine.submit(make_request(5, 3));
    while (engine.step()) {
    }
    const auto out = future.get();
    EXPECT_EQ(out.tokens, (std::vector<Token>{1005, 1006, 1007}));
    EXPECT_EQ(out.text, "<1005><1006><1007>");
    EXPECT_EQ(out.finish_reason, FinishReason::length);
    EXPECT_EQ(out.metrics.generated_tokens, 3u);
    EXPECT_TRUE(exec.released.contains(out.id));
    EXPECT_EQ(engine.stats().completed, 1u);
    EXPECT_EQ(exec.resets, 1) << "a new engine must start from a clean backend";
}

TEST(Engine, ServesConcurrentSubmittersInBatches) {
    FakeExecutor exec;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    engine.start();

    constexpr int kThreads = 4;
    constexpr int kPerThread = 5;
    std::vector<std::future<RequestOutput>> futures(kThreads * kPerThread);
    std::vector<std::jthread> submitters;
    for (int t = 0; t < kThreads; ++t) {
        submitters.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                futures[t * kPerThread + i] = engine.submit(make_request(6 + i, 8, static_cast<Token>(t * 100)));
            }
        });
    }
    submitters.clear(); // join
    engine.wait_idle();
    for (std::size_t i = 0; i < futures.size(); ++i) {
        const auto out = futures[i].get();
        EXPECT_EQ(out.tokens.size(), 8u);
        EXPECT_EQ(out.tokens.front(), static_cast<Token>(1000 + 6 + i % kPerThread));
    }
    EXPECT_GT(exec.max_batch, 1u) << "requests should share forward passes";
    EXPECT_EQ(engine.stats().completed, futures.size());
    engine.stop();
}

TEST(Engine, StreamsTextAsItIsGenerated) {
    FakeExecutor exec;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    std::string streamed;
    auto req = make_request(4, 3);
    req.on_text = [&](std::string_view piece) { streamed += piece; };
    auto future = engine.submit(std::move(req));
    while (engine.step()) {
    }
    EXPECT_EQ(streamed, future.get().text);
}

TEST(Engine, StopsAtEndOfGeneration) {
    FakeExecutor exec;
    exec.eog = 1006;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    auto future = engine.submit(make_request(4, 50));
    while (engine.step()) {
    }
    const auto out = future.get();
    EXPECT_EQ(out.finish_reason, FinishReason::stop);
    EXPECT_EQ(out.tokens, (std::vector<Token>{1004, 1005}));
}

TEST(Engine, InvalidRequestFailsItsFuture) {
    FakeExecutor exec;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    auto future = engine.submit(make_request(0, 3));
    engine.step();
    EXPECT_THROW(future.get(), std::invalid_argument);
}

TEST(Engine, RecoversFromKvExhaustion) {
    FakeExecutor exec;
    exec.fail_kv_times = 2;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    auto a = engine.submit(make_request(5, 4));
    auto b = engine.submit(make_request(5, 4, 50));
    while (engine.step()) {
    }
    EXPECT_EQ(a.get().tokens.size(), 4u);
    EXPECT_EQ(b.get().tokens.size(), 4u);
    EXPECT_EQ(engine.stats().kv_exhausted, 2u);
    EXPECT_GE(engine.stats().preemptions, 1u);
}

TEST(Engine, BackendFailurePropagatesToFutures) {
    FakeExecutor exec;
    exec.fail_hard = true;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    engine.start();
    auto future = engine.submit(make_request(5, 4));
    EXPECT_THROW(future.get(), std::runtime_error);
    engine.wait_idle();
}

TEST(Engine, AppliesAndResetsClockCap) {
    FakeExecutor exec;
    test_support::FakeGpu gpu;
    {
        FixedClockPolicy policy(1800);
        Engine engine(exec, small_engine(), policy, &gpu);
        EXPECT_EQ(gpu.cap_mhz(), 1800u);
        const auto stats = engine.stats();
        ASSERT_EQ(stats.clock_events.size(), 1u);
        EXPECT_TRUE(stats.clock_events[0].applied);
    }
    EXPECT_EQ(gpu.cap_mhz(), test_support::FakeGpu::kMaxClock);
}

TEST(Engine, RecordsClockRequestsWithoutPermission) {
    FakeExecutor exec;
    test_support::FakeGpu gpu(/*can_control=*/false);
    FixedClockPolicy policy(1800);
    Engine engine(exec, small_engine(), policy, &gpu);
    const auto stats = engine.stats();
    ASSERT_EQ(stats.clock_events.size(), 1u);
    EXPECT_FALSE(stats.clock_events[0].applied);
    EXPECT_EQ(gpu.lock_calls(), 0);
}

TEST(Engine, SharedSystemPromptHitsPrefixCache) {
    FakeExecutor exec;
    FixedClockPolicy policy;
    Engine engine(exec, small_engine(), policy);
    auto first = engine.submit(make_request(40, 2));
    while (engine.step()) {
    }
    auto req = make_request(40, 2);
    req.prompt.push_back(4242);
    auto second = engine.submit(std::move(req));
    while (engine.step()) {
    }
    (void)first.get();
    EXPECT_EQ(second.get().metrics.cached_prompt_tokens, 40u);
    EXPECT_GT(engine.stats().kv.prefix_hit_tokens, 0u);
}
