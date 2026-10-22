// CPU-side hot paths of the engine. The scheduler runs once per forward pass,
// so its cost must stay far below a GPU step (milliseconds).

#include "avis/engine/clock_policy.hpp"
#include "avis/engine/scheduler.hpp"
#include "avis/kv/block_manager.hpp"
#include "avis/report/chart.hpp"

#include <benchmark/benchmark.h>

#include <cmath>
#include <numeric>

using namespace avis;

namespace {

std::vector<Token> iota_tokens(std::size_t n, Token start = 0) {
    std::vector<Token> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

} // namespace

// Full lifecycle of one sequence: admit, reserve, commit and release with caching.
void BM_BlockManagerLifecycle(benchmark::State& state) {
    const auto prompt_len = static_cast<std::size_t>(state.range(0));
    kv::BlockManager bm({.num_blocks = 4096, .block_size = 16, .num_slots = 64});
    Token salt = 0;
    for (auto _ : state) {
        auto prompt = iota_tokens(prompt_len, salt += 7919); // unique prompt: no cache hits
        auto adm = bm.admit(prompt);
        benchmark::DoNotOptimize(bm.reserve(adm->slot, prompt.size()));
        bm.commit(adm->slot, prompt, prompt.size());
        bm.release(adm->slot, true);
        benchmark::DoNotOptimize(bm.take_ops());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BlockManagerLifecycle)->Arg(128)->Arg(512)->Arg(2048);

// Admission of a prompt that shares a long cached prefix.
void BM_PrefixCacheHit(benchmark::State& state) {
    const auto prefix_len = static_cast<std::size_t>(state.range(0));
    kv::BlockManager bm({.num_blocks = 4096, .block_size = 16, .num_slots = 64});
    const auto system = iota_tokens(prefix_len);
    auto donor_prompt = system;
    donor_prompt.push_back(1);
    auto donor = bm.admit(donor_prompt);
    (void)bm.reserve(donor->slot, donor_prompt.size());
    bm.commit(donor->slot, donor_prompt, donor_prompt.size());
    auto prompt = system;
    prompt.push_back(2);
    for (auto _ : state) {
        auto adm = bm.admit(prompt);
        benchmark::DoNotOptimize(adm);
        bm.release(adm->slot, false);
        benchmark::DoNotOptimize(bm.take_ops());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_PrefixCacheHit)->Arg(256)->Arg(1024)->Arg(4096);

// One scheduling iteration with `n` decoding sequences.
void BM_SchedulerDecodeStep(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    kv::BlockManager bm({.num_blocks = 1 << 16, .block_size = 16, .num_slots = n + 8});
    engine::Scheduler sched({.max_running = n, .max_step_tokens = 4096, .prefill_chunk = 512, .max_seq_len = 1 << 20},
                            bm);
    for (std::size_t i = 0; i < n; ++i) {
        engine::Request r;
        r.prompt = iota_tokens(64, static_cast<Token>(i * 1000));
        r.max_new_tokens = 1 << 30;
        (void)sched.add(i, std::move(r), Clock::now());
    }
    const auto never_eog = [](Token) { return false; };
    std::vector<Token> sampled;
    for (auto _ : state) {
        auto plan = sched.schedule(Clock::now());
        sampled.assign(plan.chunks.size(), 42);
        auto outcome = sched.complete(plan, sampled, never_eog, Clock::now());
        benchmark::DoNotOptimize(outcome);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
    state.SetLabel("items = sequence-steps");
}
BENCHMARK(BM_SchedulerDecodeStep)->RangeMultiplier(4)->Range(1, 256);

void BM_EnergyPolicyStep(benchmark::State& state) {
    std::vector<std::uint32_t> clocks;
    for (std::uint32_t c = 210; c <= 3090; c += 15) {
        clocks.push_back(c);
    }
    engine::EnergyAwarePolicy policy(clocks);
    auto now = Clock::now();
    for (auto _ : state) {
        now += std::chrono::milliseconds(12);
        benchmark::DoNotOptimize(policy.on_step(
            {.num_tokens = 8, .num_seqs = 8, .step_s = 0.012, .waiting = 0, .running = 8, .max_running = 8}, now));
    }
}
BENCHMARK(BM_EnergyPolicyStep);

void BM_RenderPowerChart(benchmark::State& state) {
    report::Series s{"power", {}};
    for (int i = 0; i < 5000; ++i) {
        s.points.emplace_back(i * 0.02, 60 + 30 * std::sin(i * 0.01));
    }
    const std::vector series{s};
    for (auto _ : state) {
        benchmark::DoNotOptimize(report::render_line_chart(series, {.width = 90, .height = 16, .color = false}));
    }
}
BENCHMARK(BM_RenderPowerChart);
