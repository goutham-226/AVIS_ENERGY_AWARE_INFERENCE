#pragma once

#include "avis/engine/engine.hpp"
#include "avis/report/stats.hpp"
#include "avis/telemetry/power_sampler.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace avis::bench {

struct WorkloadItem {
    std::vector<Token> prompt;
    double arrival_s = 0.0;
    std::size_t max_new_tokens = 128;
    engine::SamplingParams sampling;
};

struct ScenarioConfig {
    std::string name;
    engine::EngineConfig engine;
    bool ignore_eos = true; // fixed output lengths make energy per token comparable
    std::chrono::milliseconds sample_interval{20};
};

struct ScenarioResult {
    std::string name;
    std::string policy;
    double wall_s = 0.0;
    std::size_t requests = 0;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t cached_prompt_tokens = 0;
    bool has_energy = false;
    telemetry::EnergyReport energy;
    report::Summary ttft_s;
    report::Summary tpot_s;
    report::Summary e2e_s;
    std::vector<telemetry::PowerSample> power;
    engine::EngineStats engine;

    [[nodiscard]] double tokens_per_s() const { return wall_s > 0 ? generated_tokens / wall_s : 0.0; }
    [[nodiscard]] double joules_per_token() const {
        return has_energy && generated_tokens ? energy.joules / static_cast<double>(generated_tokens) : 0.0;
    }
    [[nodiscard]] double tokens_per_joule() const {
        return has_energy && energy.joules > 0 ? static_cast<double>(generated_tokens) / energy.joules : 0.0;
    }
};

// Replays `workload` (requests submitted at their arrival offsets) through a
// fresh engine and measures energy, throughput and latency.
[[nodiscard]] ScenarioResult run_scenario(engine::Executor& executor, telemetry::GpuDevice* gpu,
                                          engine::ClockPolicy& policy, const ScenarioConfig& config,
                                          std::span<const WorkloadItem> workload);

} // namespace avis::bench
