#include "avis/bench/runner.hpp"

#include <memory>
#include <optional>
#include <thread>

namespace avis::bench {

ScenarioResult run_scenario(engine::Executor& executor, telemetry::GpuDevice* gpu, engine::ClockPolicy& policy,
                            const ScenarioConfig& config, std::span<const WorkloadItem> workload) {
    ScenarioResult result;
    result.name = config.name;
    result.policy = policy.name();
    result.requests = workload.size();

    std::optional<telemetry::PowerSampler> sampler;
    std::optional<telemetry::EnergyMeter> meter;
    if (gpu) {
        sampler.emplace(*gpu, config.sample_interval);
        meter.emplace(*gpu);
        sampler->start();
        meter->begin();
    }

    engine::Engine engine(executor, config.engine, policy, gpu);
    engine.start();

    std::vector<std::future<engine::RequestOutput>> futures;
    futures.reserve(workload.size());
    const auto t0 = Clock::now();
    for (const auto& item : workload) {
        std::this_thread::sleep_until(t0 + std::chrono::duration_cast<Duration>(std::chrono::duration<double>(item.arrival_s)));
        engine::Request request;
        request.prompt = item.prompt;
        request.max_new_tokens = item.max_new_tokens;
        request.sampling = item.sampling;
        request.ignore_eos = config.ignore_eos;
        futures.push_back(engine.submit(std::move(request)));
    }

    std::vector<double> ttft, tpot, e2e;
    for (auto& f : futures) {
        const auto out = f.get();
        result.prompt_tokens += out.metrics.prompt_tokens;
        result.generated_tokens += out.metrics.generated_tokens;
        result.cached_prompt_tokens += out.metrics.cached_prompt_tokens;
        if (const auto v = out.metrics.ttft_s()) {
            ttft.push_back(*v);
        }
        if (const auto v = out.metrics.tpot_s()) {
            tpot.push_back(*v);
        }
        if (const auto v = out.metrics.e2e_s()) {
            e2e.push_back(*v);
        }
    }
    result.wall_s = to_seconds(Clock::now() - t0);
    engine.stop();
    result.engine = engine.stats();

    if (sampler) {
        sampler->stop();
        result.power = sampler->samples();
        result.energy = meter->end(result.power);
        result.has_energy = true;
    }
    result.ttft_s = report::summarize(ttft);
    result.tpot_s = report::summarize(tpot);
    result.e2e_s = report::summarize(e2e);
    return result;
}

} // namespace avis::bench
