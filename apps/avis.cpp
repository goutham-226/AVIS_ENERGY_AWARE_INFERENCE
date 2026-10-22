// avis: energy-aware LLM inference CLI.
//
//   avis info
//   avis run   --prompt TEXT [--system TEXT] [--model PATH] [--max-tokens N] ...
//   avis bench [--prompts FILE] [--requests N] [--rate R] [--scenarios LIST] ...
//
// Run `avis help` for every option.

#include "avis/bench/runner.hpp"
#include "avis/engine/clock_policy.hpp"
#include "avis/engine/engine.hpp"
#include "avis/llama/llama_executor.hpp"
#include "avis/report/chart.hpp"
#include "avis/telemetry/power_sampler.hpp"
#include "avis/workload/workload.hpp"

#ifdef AVIS_HAS_NVML
#include "avis/telemetry/nvml_device.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using namespace avis;

namespace {

constexpr std::string_view kDefaultModel = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 2; i < argc; ++i) {
            std::string_view arg = argv[i];
            if (!arg.starts_with("--")) {
                throw std::invalid_argument(std::format("unexpected argument '{}'", arg));
            }
            arg.remove_prefix(2);
            if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--")) {
                values_[std::string(arg)] = argv[++i];
            } else {
                values_[std::string(arg)] = "";
            }
        }
    }

    [[nodiscard]] bool has(const std::string& key) const { return values_.contains(key); }

    [[nodiscard]] std::string str(const std::string& key, std::string_view fallback) const {
        used_.insert(key);
        const auto it = values_.find(key);
        return it == values_.end() ? std::string(fallback) : it->second;
    }

    template <class T>
    [[nodiscard]] T num(const std::string& key, T fallback) const {
        used_.insert(key);
        const auto it = values_.find(key);
        if (it == values_.end()) {
            return fallback;
        }
        T out{};
        const auto& s = it->second;
        const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
        if (ec != std::errc{} || ptr != s.data() + s.size()) {
            throw std::invalid_argument(std::format("--{} expects a number, got '{}'", key, s));
        }
        return out;
    }

    [[nodiscard]] bool flag(const std::string& key) const {
        used_.insert(key);
        return values_.contains(key);
    }

    void reject_unknown() const {
        for (const auto& [key, _] : values_) {
            if (!used_.contains(key)) {
                throw std::invalid_argument(std::format("unknown option --{}", key));
            }
        }
    }

private:
    std::map<std::string, std::string> values_;
    mutable std::set<std::string> used_;
};

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------
std::unique_ptr<telemetry::GpuDevice> open_gpu() {
#ifdef AVIS_HAS_NVML
    if (auto gpu = telemetry::NvmlDevice::open(0)) {
        return std::move(*gpu);
    } else {
        std::println(stderr, "avis: GPU telemetry unavailable: {}", gpu.error().message);
    }
#endif
    return nullptr;
}

struct EngineSetup {
    std::uint32_t n_ctx = 16384;
    std::size_t max_running = 8;
    std::size_t step_tokens = 512;
    std::size_t prefill_chunk = 256;
    std::size_t block_size = 16;
    std::size_t cache_slots = 8; // extra backend slots that keep finished prompts cached

    [[nodiscard]] std::uint32_t num_slots() const { return static_cast<std::uint32_t>(max_running + cache_slots); }

    [[nodiscard]] engine::EngineConfig engine_config(std::size_t running) const {
        return {
            .scheduler = {.max_running = running,
                          .max_step_tokens = step_tokens,
                          .prefill_chunk = prefill_chunk,
                          .max_seq_len = n_ctx},
            .kv = {.num_blocks = n_ctx / block_size,
                   .block_size = block_size,
                   .num_slots = num_slots(),
                   .enable_prefix_caching = true},
        };
    }
};

EngineSetup engine_setup(const Args& args) {
    EngineSetup s;
    s.n_ctx = args.num<std::uint32_t>("ctx", s.n_ctx);
    s.max_running = args.num<std::size_t>("max-running", s.max_running);
    s.step_tokens = args.num<std::size_t>("step-tokens", s.step_tokens);
    s.prefill_chunk = std::min(s.step_tokens, args.num<std::size_t>("prefill-chunk", s.prefill_chunk));
    return s;
}

std::unique_ptr<llama::LlamaExecutor> load_model(const Args& args, const EngineSetup& setup) {
    llama::LlamaConfig cfg;
    cfg.model_path = args.str("model", kDefaultModel);
    cfg.n_ctx = setup.n_ctx;
    cfg.n_batch = static_cast<std::uint32_t>(setup.step_tokens);
    cfg.n_seq_max = setup.num_slots();
    cfg.n_gpu_layers = args.num<std::int32_t>("gpu-layers", 99);
    auto exec = llama::LlamaExecutor::load(cfg);
    if (!exec) {
        throw std::runtime_error(exec.error().message);
    }
    return std::move(*exec);
}

engine::EnergyAwareConfig energy_config(const Args& args) {
    engine::EnergyAwareConfig cfg;
    cfg.latency_budget = args.num<double>("latency-budget", cfg.latency_budget);
    cfg.window = std::chrono::milliseconds(args.num<int>("window-ms", static_cast<int>(cfg.window.count())));
    cfg.floor_fraction = args.num<double>("clock-floor", cfg.floor_fraction);
    return cfg;
}

std::string_view color(bool on, std::string_view code) { return on ? code : std::string_view{}; }

// ---------------------------------------------------------------------------
// avis info
// ---------------------------------------------------------------------------
int cmd_info(const Args& args) {
    args.reject_unknown();
    auto gpu = open_gpu();
    if (!gpu) {
        std::println("No NVIDIA GPU telemetry available.");
        return 1;
    }
    const auto reading = gpu->read();
    const auto clocks = gpu->supported_sm_clocks();
    std::println("GPU            : {}", gpu->name());
    if (reading) {
        std::println("Power          : {:.1f} W", reading->power_w);
        std::println("SM / mem clock : {} / {} MHz", reading->sm_clock_mhz, reading->mem_clock_mhz);
        std::println("Temperature    : {} C", reading->temp_c);
    }
    std::println("Energy counter : {}", gpu->energy_mj() ? "yes" : "no (sampled power is integrated)");
    if (!clocks.empty()) {
        std::println("SM clocks      : {} - {} MHz ({} steps)", clocks.front(), clocks.back(), clocks.size());
    }
    std::println("Clock control  : {}", gpu->can_control_clocks()
                                            ? "yes"
                                            : "no (needs root: run with sudo for the energy-aware policy)");
    return 0;
}

// ---------------------------------------------------------------------------
// avis run
// ---------------------------------------------------------------------------
int cmd_run(const Args& args) {
    const bool use_color = !args.flag("no-color") && isatty(STDOUT_FILENO);
    const auto setup = engine_setup(args);
    const std::string prompt = args.str("prompt", "");
    if (prompt.empty()) {
        throw std::invalid_argument("--prompt is required");
    }
    const std::string system = args.str("system", "You are AVIS, a concise technical assistant.");
    const auto max_tokens = args.num<std::size_t>("max-tokens", 512);
    engine::SamplingParams sampling{
        .temperature = args.num<float>("temp", 0.7f),
        .top_p = args.num<float>("top-p", 0.9f),
        .top_k = args.num<std::int32_t>("top-k", 40),
        .seed = args.num<std::uint32_t>("seed", 42),
    };
    const bool energy_aware = args.flag("energy-aware");
    const auto policy_cfg = energy_config(args);
    const bool graph = !args.flag("no-graph");

    auto exec = load_model(args, setup);
    args.reject_unknown();
    auto gpu = open_gpu();

    std::unique_ptr<engine::ClockPolicy> policy;
    if (energy_aware && gpu && !gpu->supported_sm_clocks().empty()) {
        policy = std::make_unique<engine::EnergyAwarePolicy>(gpu->supported_sm_clocks(), policy_cfg);
    } else {
        policy = std::make_unique<engine::FixedClockPolicy>();
    }

    auto tokens = exec->tokenize(exec->format_chat(system, prompt));
    if (!tokens) {
        throw std::runtime_error(tokens.error().message);
    }

    std::optional<telemetry::PowerSampler> sampler;
    std::optional<telemetry::EnergyMeter> meter;
    if (gpu) {
        sampler.emplace(*gpu, std::chrono::milliseconds{10});
        meter.emplace(*gpu);
        sampler->start();
        meter->begin();
    }

    engine::Engine engine(*exec, setup.engine_config(1), *policy, gpu.get());
    engine::Request request;
    request.prompt = std::move(*tokens);
    request.max_new_tokens = max_tokens;
    request.sampling = sampling;
    request.on_text = [&](std::string_view piece) {
        std::print("{}{}{}", color(use_color, "\033[33m"), piece, color(use_color, "\033[0m"));
        std::fflush(stdout);
    };
    std::println("{}{}{}", color(use_color, "\033[2m"), exec->description(), color(use_color, "\033[0m"));
    auto future = engine.submit(std::move(request));
    while (engine.step()) {
    }
    const auto out = future.get();
    std::println("\n");

    const auto& m = out.metrics;
    std::println("{}Latency{}    TTFT {:.3f} s · TPOT {:.2f} ms · total {:.2f} s · {} prompt + {} generated tokens",
                 color(use_color, "\033[36m"), color(use_color, "\033[0m"), m.ttft_s().value_or(0),
                 m.tpot_s().value_or(0) * 1e3, m.e2e_s().value_or(0), m.prompt_tokens, m.generated_tokens);
    if (sampler) {
        sampler->stop();
        const auto samples = sampler->samples();
        const auto energy = meter->end(samples);
        std::println("{}Energy{}     {:.1f} J · {:.1f} W avg · {:.3f} J/token · {:.2f} tokens/J ({})",
                     color(use_color, "\033[36m"), color(use_color, "\033[0m"), energy.joules, energy.avg_power_w,
                     energy.joules / std::max<double>(1, m.generated_tokens),
                     m.generated_tokens / std::max(energy.joules, 1e-9),
                     energy.from_counter ? "NVML energy counter" : "integrated samples");
        if (graph && samples.size() > 1) {
            report::Series power{"GPU power (W)", {}};
            double joules = 0;
            report::Series cumulative{"energy (J)", {}};
            for (std::size_t i = 0; i < samples.size(); ++i) {
                power.points.emplace_back(samples[i].t_s, samples[i].reading.power_w);
                if (i > 0) {
                    joules += 0.5 * (samples[i].reading.power_w + samples[i - 1].reading.power_w) *
                              (samples[i].t_s - samples[i - 1].t_s);
                }
                cumulative.points.emplace_back(samples[i].t_s, joules);
            }
            std::println();
            std::print("{}", report::render_line_chart(std::vector{power},
                                                       {.title = "GPU power during generation", .x_label = "time (s)",
                                                        .y_label = "W", .color = use_color, .y_min = 0.0}));
            std::println();
            std::print("{}", report::render_line_chart(std::vector{cumulative},
                                                       {.height = 8, .title = "Cumulative energy", .x_label = "time (s)",
                                                        .y_label = "J", .color = use_color, .y_min = 0.0}));
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// avis bench
// ---------------------------------------------------------------------------
struct ScenarioSpec {
    std::string name;
    std::size_t max_running;
    std::optional<std::uint32_t> fixed_cap;
    bool energy_aware = false;
};

std::vector<ScenarioSpec> parse_scenarios(std::string_view list, std::size_t max_running) {
    std::vector<ScenarioSpec> specs;
    for (const auto part : std::views::split(list, ',')) {
        const std::string name(part.begin(), part.end());
        if (name == "sequential") {
            specs.push_back({name, 1, std::nullopt, false});
        } else if (name == "batched") {
            specs.push_back({name, max_running, std::nullopt, false});
        } else if (name == "avis") {
            specs.push_back({name, max_running, std::nullopt, true});
        } else if (name.starts_with("fixed:")) {
            std::uint32_t mhz = 0;
            const auto digits = std::string_view(name).substr(6);
            if (std::from_chars(digits.data(), digits.data() + digits.size(), mhz).ec != std::errc{}) {
                throw std::invalid_argument(std::format("bad scenario '{}'", name));
            }
            specs.push_back({name, max_running, mhz, false});
        } else {
            throw std::invalid_argument(std::format(
                "unknown scenario '{}' (expected sequential, batched, avis or fixed:<MHz>)", name));
        }
    }
    return specs;
}

void cool_down(telemetry::GpuDevice* gpu, double seconds) {
    if (seconds <= 0) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    if (!gpu) {
        return;
    }
    // Then wait (bounded) for the GPU to approach idle temperature.
    for (int i = 0; i < 60; ++i) {
        const auto r = gpu->read();
        if (!r || r->temp_c <= 50) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

double pct_change(double now, double base) { return base != 0.0 ? 100.0 * (now - base) / base : 0.0; }

struct Aggregate {
    std::string name;
    std::string policy;
    std::vector<bench::ScenarioResult> runs;

    template <class F>
    [[nodiscard]] double mean(F&& f) const {
        double sum = 0;
        for (const auto& r : runs) {
            sum += f(r);
        }
        return runs.empty() ? 0.0 : sum / static_cast<double>(runs.size());
    }
};

void write_power_csv(const fs::path& path, const bench::ScenarioResult& r) {
    std::ofstream out(path);
    out << "t_s,power_w,sm_clock_mhz,mem_clock_mhz,util_pct,temp_c,mem_used_mb\n";
    for (const auto& s : r.power) {
        out << std::format("{:.4f},{:.2f},{},{},{},{},{}\n", s.t_s, s.reading.power_w, s.reading.sm_clock_mhz,
                           s.reading.mem_clock_mhz, s.reading.util_pct, s.reading.temp_c, s.reading.mem_used_mb);
    }
}

int cmd_bench(const Args& args) {
    const bool use_color = !args.flag("no-color") && isatty(STDOUT_FILENO);
    const auto setup = engine_setup(args);
    const auto prompts_path = args.str("prompts", "prompts.txt");
    const auto num_requests = args.num<std::size_t>("requests", 32);
    const auto rate = args.num<double>("rate", 2.0);
    const auto max_tokens = args.num<std::size_t>("max-tokens", 128);
    const auto seed = args.num<std::uint64_t>("seed", 42);
    const auto repeat = std::max<std::size_t>(1, args.num<std::size_t>("repeat", 1));
    const auto cooldown = args.num<double>("cooldown", 5.0);
    const fs::path out_dir = args.str("out", "results");
    const bool respect_eos = args.flag("respect-eos");
    const auto policy_cfg = energy_config(args);
    const auto scenarios = parse_scenarios(args.str("scenarios", "sequential,batched,avis"), setup.max_running);

    auto specs = workload::load_prompts(prompts_path);
    if (!specs) {
        throw std::runtime_error(specs.error().message);
    }
    if (specs->empty()) {
        throw std::runtime_error("the prompts file has no requests");
    }

    auto exec = load_model(args, setup);
    args.reject_unknown();
    auto gpu = open_gpu();
    const bool can_clock = gpu && gpu->can_control_clocks();

    std::println("Model     : {}", exec->description());
    std::println("GPU       : {} · clock control {}", gpu ? gpu->name() : "none",
                 can_clock ? "available" : "UNAVAILABLE (run as root for clock policies)");
    std::println("Workload  : {} requests · Poisson {:.2f} req/s · {} new tokens each · {} prompts",
                 num_requests, rate, max_tokens, specs->size());

    // Build the workload once so every scenario replays identical requests.
    const auto arrivals = workload::poisson_arrivals(num_requests, rate, seed);
    std::vector<bench::WorkloadItem> items;
    items.reserve(num_requests);
    for (std::size_t i = 0; i < num_requests; ++i) {
        const auto& spec = (*specs)[i % specs->size()];
        auto tokens = exec->tokenize(exec->format_chat(spec.system_prompt, spec.prompt));
        if (!tokens) {
            throw std::runtime_error(tokens.error().message);
        }
        items.push_back({.prompt = std::move(*tokens),
                         .arrival_s = arrivals[i],
                         .max_new_tokens = std::min(max_tokens, spec.max_new_tokens),
                         .sampling = {.temperature = spec.temperature,
                                      .top_p = spec.top_p,
                                      .top_k = spec.top_k,
                                      .seed = static_cast<std::uint32_t>(seed + i)}});
    }

    // Warm up kernels and the allocator so the first scenario is not penalised.
    {
        std::println("Warming up...");
        engine::FixedClockPolicy none;
        const auto warm = std::span(items).first(std::min<std::size_t>(4, items.size()));
        std::vector<bench::WorkloadItem> burst(warm.begin(), warm.end());
        for (auto& w : burst) {
            w.arrival_s = 0;
            w.max_new_tokens = 16;
        }
        (void)bench::run_scenario(*exec, gpu.get(), none, {.name = "warmup", .engine = setup.engine_config(setup.max_running)},
                                  burst);
    }

    std::vector<Aggregate> aggregates(scenarios.size());
    for (std::size_t r = 0; r < repeat; ++r) {
        // Rotate the order each repeat so thermal drift does not favour one scenario.
        for (std::size_t k = 0; k < scenarios.size(); ++k) {
            const std::size_t idx = (k + r) % scenarios.size();
            const auto& sc = scenarios[idx];
            std::unique_ptr<engine::ClockPolicy> policy;
            if (sc.energy_aware) {
                if (!can_clock) {
                    std::println("  note: '{}' cannot change clocks without root; it will run at default clocks",
                                 sc.name);
                }
                const auto clocks = gpu ? gpu->supported_sm_clocks() : std::vector<std::uint32_t>{};
                if (clocks.empty()) {
                    policy = std::make_unique<engine::FixedClockPolicy>();
                } else {
                    policy = std::make_unique<engine::EnergyAwarePolicy>(clocks, policy_cfg);
                }
            } else {
                policy = std::make_unique<engine::FixedClockPolicy>(sc.fixed_cap);
            }
            cool_down(gpu.get(), cooldown);
            std::println("[{}/{}] {} (max_running={}, policy={})", r * scenarios.size() + k + 1,
                         repeat * scenarios.size(), sc.name, sc.max_running, policy->name());
            auto result = bench::run_scenario(
                *exec, gpu.get(), *policy,
                {.name = sc.name, .engine = setup.engine_config(sc.max_running), .ignore_eos = !respect_eos},
                items);
            std::println("        {:.1f} s · {:.1f} tok/s · {:.1f} J · {:.4f} J/tok · E2E p50 {:.2f} s",
                         result.wall_s, result.tokens_per_s(), result.energy.joules, result.joules_per_token(),
                         result.e2e_s.p50);
            aggregates[idx].name = sc.name;
            aggregates[idx].policy = result.policy;
            aggregates[idx].runs.push_back(std::move(result));
        }
    }

    // ---- Summary table ----
    const auto c = [&](std::string_view code) { return color(use_color, code); };
    std::println("\n{}Results (mean of {} run{}){}", c("\033[1m"), repeat, repeat > 1 ? "s" : "", c("\033[0m"));
    std::println("{:<14} {:>8} {:>9} {:>8} {:>8} {:>10} {:>9} {:>9} {:>9} {:>9} {:>7}", "scenario", "tok/s",
                 "energy J", "avg W", "J/tok", "tok/J", "TTFT p50", "TPOT p50", "E2E p50", "E2E p95", "clk Δ");
    for (const auto& a : aggregates) {
        std::println("{:<14} {:>8.1f} {:>9.1f} {:>8.1f} {:>8.4f} {:>10.2f} {:>8.3f}s {:>7.1f}ms {:>8.2f}s {:>8.2f}s {:>7.1f}",
                     a.name, a.mean([](auto& r) { return r.tokens_per_s(); }),
                     a.mean([](auto& r) { return r.energy.joules; }),
                     a.mean([](auto& r) { return r.energy.avg_power_w; }),
                     a.mean([](auto& r) { return r.joules_per_token(); }),
                     a.mean([](auto& r) { return r.tokens_per_joule(); }), a.mean([](auto& r) { return r.ttft_s.p50; }),
                     a.mean([](auto& r) { return r.tpot_s.p50 * 1e3; }), a.mean([](auto& r) { return r.e2e_s.p50; }),
                     a.mean([](auto& r) { return r.e2e_s.p95; }),
                     a.mean([](auto& r) { return static_cast<double>(r.engine.clock_events.size()); }));
    }

    // ---- Relative to the batched baseline ----
    const auto base_it = std::ranges::find(aggregates, std::string("batched"), &Aggregate::name);
    const Aggregate& base = base_it != aggregates.end() ? *base_it : aggregates.front();
    std::println("\n{}Change versus '{}'{}", c("\033[1m"), base.name, c("\033[0m"));
    std::println("{:<14} {:>12} {:>12} {:>12} {:>12} {:>12}", "scenario", "J/token", "throughput", "TPOT p50",
                 "E2E p50", "E2E p95");
    for (const auto& a : aggregates) {
        if (&a == &base) {
            continue;
        }
        const auto delta = [&](auto f) { return pct_change(a.mean(f), base.mean(f)); };
        std::println("{:<14} {:>+11.1f}% {:>+11.1f}% {:>+11.1f}% {:>+11.1f}% {:>+11.1f}%", a.name,
                     delta([](auto& r) { return r.joules_per_token(); }), delta([](auto& r) { return r.tokens_per_s(); }),
                     delta([](auto& r) { return r.tpot_s.p50; }), delta([](auto& r) { return r.e2e_s.p50; }),
                     delta([](auto& r) { return r.e2e_s.p95; }));
    }

    // ---- Graphs ----
    std::vector<report::Bar> jpt, tps;
    std::vector<report::Series> power, clocks;
    for (const auto& a : aggregates) {
        jpt.push_back({a.name, a.mean([](auto& r) { return r.joules_per_token(); })});
        tps.push_back({a.name, a.mean([](auto& r) { return r.tokens_per_s(); })});
        const auto& first = a.runs.front();
        report::Series p{a.name, {}};
        report::Series clk{a.name, {}};
        for (const auto& s : first.power) {
            p.points.emplace_back(s.t_s, s.reading.power_w);
            clk.points.emplace_back(s.t_s, s.reading.sm_clock_mhz);
        }
        power.push_back(std::move(p));
        clocks.push_back(std::move(clk));
    }
    std::println();
    std::print("{}", report::render_bar_chart(jpt, "Energy per generated token (lower is better)", "J", 48, use_color));
    std::println();
    std::print("{}", report::render_bar_chart(tps, "Throughput (higher is better)", "tok/s", 48, use_color));
    if (gpu) {
        const report::ChartOptions power_opts{.width = 90, .height = 16, .title = "GPU power over the run",
                                              .x_label = "time (s)", .y_label = "W", .color = use_color, .y_min = 0.0};
        const report::ChartOptions clock_opts{.width = 90, .height = 10, .title = "SM clock over the run",
                                              .x_label = "time (s)", .y_label = "MHz", .color = use_color,
                                              .y_min = 0.0};
        std::println();
        std::print("{}", report::render_line_chart(power, power_opts));
        std::println();
        std::print("{}", report::render_line_chart(clocks, clock_opts));

        fs::create_directories(out_dir);
        auto svg_power = power_opts;
        svg_power.color = false;
        std::ofstream(out_dir / "power.svg") << report::render_svg_line_chart(power, svg_power);
        auto svg_clock = clock_opts;
        svg_clock.color = false;
        std::ofstream(out_dir / "sm_clock.svg") << report::render_svg_line_chart(clocks, svg_clock);
    }

    // ---- Files ----
    fs::create_directories(out_dir);
    std::ofstream summary(out_dir / "summary.csv");
    summary << "scenario,run,policy,requests,generated_tokens,wall_s,tokens_per_s,energy_j,avg_power_w,"
               "joules_per_token,tokens_per_joule,ttft_p50_s,ttft_p95_s,tpot_p50_s,tpot_p95_s,e2e_p50_s,e2e_p95_s,"
               "clock_changes,preemptions,prefix_hit_tokens,energy_source\n";
    for (const auto& a : aggregates) {
        for (std::size_t i = 0; i < a.runs.size(); ++i) {
            const auto& r = a.runs[i];
            summary << std::format("{},{},{},{},{},{:.3f},{:.2f},{:.2f},{:.2f},{:.5f},{:.3f},{:.4f},{:.4f},{:.5f},{:.5f},"
                                   "{:.3f},{:.3f},{},{},{},{}\n",
                                   a.name, i, r.policy, r.requests, r.generated_tokens, r.wall_s, r.tokens_per_s(),
                                   r.energy.joules, r.energy.avg_power_w, r.joules_per_token(), r.tokens_per_joule(),
                                   r.ttft_s.p50, r.ttft_s.p95, r.tpot_s.p50, r.tpot_s.p95, r.e2e_s.p50, r.e2e_s.p95,
                                   r.engine.clock_events.size(), r.engine.preemptions, r.engine.kv.prefix_hit_tokens,
                                   r.energy.from_counter ? "nvml_counter" : "integrated");
            write_power_csv(out_dir / std::format("{}_run{}_power.csv", a.name, i), r);
        }
    }
    std::println("\nWrote {}/summary.csv, per-run power traces{}", out_dir.string(),
                 gpu ? ", power.svg and sm_clock.svg" : "");
    return 0;
}

void print_help() {
    std::println(R"(avis - energy-aware LLM inference

Usage:
  avis info                         GPU telemetry and clock-control status
  avis run   --prompt TEXT [opts]   generate one response and report its energy
  avis bench [opts]                 replay a request workload under several scenarios

Common options:
  --model PATH          GGUF model (default {})
  --ctx N               KV cache size in tokens shared by all requests (16384)
  --max-running N       max concurrent sequences in a batch (8)
  --step-tokens N       max tokens per forward pass (512)
  --prefill-chunk N     max prompt tokens per sequence per step (256)
  --gpu-layers N        layers to offload (99)
  --no-color            disable ANSI colours

run:
  --system TEXT --max-tokens N (512) --temp F (0.7) --top-p F (0.9) --top-k N (40) --seed N
  --energy-aware        let the clock policy manage SM clocks (needs root)
  --no-graph            skip the power/energy graphs

bench:
  --prompts FILE        prompts file (prompts.txt)
  --requests N          number of requests (32), cycling through the prompts
  --rate R              Poisson arrival rate in requests/s (2.0; 0 = all at once)
  --max-tokens N        new tokens per request (128)
  --scenarios LIST      comma list of sequential, batched, avis, fixed:<MHz>
                        (default sequential,batched,avis)
  --latency-budget F    allowed step slowdown for the avis policy (0.10)
  --window-ms N         clock policy control window (300)
  --clock-floor F       lowest clock cap as a fraction of max (0.40)
  --repeat N            repetitions, scenario order rotated each time (1)
  --cooldown S          idle seconds between scenarios (5)
  --respect-eos         stop at end-of-generation instead of fixed lengths
  --seed N              workload seed (42)
  --out DIR             output directory for CSV/SVG (results))",
                 kDefaultModel);
}

} // namespace

int main(int argc, char** argv) {
    const std::string command = argc > 1 ? argv[1] : "help";
    try {
        const Args args(argc, argv);
        if (command == "info") {
            return cmd_info(args);
        }
        if (command == "run") {
            return cmd_run(args);
        }
        if (command == "bench") {
            return cmd_bench(args);
        }
        print_help();
        return command == "help" || command == "--help" || command == "-h" ? 0 : 2;
    } catch (const std::exception& e) {
        std::println(stderr, "avis: {}", e.what());
        return 1;
    }
}
