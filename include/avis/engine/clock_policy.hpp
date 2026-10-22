#pragma once

#include "avis/core/types.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace avis::engine {

// What the engine observed while executing one batched forward pass.
struct StepObservation {
    std::size_t num_tokens = 0; // prefill + decode tokens in the batch
    std::size_t num_seqs = 0;
    double step_s = 0.0;
    std::size_t waiting = 0; // requests queued after the step
    std::size_t running = 0;
    std::size_t max_running = 0;
};

// Chooses an SM clock cap for the GPU.
//
// With continuous batching a single forward pass mixes prefill chunks of some
// requests with decode tokens of others, so clocks cannot follow per-request
// phases. Policies instead pick one cap for the whole batch stream and revise
// it at most once per control window.
class ClockPolicy {
public:
    virtual ~ClockPolicy() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    // Cap to apply before the first step, or nullopt to leave the GPU alone.
    [[nodiscard]] virtual std::optional<std::uint32_t> initial_cap() const = 0;
    // Returns a new cap when it should change.
    [[nodiscard]] virtual std::optional<std::uint32_t> on_step(const StepObservation& obs, TimePoint now) = 0;
};

// Never touches the clocks (the GPU's default boost behaviour), or holds one fixed cap.
class FixedClockPolicy final : public ClockPolicy {
public:
    explicit FixedClockPolicy(std::optional<std::uint32_t> cap_mhz = std::nullopt) : cap_(cap_mhz) {}

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] std::optional<std::uint32_t> initial_cap() const override { return cap_; }
    [[nodiscard]] std::optional<std::uint32_t> on_step(const StepObservation&, TimePoint) override {
        return std::nullopt;
    }

private:
    std::optional<std::uint32_t> cap_;
};

struct EnergyAwareConfig {
    double latency_budget = 0.10; // tolerated step-time slowdown versus max clock (0.10 = 10%)
    std::chrono::milliseconds window{300};
    std::chrono::milliseconds recalibrate_every{10'000};
    std::size_t num_levels = 10;          // clock caps between the floor and max
    double floor_fraction = 0.40;         // lowest cap as a fraction of max clock
    double saturation_threshold = 0.5;    // fraction of steps with a full batch and a queue
    std::size_t min_samples = 3;          // comparable steps needed to act on a window
    double reference_smoothing = 0.2;     // EMA weight for max-clock reference step times
};

// Closed-loop controller that finds the lowest clock cap whose step-time
// slowdown stays inside the latency budget.
//
// Step time is compared with a reference measured at max clock for steps of a
// similar size (bucketed by token count), so the slowdown estimate holds as the
// batch mix changes. The cap moves one level per window: up when the slowdown
// exceeds the budget, two levels when the scheduler is saturated (a full batch
// with requests waiting, so throughput is what is short), and down when there
// is headroom. Periodic windows at max clock keep the references fresh.
class EnergyAwarePolicy final : public ClockPolicy {
public:
    EnergyAwarePolicy(std::span<const std::uint32_t> supported_clocks, EnergyAwareConfig config = {});

    [[nodiscard]] std::string name() const override { return "energy-aware"; }
    [[nodiscard]] std::optional<std::uint32_t> initial_cap() const override { return levels_.back(); }
    [[nodiscard]] std::optional<std::uint32_t> on_step(const StepObservation& obs, TimePoint now) override;

    [[nodiscard]] std::span<const std::uint32_t> levels() const { return levels_; }
    [[nodiscard]] std::uint32_t current_cap() const { return levels_[level_]; }
    [[nodiscard]] bool calibrating() const { return calibrating_; }
    [[nodiscard]] std::optional<double> reference_s(std::size_t num_tokens) const;
    [[nodiscard]] std::optional<double> last_slowdown() const { return last_slowdown_; }

private:
    static constexpr std::size_t kBuckets = 16;
    [[nodiscard]] static std::size_t bucket_of(std::size_t num_tokens);
    std::optional<std::uint32_t> end_window(TimePoint now);

    EnergyAwareConfig config_;
    std::vector<std::uint32_t> levels_; // ascending; back() is max clock
    std::size_t level_ = 0;
    std::size_t resume_level_ = 0;
    bool calibrating_ = true;
    bool started_ = false;
    TimePoint window_start_{};
    TimePoint last_calibration_{};
    std::array<std::optional<double>, kBuckets> reference_{};
    double ratio_sum_ = 0.0;
    std::size_t ratio_count_ = 0;
    std::size_t steps_ = 0;
    std::size_t saturated_steps_ = 0;
    std::optional<double> last_slowdown_;
};

} // namespace avis::engine
