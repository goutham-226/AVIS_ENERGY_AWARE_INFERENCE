#include "avis/engine/clock_policy.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <stdexcept>

namespace avis::engine {

std::string FixedClockPolicy::name() const {
    return cap_ ? std::format("fixed-{}MHz", *cap_) : std::string("default-clocks");
}

EnergyAwarePolicy::EnergyAwarePolicy(std::span<const std::uint32_t> supported_clocks, EnergyAwareConfig config)
    : config_(config) {
    if (supported_clocks.empty()) {
        throw std::invalid_argument("EnergyAwarePolicy needs the GPU's supported clocks");
    }
    std::vector<std::uint32_t> clocks(supported_clocks.begin(), supported_clocks.end());
    std::ranges::sort(clocks);
    const std::uint32_t max_clock = clocks.back();
    const auto floor = static_cast<std::uint32_t>(max_clock * std::clamp(config_.floor_fraction, 0.0, 1.0));
    std::erase_if(clocks, [&](std::uint32_t c) { return c < floor; });

    // Evenly spaced targets between the floor and max, snapped to supported clocks.
    const std::size_t n = std::max<std::size_t>(2, config_.num_levels);
    const double lo = clocks.front();
    for (std::size_t i = 0; i < n; ++i) {
        const double target = lo + (max_clock - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
        const auto nearest = std::ranges::min(clocks, {}, [&](std::uint32_t c) { return std::abs(c - target); });
        if (levels_.empty() || levels_.back() != nearest) {
            levels_.push_back(nearest);
        }
    }
    level_ = levels_.size() - 1;
    resume_level_ = level_;
}

std::size_t EnergyAwarePolicy::bucket_of(std::size_t num_tokens) {
    return std::min<std::size_t>(std::bit_width(num_tokens), kBuckets - 1);
}

std::optional<double> EnergyAwarePolicy::reference_s(std::size_t num_tokens) const {
    return reference_[bucket_of(num_tokens)];
}

std::optional<std::uint32_t> EnergyAwarePolicy::on_step(const StepObservation& obs, TimePoint now) {
    if (!started_) {
        started_ = true;
        window_start_ = now;
        last_calibration_ = now;
    }

    ++steps_;
    if (obs.waiting > 0 && obs.running >= obs.max_running) {
        ++saturated_steps_;
    }
    if (obs.num_tokens > 0 && obs.step_s > 0.0) {
        auto& ref = reference_[bucket_of(obs.num_tokens)];
        if (level_ == levels_.size() - 1) {
            ref = ref ? (1.0 - config_.reference_smoothing) * *ref + config_.reference_smoothing * obs.step_s
                      : obs.step_s;
        } else if (ref) {
            ratio_sum_ += obs.step_s / *ref;
            ++ratio_count_;
        }
    }

    if (now - window_start_ < config_.window) {
        return std::nullopt;
    }
    return end_window(now);
}

std::optional<std::uint32_t> EnergyAwarePolicy::end_window(TimePoint now) {
    const std::size_t old_level = level_;
    const std::size_t top = levels_.size() - 1;
    const double saturation = steps_ > 0 ? static_cast<double>(saturated_steps_) / static_cast<double>(steps_) : 0.0;

    if (calibrating_) {
        // References are fresh; resume where we were (or start descending).
        calibrating_ = false;
        last_calibration_ = now;
        level_ = resume_level_ == top && top > 0 ? top - 1 : resume_level_;
        if (saturation > config_.saturation_threshold) {
            level_ = top;
        }
    } else if (saturation > config_.saturation_threshold) {
        level_ = std::min(top, level_ + 2);
        last_slowdown_.reset();
    } else if (ratio_count_ >= config_.min_samples) {
        const double slowdown = ratio_sum_ / static_cast<double>(ratio_count_) - 1.0;
        last_slowdown_ = slowdown;
        if (slowdown > config_.latency_budget) {
            level_ = std::min(top, level_ + 1);
        } else if (slowdown < 0.5 * config_.latency_budget && level_ > 0) {
            --level_;
        }
    } else if (level_ == top && top > 0) {
        // At max clock there is nothing to compare against; probe one level down.
        --level_;
    }

    if (!calibrating_ && level_ != top && now - last_calibration_ >= config_.recalibrate_every) {
        calibrating_ = true;
        resume_level_ = level_;
        level_ = top;
    }

    window_start_ = now;
    ratio_sum_ = 0.0;
    ratio_count_ = 0;
    steps_ = 0;
    saturated_steps_ = 0;
    if (level_ == old_level) {
        return std::nullopt;
    }
    return levels_[level_];
}

} // namespace avis::engine
