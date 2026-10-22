#pragma once

#include "avis/telemetry/gpu_device.hpp"

#include <chrono>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

namespace avis::telemetry {

struct PowerSample {
    double t_s = 0.0; // seconds since the sampler was started
    GpuReading reading;
};

// Polls a GpuDevice on a background thread and keeps the samples in memory.
class PowerSampler {
public:
    explicit PowerSampler(GpuDevice& device,
                          std::chrono::milliseconds interval = std::chrono::milliseconds{20});
    PowerSampler(const PowerSampler&) = delete;
    PowerSampler& operator=(const PowerSampler&) = delete;
    ~PowerSampler();

    // Clears previous samples and starts polling. Restarting a running sampler restarts it.
    void start();
    void stop();

    [[nodiscard]] std::vector<PowerSample> samples() const;
    [[nodiscard]] TimePoint origin() const { return origin_; }

private:
    void run(std::stop_token stop);

    GpuDevice& device_;
    std::chrono::milliseconds interval_;
    TimePoint origin_{};
    mutable std::mutex mutex_;
    std::vector<PowerSample> samples_;
    std::jthread worker_;
};

struct EnergyReport {
    double joules = 0.0;
    double seconds = 0.0;
    double avg_power_w = 0.0;
    bool from_counter = false; // true when the NVML energy counter was used
};

// Trapezoidal integral of power over the samples, in joules.
[[nodiscard]] double integrate_energy_j(std::span<const PowerSample> samples);

// Measures energy across a window, preferring the hardware energy counter and
// falling back to integrating sampled power.
class EnergyMeter {
public:
    explicit EnergyMeter(GpuDevice& device) : device_(device) {}

    void begin();
    [[nodiscard]] EnergyReport end(std::span<const PowerSample> fallback_samples);

private:
    GpuDevice& device_;
    TimePoint start_{};
    std::optional<std::uint64_t> start_mj_;
};

} // namespace avis::telemetry
