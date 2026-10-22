#include "avis/telemetry/power_sampler.hpp"

namespace avis::telemetry {

PowerSampler::PowerSampler(GpuDevice& device, std::chrono::milliseconds interval)
    : device_(device), interval_(interval) {}

PowerSampler::~PowerSampler() { stop(); }

void PowerSampler::start() {
    stop();
    {
        std::scoped_lock lock(mutex_);
        samples_.clear();
    }
    origin_ = Clock::now();
    worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

void PowerSampler::stop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
}

std::vector<PowerSample> PowerSampler::samples() const {
    std::scoped_lock lock(mutex_);
    return samples_;
}

void PowerSampler::run(std::stop_token stop) {
    auto next = Clock::now();
    while (!stop.stop_requested()) {
        if (auto reading = device_.read()) {
            PowerSample sample{to_seconds(Clock::now() - origin_), *reading};
            std::scoped_lock lock(mutex_);
            samples_.push_back(sample);
        }
        next += interval_;
        // Fixed-rate polling; if we fell behind, resynchronise instead of bursting.
        if (const auto now = Clock::now(); next < now) {
            next = now;
        }
        std::this_thread::sleep_until(next);
    }
}

double integrate_energy_j(std::span<const PowerSample> samples) {
    double joules = 0.0;
    for (std::size_t i = 1; i < samples.size(); ++i) {
        const double dt = samples[i].t_s - samples[i - 1].t_s;
        joules += 0.5 * (samples[i].reading.power_w + samples[i - 1].reading.power_w) * dt;
    }
    return joules;
}

void EnergyMeter::begin() {
    start_ = Clock::now();
    start_mj_ = device_.energy_mj();
}

EnergyReport EnergyMeter::end(std::span<const PowerSample> fallback_samples) {
    EnergyReport report;
    report.seconds = to_seconds(Clock::now() - start_);
    const auto end_mj = device_.energy_mj();
    if (start_mj_ && end_mj && *end_mj >= *start_mj_) {
        report.joules = static_cast<double>(*end_mj - *start_mj_) / 1000.0;
        report.from_counter = true;
    } else {
        report.joules = integrate_energy_j(fallback_samples);
    }
    if (report.seconds > 0.0) {
        report.avg_power_w = report.joules / report.seconds;
    }
    return report;
}

} // namespace avis::telemetry
