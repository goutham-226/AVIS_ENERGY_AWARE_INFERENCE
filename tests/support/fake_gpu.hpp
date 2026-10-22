#pragma once

#include "avis/telemetry/gpu_device.hpp"

#include <algorithm>
#include <mutex>

namespace avis::test_support {

// Thread-safe fake GPU: power is a simple function of the locked clock cap, and
// the energy counter advances with wall time at that power.
class FakeGpu final : public telemetry::GpuDevice {
public:
    explicit FakeGpu(bool can_control = true, bool energy_counter = true)
        : can_control_(can_control), energy_counter_(energy_counter), last_(Clock::now()) {}

    std::string name() const override { return "FakeGpu"; }

    Result<telemetry::GpuReading> read() override {
        std::scoped_lock lock(mutex_);
        advance();
        telemetry::GpuReading r;
        r.power_w = power_w();
        r.sm_clock_mhz = cap_mhz_;
        return r;
    }

    std::optional<std::uint64_t> energy_mj() override {
        if (!energy_counter_) {
            return std::nullopt;
        }
        std::scoped_lock lock(mutex_);
        advance();
        return static_cast<std::uint64_t>(energy_mj_);
    }

    std::vector<std::uint32_t> supported_sm_clocks() override {
        std::vector<std::uint32_t> clocks;
        for (std::uint32_t c = 300; c <= kMaxClock; c += 150) {
            clocks.push_back(c);
        }
        return clocks;
    }

    bool can_control_clocks() const override { return can_control_; }

    Result<void> lock_sm_clocks(std::uint32_t, std::uint32_t max_mhz) override {
        if (!can_control_) {
            return make_error("no permission", ErrorCode::permission_denied);
        }
        std::scoped_lock lock(mutex_);
        advance();
        cap_mhz_ = max_mhz;
        ++lock_calls_;
        return {};
    }

    Result<void> reset_sm_clocks() override {
        std::scoped_lock lock(mutex_);
        advance();
        cap_mhz_ = kMaxClock;
        return {};
    }

    std::uint32_t cap_mhz() const {
        std::scoped_lock lock(mutex_);
        return cap_mhz_;
    }
    int lock_calls() const {
        std::scoped_lock lock(mutex_);
        return lock_calls_;
    }

    static constexpr std::uint32_t kMaxClock = 3000;

private:
    double power_w() const { return 20.0 + 100.0 * static_cast<double>(cap_mhz_) / kMaxClock; }

    void advance() {
        const auto now = Clock::now();
        energy_mj_ += power_w() * to_seconds(now - last_) * 1000.0;
        last_ = now;
    }

    mutable std::mutex mutex_;
    bool can_control_;
    bool energy_counter_;
    std::uint32_t cap_mhz_ = kMaxClock;
    double energy_mj_ = 0.0;
    TimePoint last_;
    int lock_calls_ = 0;
};

} // namespace avis::test_support
