#pragma once

#include "avis/telemetry/gpu_device.hpp"

#include <memory>

namespace avis::telemetry {

// NVML-backed GPU. NVML is initialised for the lifetime of the object and any
// clock lock it applied is reset on destruction.
class NvmlDevice final : public GpuDevice {
public:
    [[nodiscard]] static Result<std::unique_ptr<NvmlDevice>> open(unsigned index = 0);

    NvmlDevice(const NvmlDevice&) = delete;
    NvmlDevice& operator=(const NvmlDevice&) = delete;
    ~NvmlDevice() override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] Result<GpuReading> read() override;
    [[nodiscard]] std::optional<std::uint64_t> energy_mj() override;
    [[nodiscard]] std::vector<std::uint32_t> supported_sm_clocks() override;

    [[nodiscard]] bool can_control_clocks() const override { return can_control_clocks_; }
    Result<void> lock_sm_clocks(std::uint32_t min_mhz, std::uint32_t max_mhz) override;
    Result<void> reset_sm_clocks() override;

private:
    struct Impl;
    explicit NvmlDevice(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
    bool can_control_clocks_ = false;
    bool clocks_locked_ = false;
};

} // namespace avis::telemetry
