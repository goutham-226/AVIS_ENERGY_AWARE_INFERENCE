#pragma once

#include "avis/core/types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace avis::telemetry {

struct GpuReading {
    double power_w = 0.0;
    std::uint32_t sm_clock_mhz = 0;
    std::uint32_t mem_clock_mhz = 0;
    std::uint32_t util_pct = 0;
    std::uint32_t temp_c = 0;
    std::uint64_t mem_used_mb = 0;
};

// Abstraction over a GPU's telemetry and clock controls so the engine and the
// tests do not depend on NVML directly.
class GpuDevice {
public:
    virtual ~GpuDevice() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual Result<GpuReading> read() = 0;

    // Cumulative board energy in millijoules since driver load, if the GPU exposes a counter.
    [[nodiscard]] virtual std::optional<std::uint64_t> energy_mj() = 0;

    // Supported graphics (SM) clocks in ascending order.
    [[nodiscard]] virtual std::vector<std::uint32_t> supported_sm_clocks() = 0;

    [[nodiscard]] virtual bool can_control_clocks() const = 0;
    virtual Result<void> lock_sm_clocks(std::uint32_t min_mhz, std::uint32_t max_mhz) = 0;
    virtual Result<void> reset_sm_clocks() = 0;
};

} // namespace avis::telemetry
