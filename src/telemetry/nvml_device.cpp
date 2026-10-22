#include "avis/telemetry/nvml_device.hpp"

#include <nvml.h>

#include <algorithm>
#include <array>
#include <format>

namespace avis::telemetry {

namespace {

Error nvml_error(std::string_view what, nvmlReturn_t rc) {
    const auto code = rc == NVML_ERROR_NO_PERMISSION ? ErrorCode::permission_denied
                      : rc == NVML_ERROR_NOT_SUPPORTED ? ErrorCode::unavailable
                                                       : ErrorCode::generic;
    return Error{code, std::format("{}: {}", what, nvmlErrorString(rc))};
}

} // namespace

struct NvmlDevice::Impl {
    nvmlDevice_t handle{};
    bool has_energy_counter = false;
};

NvmlDevice::NvmlDevice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Result<std::unique_ptr<NvmlDevice>> NvmlDevice::open(unsigned index) {
    if (auto rc = nvmlInit_v2(); rc != NVML_SUCCESS) {
        return std::unexpected(nvml_error("nvmlInit", rc));
    }
    auto impl = std::make_unique<Impl>();
    if (auto rc = nvmlDeviceGetHandleByIndex_v2(index, &impl->handle); rc != NVML_SUCCESS) {
        nvmlShutdown();
        return std::unexpected(nvml_error("nvmlDeviceGetHandleByIndex", rc));
    }
    unsigned long long energy = 0;
    impl->has_energy_counter =
        nvmlDeviceGetTotalEnergyConsumption(impl->handle, &energy) == NVML_SUCCESS;

    std::unique_ptr<NvmlDevice> device(new NvmlDevice(std::move(impl)));
    // Resetting locked clocks is harmless and tells us whether we hold the
    // privileges (root) needed to change them later.
    device->can_control_clocks_ =
        nvmlDeviceResetGpuLockedClocks(device->impl_->handle) == NVML_SUCCESS;
    return device;
}

NvmlDevice::~NvmlDevice() {
    if (clocks_locked_) {
        nvmlDeviceResetGpuLockedClocks(impl_->handle);
    }
    nvmlShutdown();
}

std::string NvmlDevice::name() const {
    std::array<char, NVML_DEVICE_NAME_V2_BUFFER_SIZE> buffer{};
    if (nvmlDeviceGetName(impl_->handle, buffer.data(), buffer.size()) != NVML_SUCCESS) {
        return "unknown NVIDIA GPU";
    }
    return buffer.data();
}

Result<GpuReading> NvmlDevice::read() {
    GpuReading reading;
    unsigned power_mw = 0;
    if (auto rc = nvmlDeviceGetPowerUsage(impl_->handle, &power_mw); rc != NVML_SUCCESS) {
        return std::unexpected(nvml_error("nvmlDeviceGetPowerUsage", rc));
    }
    reading.power_w = power_mw / 1000.0;

    // The remaining fields are best-effort: a failure leaves them at zero.
    nvmlDeviceGetClockInfo(impl_->handle, NVML_CLOCK_SM, &reading.sm_clock_mhz);
    nvmlDeviceGetClockInfo(impl_->handle, NVML_CLOCK_MEM, &reading.mem_clock_mhz);
    if (nvmlUtilization_t util{}; nvmlDeviceGetUtilizationRates(impl_->handle, &util) == NVML_SUCCESS) {
        reading.util_pct = util.gpu;
    }
    nvmlDeviceGetTemperature(impl_->handle, NVML_TEMPERATURE_GPU, &reading.temp_c);
    if (nvmlMemory_t memory{}; nvmlDeviceGetMemoryInfo(impl_->handle, &memory) == NVML_SUCCESS) {
        reading.mem_used_mb = memory.used / (1024 * 1024);
    }
    return reading;
}

std::optional<std::uint64_t> NvmlDevice::energy_mj() {
    if (!impl_->has_energy_counter) {
        return std::nullopt;
    }
    unsigned long long energy = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(impl_->handle, &energy) != NVML_SUCCESS) {
        return std::nullopt;
    }
    return energy;
}

std::vector<std::uint32_t> NvmlDevice::supported_sm_clocks() {
    unsigned mem_clock = 0;
    if (nvmlDeviceGetClockInfo(impl_->handle, NVML_CLOCK_MEM, &mem_clock) != NVML_SUCCESS) {
        return {};
    }
    // Supported graphics clocks depend on the memory clock; ask for the one in use.
    std::array<unsigned, 512> clocks{};
    unsigned count = clocks.size();
    if (nvmlDeviceGetSupportedGraphicsClocks(impl_->handle, mem_clock, &count, clocks.data()) !=
        NVML_SUCCESS) {
        // Fall back to the maximum memory clock the board reports.
        std::array<unsigned, 64> mem_clocks{};
        unsigned mem_count = mem_clocks.size();
        if (nvmlDeviceGetSupportedMemoryClocks(impl_->handle, &mem_count, mem_clocks.data()) !=
                NVML_SUCCESS ||
            mem_count == 0) {
            return {};
        }
        count = clocks.size();
        if (nvmlDeviceGetSupportedGraphicsClocks(impl_->handle, mem_clocks[0], &count,
                                                 clocks.data()) != NVML_SUCCESS) {
            return {};
        }
    }
    std::vector<std::uint32_t> result(clocks.begin(), clocks.begin() + count);
    std::ranges::sort(result);
    const auto [first, last] = std::ranges::unique(result);
    result.erase(first, last);
    return result;
}

Result<void> NvmlDevice::lock_sm_clocks(std::uint32_t min_mhz, std::uint32_t max_mhz) {
    if (auto rc = nvmlDeviceSetGpuLockedClocks(impl_->handle, min_mhz, max_mhz); rc != NVML_SUCCESS) {
        return std::unexpected(nvml_error("nvmlDeviceSetGpuLockedClocks", rc));
    }
    clocks_locked_ = true;
    return {};
}

Result<void> NvmlDevice::reset_sm_clocks() {
    if (auto rc = nvmlDeviceResetGpuLockedClocks(impl_->handle); rc != NVML_SUCCESS) {
        return std::unexpected(nvml_error("nvmlDeviceResetGpuLockedClocks", rc));
    }
    clocks_locked_ = false;
    return {};
}

} // namespace avis::telemetry
