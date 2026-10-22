#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace avis {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

using Token = std::int32_t;
using RequestId = std::uint64_t;
using SlotId = std::uint32_t;

enum class ErrorCode : std::uint8_t {
    generic,
    unavailable,       // feature or hardware not present
    permission_denied, // e.g. clock control without root
    kv_exhausted,      // the backend could not find room in the KV cache
    invalid_argument,
};

struct Error {
    ErrorCode code = ErrorCode::generic;
    std::string message;
};

template <class T>
using Result = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> make_error(std::string message,
                                                       ErrorCode code = ErrorCode::generic) {
    return std::unexpected(Error{code, std::move(message)});
}

[[nodiscard]] inline double to_seconds(Duration d) {
    return std::chrono::duration<double>(d).count();
}

[[nodiscard]] inline double to_millis(Duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

} // namespace avis
