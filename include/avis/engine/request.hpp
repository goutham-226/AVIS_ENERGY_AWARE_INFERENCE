#pragma once

#include "avis/core/types.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace avis::engine {

struct SamplingParams {
    float temperature = 0.8f;
    float top_p = 0.95f;
    std::int32_t top_k = 40;
    std::uint32_t seed = 0;
};

struct Request {
    std::vector<Token> prompt;
    std::size_t max_new_tokens = 256;
    SamplingParams sampling;
    bool ignore_eos = false; // generate exactly max_new_tokens (useful for benchmarking)
    // Called on the engine thread with each new piece of text.
    std::function<void(std::string_view)> on_text;
};

enum class FinishReason : std::uint8_t { length, stop, context };

struct RequestMetrics {
    TimePoint arrival{};
    std::optional<TimePoint> first_scheduled;
    std::optional<TimePoint> first_token;
    std::optional<TimePoint> finished;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t cached_prompt_tokens = 0; // prompt tokens served from the prefix cache
    std::size_t preemptions = 0;

    [[nodiscard]] std::optional<double> ttft_s() const {
        return first_token ? std::optional(to_seconds(*first_token - arrival)) : std::nullopt;
    }
    [[nodiscard]] std::optional<double> e2e_s() const {
        return finished ? std::optional(to_seconds(*finished - arrival)) : std::nullopt;
    }
    // Mean time per output token after the first one.
    [[nodiscard]] std::optional<double> tpot_s() const {
        if (!first_token || !finished || generated_tokens < 2) {
            return std::nullopt;
        }
        return to_seconds(*finished - *first_token) / static_cast<double>(generated_tokens - 1);
    }
};

struct RequestOutput {
    RequestId id = 0;
    std::vector<Token> tokens; // generated tokens only
    std::string text;
    FinishReason finish_reason = FinishReason::length;
    RequestMetrics metrics;
};

// A request as tracked by the scheduler.
struct Sequence {
    RequestId id = 0;
    Request request;
    std::vector<Token> tokens; // prompt followed by generated tokens
    std::size_t prompt_len = 0;
    std::size_t num_computed = 0; // tokens whose KV is materialised in the cache
    std::optional<SlotId> slot;
    std::optional<FinishReason> finish_reason;
    RequestMetrics metrics;

    [[nodiscard]] std::size_t generated() const { return tokens.size() - prompt_len; }
    [[nodiscard]] std::size_t pending() const { return tokens.size() - num_computed; }
    [[nodiscard]] std::span<const Token> generated_tokens() const {
        return std::span(tokens).subspan(prompt_len);
    }
};

} // namespace avis::engine
