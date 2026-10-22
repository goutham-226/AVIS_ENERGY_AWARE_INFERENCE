#pragma once

#include "avis/engine/scheduler.hpp"

#include <string>
#include <vector>

namespace avis::engine {

// Runs batched forward passes for the engine (llama.cpp in production, a fake in tests).
class Executor {
public:
    virtual ~Executor() = default;

    // Applies plan.kv_ops, runs one forward pass over every chunk and samples one
    // token for each chunk with sample == true, in chunk order. Returns an error
    // with ErrorCode::kv_exhausted if the backend ran out of KV space; the KV ops
    // stay applied but no tokens are added.
    [[nodiscard]] virtual Result<std::vector<Token>> execute(const StepPlan& plan) = 0;

    // Drops per-request state such as the sampler.
    virtual void release(RequestId id) = 0;

    // Clears all KV state and per-request state. Called when an engine takes
    // ownership, since its block manager starts from an empty cache.
    virtual void reset() = 0;

    [[nodiscard]] virtual bool is_eog(Token token) const = 0;
    [[nodiscard]] virtual std::string token_to_piece(Token token) const = 0;
};

} // namespace avis::engine
