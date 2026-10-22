#pragma once

#include "avis/engine/request.hpp"
#include "avis/kv/block_manager.hpp"

#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace avis::engine {

struct SchedulerConfig {
    std::size_t max_running = 8;       // sequences resident in a batch at once
    std::size_t max_step_tokens = 512; // tokens per forward pass (prefill + decode)
    std::size_t prefill_chunk = 256;   // max prompt tokens of one sequence per step
    std::size_t max_seq_len = 4096;    // prompt + generated tokens of one sequence
    std::size_t watermark_blocks = 0;  // free blocks kept back for decode growth (0 = 1% of the cache)
};

// A contiguous run of one sequence's tokens to feed in this step.
struct SeqChunk {
    Sequence* seq = nullptr;
    std::size_t begin = 0;
    std::size_t count = 0;
    bool sample = false; // the chunk ends the sequence's known tokens, so sample the next one
};

struct StepPlan {
    std::vector<kv::KvOp> kv_ops;
    std::vector<SeqChunk> chunks;
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;

    [[nodiscard]] bool empty() const { return chunks.empty(); }
    [[nodiscard]] std::size_t num_tokens() const { return prefill_tokens + decode_tokens; }
};

struct NewToken {
    Sequence* seq;
    Token token;
};

struct StepOutcome {
    std::vector<NewToken> new_tokens;                 // tokens to stream (end-of-generation tokens excluded)
    std::vector<std::unique_ptr<Sequence>> finished;
};

// Iteration-level (continuous batching) scheduler.
//
// Every step it builds one mixed batch: first one decode token for each running
// sequence (oldest first, keeping inter-token latency low), then chunks of
// pending prompts, then newly admitted requests in FCFS order, all bounded by a
// per-step token budget. Admission is gated on free KV blocks; when a running
// sequence cannot grow, the most recently admitted sequence is preempted and
// later recomputed.
class Scheduler {
public:
    Scheduler(SchedulerConfig config, kv::BlockManager& blocks);

    Result<void> add(RequestId id, Request request, TimePoint arrival);

    [[nodiscard]] StepPlan schedule(TimePoint now);

    // Applies the results of executing `plan`. `sampled` has one token per chunk with sample == true.
    StepOutcome complete(const StepPlan& plan, std::span<const Token> sampled,
                         const std::function<bool(Token)>& is_eog, TimePoint now);

    // The backend ran out of KV space while executing `plan`: give memory back
    // by preempting the newest running sequence. The plan must be discarded.
    void on_kv_exhausted();

    [[nodiscard]] bool has_work() const { return !waiting_.empty() || !running_.empty(); }
    [[nodiscard]] std::size_t num_waiting() const { return waiting_.size(); }
    [[nodiscard]] std::size_t num_running() const { return running_.size(); }
    [[nodiscard]] std::size_t num_preemptions() const { return preemptions_; }
    [[nodiscard]] const SchedulerConfig& config() const { return config_; }

private:
    void preempt(std::size_t running_index);
    void add_chunk(StepPlan& plan, Sequence& seq, std::size_t count) const;

    SchedulerConfig config_;
    kv::BlockManager& blocks_;
    std::deque<std::unique_ptr<Sequence>> waiting_;
    std::vector<std::unique_ptr<Sequence>> running_; // in admission order
    std::size_t preemptions_ = 0;
};

} // namespace avis::engine
