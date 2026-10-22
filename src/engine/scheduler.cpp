#include "avis/engine/scheduler.hpp"

#include <algorithm>
#include <format>

namespace avis::engine {

Scheduler::Scheduler(SchedulerConfig config, kv::BlockManager& blocks)
    : config_(config), blocks_(blocks) {
    if (config_.watermark_blocks == 0) {
        config_.watermark_blocks = std::max<std::size_t>(1, blocks_.num_blocks() / 100);
    }
    config_.max_running = std::max<std::size_t>(1, config_.max_running);
    config_.max_step_tokens = std::max<std::size_t>(1, config_.max_step_tokens);
    config_.prefill_chunk = std::max<std::size_t>(1, config_.prefill_chunk);
}

Result<void> Scheduler::add(RequestId id, Request request, TimePoint arrival) {
    if (request.prompt.empty()) {
        return make_error("request has an empty prompt", ErrorCode::invalid_argument);
    }
    if (request.prompt.size() >= config_.max_seq_len) {
        return make_error(std::format("prompt has {} tokens but max_seq_len is {}", request.prompt.size(),
                                      config_.max_seq_len),
                          ErrorCode::invalid_argument);
    }
    if (blocks_.blocks_for(request.prompt.size()) + config_.watermark_blocks > blocks_.num_blocks()) {
        return make_error("prompt does not fit in the KV cache", ErrorCode::invalid_argument);
    }
    auto seq = std::make_unique<Sequence>();
    seq->id = id;
    seq->tokens = request.prompt;
    seq->prompt_len = request.prompt.size();
    seq->metrics.arrival = arrival;
    seq->metrics.prompt_tokens = seq->prompt_len;
    seq->request = std::move(request);
    waiting_.push_back(std::move(seq));
    return {};
}

void Scheduler::add_chunk(StepPlan& plan, Sequence& seq, std::size_t count) const {
    const bool sample = seq.num_computed + count == seq.tokens.size();
    const bool decode = count == 1 && seq.pending() == 1 && seq.generated() > 0;
    plan.chunks.push_back({&seq, seq.num_computed, count, sample});
    (decode ? plan.decode_tokens : plan.prefill_tokens) += count;
}

void Scheduler::preempt(std::size_t running_index) {
    auto seq = std::move(running_[running_index]);
    running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(running_index));
    blocks_.release(*seq->slot, /*retain_cache=*/false);
    seq->slot.reset();
    seq->num_computed = 0;
    ++seq->metrics.preemptions;
    ++preemptions_;
    waiting_.push_front(std::move(seq));
}

StepPlan Scheduler::schedule(TimePoint now) {
    StepPlan plan;
    std::size_t budget = config_.max_step_tokens;

    // 1. One decode token per running sequence, oldest first. When the cache is
    //    full, preempt from the newest end until the sequence fits.
    for (std::size_t i = 0; i < running_.size() && budget > 0;) {
        auto& seq = *running_[i];
        if (seq.pending() != 1) {
            ++i;
            continue;
        }
        bool fits = blocks_.reserve(*seq.slot, seq.tokens.size());
        while (!fits && running_.size() - 1 > i) {
            preempt(running_.size() - 1);
            fits = blocks_.reserve(*seq.slot, seq.tokens.size());
        }
        if (!fits) {
            preempt(i); // it was the newest; nothing left to take memory from
            continue;
        }
        add_chunk(plan, seq, 1);
        --budget;
        ++i;
    }

    // 2. Continue chunked prefill of running sequences.
    for (std::size_t i = 0; i < running_.size() && budget > 0;) {
        auto& seq = *running_[i];
        if (seq.pending() <= 1) {
            ++i;
            continue;
        }
        const std::size_t count = std::min({seq.pending(), config_.prefill_chunk, budget});
        if (!blocks_.reserve(*seq.slot, seq.num_computed + count)) {
            if (plan.empty() && running_.size() - 1 > i) {
                preempt(running_.size() - 1); // guarantee forward progress
                continue;
            }
            ++i;
            continue;
        }
        add_chunk(plan, seq, count);
        budget -= count;
        ++i;
    }

    // 3. Admit waiting requests in arrival order.
    while (!waiting_.empty() && running_.size() < config_.max_running && budget > 0) {
        auto& seq = *waiting_.front();
        const std::size_t needed = blocks_.blocks_for(seq.tokens.size()) + config_.watermark_blocks;
        if (blocks_.available_blocks() < needed) {
            break;
        }
        const auto admission = blocks_.admit(seq.tokens);
        if (!admission) {
            break;
        }
        seq.slot = admission->slot;
        seq.num_computed = admission->cached_tokens;
        seq.metrics.cached_prompt_tokens = std::max(seq.metrics.cached_prompt_tokens, admission->cached_tokens);
        const std::size_t count = std::min({seq.pending(), config_.prefill_chunk, budget});
        if (!blocks_.reserve(*seq.slot, seq.num_computed + count)) {
            blocks_.release(*seq.slot, /*retain_cache=*/false);
            seq.slot.reset();
            seq.num_computed = 0;
            break;
        }
        if (!seq.metrics.first_scheduled) {
            seq.metrics.first_scheduled = now;
        }
        add_chunk(plan, seq, count);
        budget -= count;
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }

    plan.kv_ops = blocks_.take_ops();
    return plan;
}

StepOutcome Scheduler::complete(const StepPlan& plan, std::span<const Token> sampled,
                                const std::function<bool(Token)>& is_eog, TimePoint now) {
    StepOutcome outcome;
    std::size_t next_sample = 0;
    for (const auto& chunk : plan.chunks) {
        auto& seq = *chunk.seq;
        seq.num_computed += chunk.count;
        blocks_.commit(*seq.slot, seq.tokens, seq.num_computed);
        if (!chunk.sample) {
            continue;
        }
        const Token token = sampled[next_sample++];
        if (!seq.metrics.first_token) {
            seq.metrics.first_token = now;
        }
        if (!seq.request.ignore_eos && is_eog(token)) {
            seq.finish_reason = FinishReason::stop;
            continue;
        }
        seq.tokens.push_back(token);
        seq.metrics.generated_tokens = seq.generated();
        outcome.new_tokens.push_back({&seq, token});
        if (seq.generated() >= seq.request.max_new_tokens) {
            seq.finish_reason = FinishReason::length;
        } else if (seq.tokens.size() >= config_.max_seq_len) {
            seq.finish_reason = FinishReason::context;
        }
    }

    for (std::size_t i = 0; i < running_.size();) {
        if (!running_[i]->finish_reason) {
            ++i;
            continue;
        }
        auto seq = std::move(running_[i]);
        running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(i));
        blocks_.release(*seq->slot, /*retain_cache=*/true);
        seq->slot.reset();
        seq->metrics.finished = now;
        outcome.finished.push_back(std::move(seq));
    }
    return outcome;
}

void Scheduler::on_kv_exhausted() {
    if (!running_.empty()) {
        preempt(running_.size() - 1);
    }
}

} // namespace avis::engine
