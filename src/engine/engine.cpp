#include "avis/engine/engine.hpp"

#include <algorithm>
#include <print>
#include <stdexcept>

namespace avis::engine {

Engine::Engine(Executor& executor, EngineConfig config, ClockPolicy& policy, telemetry::GpuDevice* gpu)
    : executor_(executor),
      config_(config),
      policy_(policy),
      gpu_(gpu),
      blocks_(config.kv),
      scheduler_(config.scheduler, blocks_),
      origin_(Clock::now()) {
    executor_.reset();
    if (gpu_) {
        if (const auto clocks = gpu_->supported_sm_clocks(); !clocks.empty()) {
            min_clock_mhz_ = clocks.front();
        }
    }
    if (const auto cap = policy_.initial_cap()) {
        apply_cap(*cap);
    }
}

Engine::~Engine() {
    stop();
    if (gpu_ && clocks_changed_) {
        (void)gpu_->reset_sm_clocks();
    }
}

std::future<RequestOutput> Engine::submit(Request request) {
    Pending pending{0, std::move(request), Clock::now(), {}};
    auto future = pending.promise.get_future();
    {
        std::scoped_lock lock(mutex_);
        pending.id = next_id_++;
        inbox_.push_back(std::move(pending));
        ++in_flight_;
    }
    wake_.notify_one();
    return future;
}

void Engine::drain_inbox() {
    std::vector<Pending> incoming;
    {
        std::scoped_lock lock(mutex_);
        incoming.swap(inbox_);
    }
    for (auto& p : incoming) {
        if (auto added = scheduler_.add(p.id, std::move(p.request), p.arrival); !added) {
            p.promise.set_exception(std::make_exception_ptr(std::invalid_argument(added.error().message)));
            std::scoped_lock lock(mutex_);
            --in_flight_;
            idle_.notify_all();
            continue;
        }
        live_.emplace(p.id, Live{std::move(p.promise), {}});
    }
}

void Engine::apply_cap(std::uint32_t cap_mhz) {
    ClockEvent event{to_seconds(Clock::now() - origin_), cap_mhz, false};
    if (gpu_ && gpu_->can_control_clocks()) {
        // Cap the maximum only, so the GPU can still drop its clocks when idle.
        const std::uint32_t floor = std::min(min_clock_mhz_, cap_mhz);
        if (auto locked = gpu_->lock_sm_clocks(floor, cap_mhz)) {
            event.applied = true;
            clocks_changed_ = true;
        } else {
            std::println(stderr, "avis: failed to set clock cap {} MHz: {}", cap_mhz, locked.error().message);
        }
    }
    std::scoped_lock lock(mutex_);
    stats_.clock_events.push_back(event);
}

bool Engine::step() {
    drain_inbox();
    if (!scheduler_.has_work()) {
        return false;
    }

    const auto plan = scheduler_.schedule(Clock::now());
    if (plan.empty()) {
        // Nothing fits right now (e.g. every slot is waiting on memory); let the
        // KV ops of any preemption land and try again next iteration.
        return true;
    }

    const auto t0 = Clock::now();
    auto sampled = executor_.execute(plan);
    const auto t1 = Clock::now();
    if (!sampled) {
        if (sampled.error().code == ErrorCode::kv_exhausted) {
            scheduler_.on_kv_exhausted();
            std::scoped_lock lock(mutex_);
            ++stats_.kv_exhausted;
            return true;
        }
        throw std::runtime_error("forward pass failed: " + sampled.error().message);
    }

    auto outcome = scheduler_.complete(plan, *sampled, [this](Token t) { return executor_.is_eog(t); }, t1);

    for (const auto& [seq, token] : outcome.new_tokens) {
        const auto piece = executor_.token_to_piece(token);
        live_.at(seq->id).text += piece;
        if (seq->request.on_text) {
            seq->request.on_text(piece);
        }
    }

    for (auto& seq : outcome.finished) {
        executor_.release(seq->id);
        auto node = live_.extract(seq->id);
        RequestOutput output;
        output.id = seq->id;
        output.tokens.assign(seq->generated_tokens().begin(), seq->generated_tokens().end());
        output.text = std::move(node.mapped().text);
        output.finish_reason = seq->finish_reason.value_or(FinishReason::length);
        output.metrics = seq->metrics;
        node.mapped().promise.set_value(std::move(output));
    }

    const StepObservation obs{
        .num_tokens = plan.num_tokens(),
        .num_seqs = plan.chunks.size(),
        .step_s = to_seconds(t1 - t0),
        .waiting = scheduler_.num_waiting(),
        .running = scheduler_.num_running(),
        .max_running = scheduler_.config().max_running,
    };
    if (const auto cap = policy_.on_step(obs, t1)) {
        apply_cap(*cap);
    }

    {
        std::scoped_lock lock(mutex_);
        ++stats_.steps;
        stats_.prefill_tokens += plan.prefill_tokens;
        stats_.decode_tokens += plan.decode_tokens;
        stats_.busy_s += obs.step_s;
        stats_.max_batch_seqs = std::max(stats_.max_batch_seqs, plan.chunks.size());
        stats_.preemptions = scheduler_.num_preemptions();
        stats_.kv = blocks_.stats();
        stats_.completed += outcome.finished.size();
        in_flight_ -= outcome.finished.size();
    }
    if (!outcome.finished.empty()) {
        idle_.notify_all();
    }
    return true;
}

void Engine::run(std::stop_token stop) {
    try {
        while (!stop.stop_requested()) {
            if (step()) {
                continue;
            }
            std::unique_lock lock(mutex_);
            wake_.wait(lock, stop, [this] { return !inbox_.empty(); });
        }
    } catch (...) {
        fail_all(std::current_exception());
    }
}

void Engine::fail_all(const std::exception_ptr& error) {
    drain_inbox();
    for (auto& [id, live] : live_) {
        live.promise.set_exception(error);
    }
    live_.clear();
    std::scoped_lock lock(mutex_);
    in_flight_ = 0;
    idle_.notify_all();
}

void Engine::start() {
    if (!worker_.joinable()) {
        worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
    }
}

void Engine::stop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
        worker_ = {};
    }
}

void Engine::wait_idle() {
    std::unique_lock lock(mutex_);
    idle_.wait(lock, [this] { return in_flight_ == 0; });
}

EngineStats Engine::stats() const {
    std::scoped_lock lock(mutex_);
    return stats_;
}

} // namespace avis::engine
