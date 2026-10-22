#pragma once

#include "avis/engine/clock_policy.hpp"
#include "avis/engine/executor.hpp"
#include "avis/engine/scheduler.hpp"
#include "avis/kv/block_manager.hpp"
#include "avis/telemetry/gpu_device.hpp"

#include <condition_variable>
#include <future>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace avis::engine {

struct EngineConfig {
    SchedulerConfig scheduler;
    kv::BlockManagerConfig kv;
};

struct ClockEvent {
    double t_s = 0.0; // seconds since the engine was created
    std::uint32_t cap_mhz = 0;
    bool applied = false; // false when clock control is unavailable
};

struct EngineStats {
    std::size_t steps = 0;
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;
    std::size_t completed = 0;
    std::size_t preemptions = 0;
    std::size_t kv_exhausted = 0;
    std::size_t max_batch_seqs = 0;
    double busy_s = 0.0; // time spent inside forward passes
    kv::BlockManagerStats kv;
    std::vector<ClockEvent> clock_events;
};

// Serves many concurrent requests on one model.
//
// submit() is thread-safe and returns a future for the finished output. The
// engine either runs its own loop on a background thread (start/stop) or is
// driven synchronously with step(). Each step schedules one continuous batch,
// executes it, streams new text, completes finished requests and lets the
// clock policy adjust the GPU's SM clock cap.
class Engine {
public:
    Engine(Executor& executor, EngineConfig config, ClockPolicy& policy, telemetry::GpuDevice* gpu = nullptr);
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine();

    [[nodiscard]] std::future<RequestOutput> submit(Request request);

    // Runs one scheduling iteration on the calling thread. Returns false when idle.
    bool step();

    void start();
    void stop();
    // Blocks until every submitted request has finished.
    void wait_idle();

    [[nodiscard]] EngineStats stats() const;
    [[nodiscard]] const EngineConfig& config() const { return config_; }

private:
    struct Pending {
        RequestId id;
        Request request;
        TimePoint arrival;
        std::promise<RequestOutput> promise;
    };

    struct Live {
        std::promise<RequestOutput> promise;
        std::string text;
    };

    void drain_inbox();
    void apply_cap(std::uint32_t cap_mhz);
    void run(std::stop_token stop);
    void fail_all(const std::exception_ptr& error);

    Executor& executor_;
    EngineConfig config_;
    ClockPolicy& policy_;
    telemetry::GpuDevice* gpu_;
    kv::BlockManager blocks_;
    Scheduler scheduler_;
    TimePoint origin_;
    std::uint32_t min_clock_mhz_ = 0;
    bool clocks_changed_ = false;

    mutable std::mutex mutex_;
    std::condition_variable_any wake_;
    std::condition_variable idle_;
    std::vector<Pending> inbox_;
    std::size_t in_flight_ = 0; // submitted and not yet finished
    RequestId next_id_ = 1;
    EngineStats stats_;

    std::unordered_map<RequestId, Live> live_; // engine thread only
    std::jthread worker_;
};

} // namespace avis::engine
