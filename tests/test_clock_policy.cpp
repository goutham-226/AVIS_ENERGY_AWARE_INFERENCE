#include "avis/engine/clock_policy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace avis;
using namespace avis::engine;
using namespace std::chrono_literals;

namespace {

std::vector<std::uint32_t> clocks_300_to_3000() {
    std::vector<std::uint32_t> v;
    for (std::uint32_t c = 300; c <= 3000; c += 15) {
        v.push_back(c);
    }
    return v;
}

// Simulated GPU: a step of `tokens` tokens takes base_s * (max / cap)^sensitivity.
// sensitivity 0 = purely memory bound, 1 = purely compute bound.
struct Sim {
    EnergyAwarePolicy& policy;
    double sensitivity;
    std::uint32_t cap = 3000;
    TimePoint now = TimePoint{} + 1h;
    int changes = 0;

    double step_time(std::size_t tokens) const {
        const double base = 0.010 + 0.0001 * static_cast<double>(tokens);
        return base * std::pow(3000.0 / cap, sensitivity);
    }

    void run(Duration span, std::size_t tokens = 8, std::size_t waiting = 0, std::size_t running = 8) {
        const auto end = now + span;
        while (now < end) {
            const double dt = step_time(tokens);
            now += std::chrono::duration_cast<Duration>(std::chrono::duration<double>(dt));
            if (auto next = policy.on_step({.num_tokens = tokens, .num_seqs = running, .step_s = dt,
                                            .waiting = waiting, .running = running, .max_running = 8},
                                           now)) {
                cap = *next;
                ++changes;
            }
        }
    }
};

} // namespace

TEST(FixedClockPolicy, NeverChangesClocks) {
    FixedClockPolicy none;
    EXPECT_FALSE(none.initial_cap());
    EXPECT_EQ(none.name(), "default-clocks");
    FixedClockPolicy fixed(1800);
    EXPECT_EQ(fixed.initial_cap(), 1800u);
    EXPECT_FALSE(fixed.on_step({.num_tokens = 8, .step_s = 0.01}, Clock::now()));
}

TEST(EnergyAwarePolicy, LevelsSpanFloorToMax) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.num_levels = 10, .floor_fraction = 0.4});
    const auto levels = policy.levels();
    ASSERT_EQ(levels.size(), 10u);
    EXPECT_TRUE(std::ranges::is_sorted(levels));
    EXPECT_EQ(levels.back(), 3000u);
    EXPECT_GE(levels.front(), 1200u);
    EXPECT_LE(levels.front(), 1215u);
    EXPECT_EQ(policy.initial_cap(), 3000u);
}

TEST(EnergyAwarePolicy, MemoryBoundWorkloadSettlesNearFloor) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.latency_budget = 0.10, .recalibrate_every = 3600s});
    Sim sim{policy, /*sensitivity=*/0.05};
    sim.run(10s);
    EXPECT_LE(sim.cap, policy.levels()[1]);
    ASSERT_TRUE(policy.last_slowdown());
    EXPECT_LE(*policy.last_slowdown(), 0.10);
}

TEST(EnergyAwarePolicy, ComputeBoundWorkloadStaysNearMax) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.latency_budget = 0.10, .recalibrate_every = 3600s});
    Sim sim{policy, /*sensitivity=*/1.0};
    sim.run(10s);
    // One level below max costs ~7% (inside the 10% budget); two levels cost ~16%.
    EXPECT_GE(sim.cap, policy.levels()[policy.levels().size() - 2]);
}

TEST(EnergyAwarePolicy, SlowdownStaysWithinBudgetOnAverage) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.latency_budget = 0.10, .recalibrate_every = 3600s});
    Sim sim{policy, /*sensitivity=*/0.5};
    sim.run(5s); // converge
    const double reference = sim.step_time(8) * std::pow(sim.cap / 3000.0, 0.5);
    double total = 0.0;
    int n = 0;
    const auto end = sim.now + 10s;
    while (sim.now < end) {
        sim.run(100ms);
        total += sim.step_time(8);
        ++n;
    }
    // Probing below the sweet spot costs a little, but stays near the budget.
    EXPECT_LE(total / n / reference - 1.0, 0.12);
}

TEST(EnergyAwarePolicy, SaturationRaisesClocks) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.recalibrate_every = 3600s});
    Sim sim{policy, 0.05};
    sim.run(10s);
    const auto low = sim.cap;
    ASSERT_LT(low, 3000u);
    sim.run(3s, 8, /*waiting=*/5, /*running=*/8);
    EXPECT_EQ(sim.cap, 3000u);
}

TEST(EnergyAwarePolicy, PeriodicallyRecalibratesAtMaxClock) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.window = 300ms, .recalibrate_every = 2s});
    Sim sim{policy, 0.05};
    sim.run(1500ms);
    ASSERT_FALSE(policy.calibrating());
    bool saw_calibration = false;
    const auto end = sim.now + 3s;
    while (sim.now < end) {
        sim.run(50ms);
        if (policy.calibrating()) {
            saw_calibration = true;
            EXPECT_EQ(sim.cap, 3000u);
        }
    }
    EXPECT_TRUE(saw_calibration);
}

TEST(EnergyAwarePolicy, ChangesClocksAtMostOncePerWindow) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks, {.window = 300ms, .recalibrate_every = 3600s});
    Sim sim{policy, 0.5};
    sim.run(6s);
    EXPECT_LE(sim.changes, 20);
}

TEST(EnergyAwarePolicy, ReferencesAreBucketedByBatchSize) {
    const auto clocks = clocks_300_to_3000();
    EnergyAwarePolicy policy(clocks);
    const auto now = Clock::now();
    (void)policy.on_step({.num_tokens = 4, .step_s = 0.010}, now);
    (void)policy.on_step({.num_tokens = 300, .step_s = 0.050}, now);
    ASSERT_TRUE(policy.reference_s(4));
    ASSERT_TRUE(policy.reference_s(300));
    EXPECT_DOUBLE_EQ(*policy.reference_s(5), 0.010);
    EXPECT_DOUBLE_EQ(*policy.reference_s(260), 0.050);
    EXPECT_FALSE(policy.reference_s(40));
}

TEST(EnergyAwarePolicy, RequiresClocks) {
    EXPECT_THROW(EnergyAwarePolicy({}), std::invalid_argument);
}
