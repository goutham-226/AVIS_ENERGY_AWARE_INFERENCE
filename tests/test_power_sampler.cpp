#include "avis/telemetry/power_sampler.hpp"
#include "support/fake_gpu.hpp"

#include <gtest/gtest.h>

#include <thread>

using namespace avis;
using namespace avis::telemetry;
using namespace std::chrono_literals;

TEST(EnergyIntegration, TrapezoidOfConstantPower) {
    std::vector<PowerSample> samples;
    for (int i = 0; i <= 10; ++i) {
        samples.push_back({.t_s = i * 0.1, .reading = {.power_w = 50.0}});
    }
    EXPECT_NEAR(integrate_energy_j(samples), 50.0, 1e-9);
}

TEST(EnergyIntegration, TrapezoidOfRamp) {
    std::vector<PowerSample> samples{{.t_s = 0.0, .reading = {.power_w = 0.0}},
                                     {.t_s = 2.0, .reading = {.power_w = 100.0}}};
    EXPECT_NEAR(integrate_energy_j(samples), 100.0, 1e-9);
}

TEST(EnergyIntegration, EmptyAndSingleSampleAreZero) {
    EXPECT_EQ(integrate_energy_j({}), 0.0);
    std::vector<PowerSample> one{{.t_s = 0.0, .reading = {.power_w = 80.0}}};
    EXPECT_EQ(integrate_energy_j(one), 0.0);
}

TEST(PowerSampler, CollectsSamplesWhileRunning) {
    test_support::FakeGpu gpu;
    PowerSampler sampler(gpu, 5ms);
    sampler.start();
    std::this_thread::sleep_for(60ms);
    sampler.stop();
    const auto samples = sampler.samples();
    ASSERT_GE(samples.size(), 5u);
    EXPECT_TRUE(std::ranges::is_sorted(samples, {}, &PowerSample::t_s));
    EXPECT_NEAR(samples.front().reading.power_w, 120.0, 1e-9);

    // Stopped: no more samples arrive.
    std::this_thread::sleep_for(20ms);
    EXPECT_EQ(sampler.samples().size(), samples.size());
}

TEST(PowerSampler, RestartClearsSamples) {
    test_support::FakeGpu gpu;
    PowerSampler sampler(gpu, 5ms);
    sampler.start();
    std::this_thread::sleep_for(30ms);
    sampler.start();
    sampler.stop();
    EXPECT_LE(sampler.samples().size(), 2u);
}

TEST(EnergyMeter, UsesHardwareCounterWhenAvailable) {
    test_support::FakeGpu gpu;
    EnergyMeter meter(gpu);
    meter.begin();
    std::this_thread::sleep_for(50ms);
    const auto report = meter.end({});
    EXPECT_TRUE(report.from_counter);
    EXPECT_NEAR(report.avg_power_w, 120.0, 1.0);
}

TEST(EnergyMeter, FallsBackToIntegratedSamples) {
    test_support::FakeGpu gpu(/*can_control=*/true, /*energy_counter=*/false);
    PowerSampler sampler(gpu, 5ms);
    EnergyMeter meter(gpu);
    sampler.start();
    meter.begin();
    std::this_thread::sleep_for(60ms);
    sampler.stop();
    const auto samples = sampler.samples();
    const auto report = meter.end(samples);
    EXPECT_FALSE(report.from_counter);
    EXPECT_NEAR(report.joules, integrate_energy_j(samples), 1e-9);
    EXPECT_GT(report.joules, 0.0);
}
