#pragma once

#include <span>
#include <vector>

namespace avis::report {

struct Summary {
    std::size_t count = 0;
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

// Linear-interpolated percentile (q in [0, 1]) of unsorted values; 0 for an empty input.
[[nodiscard]] double percentile(std::span<const double> values, double q);
[[nodiscard]] Summary summarize(std::span<const double> values);

} // namespace avis::report
