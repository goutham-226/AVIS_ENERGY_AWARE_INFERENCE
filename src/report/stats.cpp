#include "avis/report/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace avis::report {

namespace {

double sorted_percentile(std::span<const double> sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    return std::lerp(sorted[lo], sorted[hi], pos - static_cast<double>(lo));
}

} // namespace

double percentile(std::span<const double> values, double q) {
    std::vector<double> sorted(values.begin(), values.end());
    std::ranges::sort(sorted);
    return sorted_percentile(sorted, q);
}

Summary summarize(std::span<const double> values) {
    Summary s;
    if (values.empty()) {
        return s;
    }
    std::vector<double> sorted(values.begin(), values.end());
    std::ranges::sort(sorted);
    s.count = sorted.size();
    s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    s.p50 = sorted_percentile(sorted, 0.50);
    s.p90 = sorted_percentile(sorted, 0.90);
    s.p95 = sorted_percentile(sorted, 0.95);
    s.p99 = sorted_percentile(sorted, 0.99);
    s.max = sorted.back();
    return s;
}

} // namespace avis::report
