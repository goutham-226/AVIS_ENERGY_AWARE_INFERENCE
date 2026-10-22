#pragma once

#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace avis::report {

struct Series {
    std::string name;
    std::vector<std::pair<double, double>> points; // (x, y), x ascending
};

struct ChartOptions {
    std::size_t width = 72;  // plot columns (excluding the y-axis labels)
    std::size_t height = 14; // plot rows
    std::string title;
    std::string x_label;
    std::string y_label;
    bool color = true; // ANSI colours, one per series
    std::optional<double> y_min;
    std::optional<double> y_max;
};

// Line chart drawn with Unicode braille cells (2x4 dots per character) for the terminal.
[[nodiscard]] std::string render_line_chart(std::span<const Series> series, const ChartOptions& options);

struct Bar {
    std::string label;
    double value = 0.0;
};

// Horizontal bar chart with 1/8-character resolution.
[[nodiscard]] std::string render_bar_chart(std::span<const Bar> bars, const std::string& title,
                                           const std::string& unit, std::size_t width = 48, bool color = true);

// The same line chart as a standalone SVG document.
[[nodiscard]] std::string render_svg_line_chart(std::span<const Series> series, const ChartOptions& options);

} // namespace avis::report
