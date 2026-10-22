#include "avis/report/chart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>

namespace avis::report {

namespace {

constexpr std::array<std::string_view, 6> kAnsi{"\033[36m", "\033[33m", "\033[32m", "\033[35m", "\033[34m", "\033[31m"};
constexpr std::string_view kReset = "\033[0m";
constexpr std::array<std::string_view, 6> kSvgColors{"#2a9d8f", "#e9a23b", "#5aa469", "#b56ab5", "#4c78a8", "#e45756"};

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

struct Bounds {
    double x0, x1, y0, y1;
};

Bounds bounds_of(std::span<const Series> series, const ChartOptions& options) {
    Bounds b{std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
             std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
    for (const auto& s : series) {
        for (const auto& [x, y] : s.points) {
            b.x0 = std::min(b.x0, x);
            b.x1 = std::max(b.x1, x);
            b.y0 = std::min(b.y0, y);
            b.y1 = std::max(b.y1, y);
        }
    }
    if (b.x0 > b.x1) {
        b = {0, 1, 0, 1};
    }
    if (options.y_min) {
        b.y0 = *options.y_min;
    }
    if (options.y_max) {
        b.y1 = *options.y_max;
    }
    if (b.x1 - b.x0 < 1e-12) {
        b.x1 = b.x0 + 1.0;
    }
    if (b.y1 - b.y0 < 1e-12) {
        b.y1 = b.y0 + 1.0;
    }
    return b;
}

std::string format_tick(double v) {
    const double a = std::abs(v);
    if (a >= 1000) {
        return std::format("{:.0f}", v);
    }
    if (a >= 10) {
        return std::format("{:.1f}", v);
    }
    return std::format("{:.2f}", v);
}

std::string xml_escape(std::string_view text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '&': out += "&amp;"; break;
        case '"': out += "&quot;"; break;
        default: out += c;
        }
    }
    return out;
}

// Braille dot bit for sub-cell position (dx in [0,1], dy in [0,3]).
constexpr std::uint8_t braille_bit(int dx, int dy) {
    constexpr std::array<std::array<std::uint8_t, 4>, 2> bits{{{0x01, 0x02, 0x04, 0x40}, {0x08, 0x10, 0x20, 0x80}}};
    return bits[dx][dy];
}

} // namespace

std::string render_line_chart(std::span<const Series> series, const ChartOptions& options) {
    const std::size_t w = std::max<std::size_t>(options.width, 8);
    const std::size_t h = std::max<std::size_t>(options.height, 3);
    const auto b = bounds_of(series, options);
    const long dots_x = static_cast<long>(w * 2);
    const long dots_y = static_cast<long>(h * 4);

    std::vector<std::uint8_t> cells(w * h, 0);
    std::vector<int> owner(w * h, -1);

    auto plot = [&](long px, long py, int s) {
        if (px < 0 || py < 0 || px >= dots_x || py >= dots_y) {
            return;
        }
        const auto cx = static_cast<std::size_t>(px / 2);
        const auto cy = static_cast<std::size_t>(py / 4);
        cells[cy * w + cx] |= braille_bit(static_cast<int>(px % 2), static_cast<int>(py % 4));
        owner[cy * w + cx] = s;
    };
    auto to_dot = [&](double x, double y) {
        const double fx = (x - b.x0) / (b.x1 - b.x0);
        const double fy = (std::clamp(y, b.y0, b.y1) - b.y0) / (b.y1 - b.y0);
        return std::pair{std::lround(fx * static_cast<double>(dots_x - 1)),
                         std::lround((1.0 - fy) * static_cast<double>(dots_y - 1))};
    };

    for (std::size_t s = 0; s < series.size(); ++s) {
        const auto& pts = series[s].points;
        for (std::size_t i = 0; i < pts.size(); ++i) {
            auto [x1, y1] = to_dot(pts[i].first, pts[i].second);
            if (i == 0) {
                plot(x1, y1, static_cast<int>(s));
                continue;
            }
            // Bresenham from the previous point.
            auto [x0, y0] = to_dot(pts[i - 1].first, pts[i - 1].second);
            const long dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            const long dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            long err = dx + dy;
            while (true) {
                plot(x0, y0, static_cast<int>(s));
                if (x0 == x1 && y0 == y1) {
                    break;
                }
                const long e2 = 2 * err;
                if (e2 >= dy) {
                    err += dy;
                    x0 += sx;
                }
                if (e2 <= dx) {
                    err += dx;
                    y0 += sy;
                }
            }
        }
    }

    const std::string top = format_tick(b.y1);
    const std::string mid = format_tick((b.y0 + b.y1) / 2);
    const std::string bottom = format_tick(b.y0);
    const std::size_t label_w = std::max({top.size(), mid.size(), bottom.size()});

    std::string out;
    if (!options.title.empty()) {
        out += std::format("{:>{}}  {}\n", "", label_w, options.title);
    }
    if (!options.y_label.empty()) {
        out += std::format("{:>{}}  {}\n", "", label_w, options.y_label);
    }
    for (std::size_t row = 0; row < h; ++row) {
        const std::string& label = row == 0 ? top : row == h / 2 ? mid : row == h - 1 ? bottom : std::string();
        out += std::format("{:>{}} ┤", label, label_w);
        int current = -1;
        for (std::size_t col = 0; col < w; ++col) {
            const auto idx = row * w + col;
            const int s = cells[idx] ? owner[idx] : -1;
            if (options.color && s != current) {
                out += s >= 0 ? kAnsi[static_cast<std::size_t>(s) % kAnsi.size()] : kReset;
                current = s;
            }
            append_utf8(out, cells[idx] ? char32_t{0x2800} + cells[idx] : U' ');
        }
        if (options.color && current >= 0) {
            out += kReset;
        }
        out += '\n';
    }
    out += std::format("{:>{}} └", "", label_w);
    for (std::size_t i = 0; i < w; ++i) {
        out += "─";
    }
    out += '\n';
    const std::string x_left = format_tick(b.x0);
    const std::string x_mid = format_tick((b.x0 + b.x1) / 2);
    const std::string x_right = format_tick(b.x1);
    std::string axis(w, ' ');
    axis.replace(0, x_left.size(), x_left);
    if (w > x_mid.size() + x_left.size() + x_right.size() + 2) {
        axis.replace(w / 2 - x_mid.size() / 2, x_mid.size(), x_mid);
    }
    axis.replace(w - x_right.size(), x_right.size(), x_right);
    out += std::format("{:>{}}  {}\n", "", label_w, axis);
    if (!options.x_label.empty()) {
        out += std::format("{:>{}}  {:^{}}\n", "", label_w, options.x_label, w);
    }

    // Legend
    out += std::format("{:>{}}  ", "", label_w);
    for (std::size_t s = 0; s < series.size(); ++s) {
        if (options.color) {
            out += kAnsi[s % kAnsi.size()];
        }
        out += "━━ ";
        out += series[s].name;
        if (options.color) {
            out += kReset;
        }
        out += "   ";
    }
    out += '\n';
    return out;
}

std::string render_bar_chart(std::span<const Bar> bars, const std::string& title, const std::string& unit,
                             std::size_t width, bool color) {
    constexpr std::array<std::string_view, 8> kEighths{"▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
    double max_value = 0.0;
    std::size_t label_w = 0;
    for (const auto& bar : bars) {
        max_value = std::max(max_value, bar.value);
        label_w = std::max(label_w, bar.label.size());
    }
    std::string out;
    if (!title.empty()) {
        out += title + '\n';
    }
    for (std::size_t i = 0; i < bars.size(); ++i) {
        const auto& bar = bars[i];
        out += std::format("  {:<{}} │", bar.label, label_w);
        const double frac = max_value > 0.0 ? bar.value / max_value : 0.0;
        const auto eighths = static_cast<std::size_t>(std::lround(frac * static_cast<double>(width * 8)));
        if (color) {
            out += kAnsi[i % kAnsi.size()];
        }
        for (std::size_t f = 0; f < eighths / 8; ++f) {
            out += kEighths[7];
        }
        if (eighths % 8 != 0) {
            out += kEighths[eighths % 8 - 1];
        }
        if (color) {
            out += kReset;
        }
        out += std::format(" {} {}\n", format_tick(bar.value), unit);
    }
    return out;
}

std::string render_svg_line_chart(std::span<const Series> series, const ChartOptions& options) {
    constexpr double W = 900, H = 420, left = 70, right = 20, top = 50, bottom = 70;
    const auto b = bounds_of(series, options);
    const double pw = W - left - right;
    const double ph = H - top - bottom;
    auto sx = [&](double x) { return left + (x - b.x0) / (b.x1 - b.x0) * pw; };
    auto sy = [&](double y) { return top + (1.0 - (std::clamp(y, b.y0, b.y1) - b.y0) / (b.y1 - b.y0)) * ph; };

    std::string svg = std::format(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{0}\" height=\"{1}\" viewBox=\"0 0 {0} {1}\" "
        "font-family=\"sans-serif\" font-size=\"12\">\n"
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n",
        W, H);
    if (!options.title.empty()) {
        svg += std::format("<text x=\"{}\" y=\"28\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\">{}</text>\n",
                           W / 2, xml_escape(options.title));
    }
    constexpr int kTicks = 5;
    for (int i = 0; i <= kTicks; ++i) {
        const double yv = b.y0 + (b.y1 - b.y0) * i / kTicks;
        const double xv = b.x0 + (b.x1 - b.x0) * i / kTicks;
        svg += std::format("<line x1=\"{0}\" x2=\"{1}\" y1=\"{2:.1f}\" y2=\"{2:.1f}\" stroke=\"#e5e5e5\"/>\n", left,
                           W - right, sy(yv));
        svg += std::format("<text x=\"{}\" y=\"{:.1f}\" text-anchor=\"end\" fill=\"#555\">{}</text>\n", left - 8,
                           sy(yv) + 4, format_tick(yv));
        svg += std::format("<text x=\"{:.1f}\" y=\"{}\" text-anchor=\"middle\" fill=\"#555\">{}</text>\n", sx(xv),
                           H - bottom + 18, format_tick(xv));
    }
    svg += std::format("<line x1=\"{0}\" x2=\"{0}\" y1=\"{1}\" y2=\"{2}\" stroke=\"#333\"/>\n", left, top, H - bottom);
    svg += std::format("<line x1=\"{0}\" x2=\"{1}\" y1=\"{2}\" y2=\"{2}\" stroke=\"#333\"/>\n", left, W - right,
                       H - bottom);
    if (!options.x_label.empty()) {
        svg += std::format("<text x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text>\n", left + pw / 2,
                           H - bottom + 40, xml_escape(options.x_label));
    }
    if (!options.y_label.empty()) {
        svg += std::format("<text transform=\"translate(18,{}) rotate(-90)\" text-anchor=\"middle\">{}</text>\n",
                           top + ph / 2, xml_escape(options.y_label));
    }
    for (std::size_t s = 0; s < series.size(); ++s) {
        const auto color = kSvgColors[s % kSvgColors.size()];
        svg += std::format("<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"1.6\" points=\"", color);
        for (const auto& [x, y] : series[s].points) {
            svg += std::format("{:.1f},{:.1f} ", sx(x), sy(y));
        }
        svg += "\"/>\n";
        const double lx = left + 10 + static_cast<double>(s) * 200;
        svg += std::format("<rect x=\"{:.1f}\" y=\"{}\" width=\"18\" height=\"4\" fill=\"{}\"/>\n", lx, H - 18, color);
        svg += std::format("<text x=\"{:.1f}\" y=\"{}\">{}</text>\n", lx + 24, H - 13, xml_escape(series[s].name));
    }
    svg += "</svg>\n";
    return svg;
}

} // namespace avis::report
