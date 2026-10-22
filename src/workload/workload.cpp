#include "avis/workload/workload.hpp"

#include <charconv>
#include <format>
#include <fstream>
#include <random>
#include <string_view>

namespace avis::workload {

namespace {

std::string_view trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        return {};
    }
    return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
}

template <class T>
Result<T> parse_number(std::string_view key, std::string_view value, std::size_t line_no) {
    T out{};
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), out);
    if (ec != std::errc{} || ptr != value.data() + value.size()) {
        return make_error(std::format("line {}: invalid value '{}' for {}", line_no, value, key),
                          ErrorCode::invalid_argument);
    }
    return out;
}

} // namespace

Result<std::vector<PromptSpec>> parse_prompts(std::istream& in) {
    std::vector<PromptSpec> specs;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const auto text = trim(line);
        if (text.empty() || text.starts_with('#')) {
            continue;
        }
        if (text.find("REQUEST") != std::string_view::npos) {
            specs.emplace_back();
            continue;
        }
        const auto eq = text.find('=');
        if (eq == std::string_view::npos) {
            return make_error(std::format("line {}: expected key=value", line_no), ErrorCode::invalid_argument);
        }
        if (specs.empty()) {
            return make_error(std::format("line {}: key before the first REQUEST marker", line_no),
                              ErrorCode::invalid_argument);
        }
        const auto key = trim(text.substr(0, eq));
        const auto value = trim(text.substr(eq + 1));
        auto& spec = specs.back();

        Result<void> ok;
        auto assign = [&]<class T>(T& field) {
            auto v = parse_number<T>(key, value, line_no);
            if (!v) {
                ok = std::unexpected(v.error());
            } else {
                field = *v;
            }
        };
        if (key == "id") {
            spec.id = value;
        } else if (key == "system_prompt") {
            spec.system_prompt = value;
        } else if (key == "prompt") {
            spec.prompt = value;
        } else if (key == "max_new_tokens") {
            assign(spec.max_new_tokens);
        } else if (key == "temperature") {
            assign(spec.temperature);
        } else if (key == "top_p") {
            assign(spec.top_p);
        } else if (key == "top_k") {
            assign(spec.top_k);
        } // unknown keys are ignored for forward compatibility
        if (!ok) {
            return std::unexpected(ok.error());
        }
    }
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (specs[i].prompt.empty()) {
            return make_error(std::format("request {} has no prompt", i + 1), ErrorCode::invalid_argument);
        }
        if (specs[i].id.empty()) {
            specs[i].id = std::to_string(i + 1);
        }
    }
    return specs;
}

Result<std::vector<PromptSpec>> load_prompts(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return make_error(std::format("cannot open prompts file '{}'", path.string()), ErrorCode::unavailable);
    }
    return parse_prompts(in);
}

std::vector<double> poisson_arrivals(std::size_t count, double rate_per_s, std::uint64_t seed) {
    std::vector<double> arrivals(count, 0.0);
    if (rate_per_s <= 0.0) {
        return arrivals;
    }
    std::mt19937_64 rng(seed);
    std::exponential_distribution<double> gap(rate_per_s);
    double t = 0.0;
    for (auto& a : arrivals) {
        a = t;
        t += gap(rng);
    }
    return arrivals;
}

} // namespace avis::workload
