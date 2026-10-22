#pragma once

#include "avis/core/types.hpp"

#include <cstdint>
#include <filesystem>
#include <istream>
#include <string>
#include <vector>

namespace avis::workload {

struct PromptSpec {
    std::string id;
    std::string system_prompt;
    std::string prompt;
    std::size_t max_new_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.95f;
    std::int32_t top_k = 40;
};

// Parses the prompts file format:
//
//   —REQUEST—            (any line containing "REQUEST" starts a new entry)
//   id=1
//   system_prompt=...
//   prompt=...
//   max_new_tokens=256
//   temperature=0.6
//   top_p=0.9
//   top_k=40
[[nodiscard]] Result<std::vector<PromptSpec>> parse_prompts(std::istream& in);
[[nodiscard]] Result<std::vector<PromptSpec>> load_prompts(const std::filesystem::path& path);

// Arrival offsets in seconds for `count` requests of a Poisson process with
// `rate_per_s` requests per second. A rate <= 0 means everything arrives at once.
[[nodiscard]] std::vector<double> poisson_arrivals(std::size_t count, double rate_per_s, std::uint64_t seed);

} // namespace avis::workload
