#include "avis/report/chart.hpp"
#include "avis/report/stats.hpp"
#include "avis/workload/workload.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace avis;

TEST(PromptFile, ParsesRequestBlocks) {
    std::istringstream in(R"(
—REQUEST—
id=1
system_prompt=You are AVIS.
prompt=Explain decode.
max_new_tokens=1000
temperature=0.6
top_p=0.9
top_k=40

—REQUEST—
prompt=Second = with equals
)");
    auto specs = workload::parse_prompts(in);
    ASSERT_TRUE(specs) << specs.error().message;
    ASSERT_EQ(specs->size(), 2u);
    const auto& a = (*specs)[0];
    EXPECT_EQ(a.id, "1");
    EXPECT_EQ(a.system_prompt, "You are AVIS.");
    EXPECT_EQ(a.prompt, "Explain decode.");
    EXPECT_EQ(a.max_new_tokens, 1000u);
    EXPECT_FLOAT_EQ(a.temperature, 0.6f);
    EXPECT_FLOAT_EQ(a.top_p, 0.9f);
    EXPECT_EQ(a.top_k, 40);
    EXPECT_EQ((*specs)[1].id, "2");
    EXPECT_EQ((*specs)[1].prompt, "Second = with equals");
}

TEST(PromptFile, RejectsBadNumbers) {
    std::istringstream in("—REQUEST—\nprompt=x\ntop_k=many\n");
    auto specs = workload::parse_prompts(in);
    ASSERT_FALSE(specs);
    EXPECT_NE(specs.error().message.find("line 3"), std::string::npos);
}

TEST(PromptFile, RejectsMissingPrompt) {
    std::istringstream in("—REQUEST—\nid=7\n");
    EXPECT_FALSE(workload::parse_prompts(in));
}

TEST(PromptFile, RejectsKeyBeforeMarker) {
    std::istringstream in("prompt=orphan\n");
    EXPECT_FALSE(workload::parse_prompts(in));
}

TEST(PromptFile, RepositoryPromptsParse) {
    auto specs = workload::load_prompts(AVIS_SOURCE_DIR "/prompts.txt");
    ASSERT_TRUE(specs) << specs.error().message;
    EXPECT_GE(specs->size(), 2u);
}

TEST(Arrivals, PoissonIsSeededAndIncreasing) {
    const auto a = workload::poisson_arrivals(200, 4.0, 7);
    const auto b = workload::poisson_arrivals(200, 4.0, 7);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.front(), 0.0);
    EXPECT_TRUE(std::ranges::is_sorted(a));
    // Mean gap of a rate-4 process is 0.25 s.
    EXPECT_NEAR(a.back() / 199.0, 0.25, 0.05);
}

TEST(Arrivals, NonPositiveRateMeansBurst) {
    const auto a = workload::poisson_arrivals(5, 0.0, 1);
    EXPECT_EQ(a, std::vector<double>(5, 0.0));
}

TEST(Stats, PercentilesInterpolate) {
    const std::vector<double> v{4, 1, 3, 2, 5};
    EXPECT_DOUBLE_EQ(report::percentile(v, 0.5), 3.0);
    EXPECT_DOUBLE_EQ(report::percentile(v, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(report::percentile(v, 1.0), 5.0);
    EXPECT_DOUBLE_EQ(report::percentile(v, 0.25), 2.0);
    EXPECT_DOUBLE_EQ(report::percentile(std::vector<double>{1, 2}, 0.5), 1.5);
    const auto s = report::summarize(v);
    EXPECT_EQ(s.count, 5u);
    EXPECT_DOUBLE_EQ(s.mean, 3.0);
    EXPECT_DOUBLE_EQ(s.max, 5.0);
    EXPECT_EQ(report::summarize({}).count, 0u);
}

TEST(Chart, LineChartHasRequestedShape) {
    report::Series s{"power", {}};
    for (int i = 0; i <= 100; ++i) {
        s.points.emplace_back(i * 0.1, 50.0 + 40.0 * std::sin(i * 0.1));
    }
    const report::ChartOptions opts{.width = 40, .height = 8, .title = "GPU power", .color = false};
    const std::vector series{s};
    const auto text = report::render_line_chart(series, opts);
    EXPECT_NE(text.find("GPU power"), std::string::npos);
    EXPECT_NE(text.find("power"), std::string::npos);
    // 8 plot rows, each with the axis glyph.
    std::size_t rows = 0;
    for (std::size_t pos = 0; (pos = text.find("┤", pos)) != std::string::npos; ++pos) {
        ++rows;
    }
    EXPECT_EQ(rows, 8u);
    // Braille glyphs live in U+2800..U+28FF, encoded as E2 A0..A3 xx.
    EXPECT_NE(text.find("\xE2\xA0"), std::string::npos);
    EXPECT_EQ(text.find("\033["), std::string::npos) << "colour disabled";
}

TEST(Chart, LineChartHandlesEmptyAndFlatSeries) {
    const std::vector<report::Series> none;
    EXPECT_FALSE(report::render_line_chart(none, {}).empty());
    const std::vector flat{report::Series{"flat", {{0, 5}, {1, 5}}}};
    EXPECT_FALSE(report::render_line_chart(flat, {.color = false}).empty());
}

TEST(Chart, BarChartScalesToLargestValue) {
    const std::vector<report::Bar> bars{{"a", 10.0}, {"b", 5.0}};
    const auto text = report::render_bar_chart(bars, "J/token", "J", 10, false);
    // Full bar is 10 full blocks; half bar is 5.
    EXPECT_NE(text.find("██████████ 10.0 J"), std::string::npos);
    EXPECT_NE(text.find("│█████ 5.00 J"), std::string::npos);
}

TEST(Chart, SvgIsWellFormedAndEscaped) {
    const std::vector series{report::Series{"a<b", {{0, 1}, {1, 2}, {2, 1.5}}}};
    const auto svg = report::render_svg_line_chart(series, {.title = "Power & clocks"});
    EXPECT_TRUE(svg.starts_with("<svg"));
    EXPECT_NE(svg.find("</svg>"), std::string::npos);
    EXPECT_NE(svg.find("<polyline"), std::string::npos);
    EXPECT_NE(svg.find("Power &amp; clocks"), std::string::npos);
    EXPECT_NE(svg.find("a&lt;b"), std::string::npos);
}
