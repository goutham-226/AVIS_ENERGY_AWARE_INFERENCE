#pragma once

#include "avis/engine/executor.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;

namespace avis::llama {

struct LlamaConfig {
    std::string model_path;
    std::uint32_t n_ctx = 16384;  // total KV cells shared by all sequences
    std::uint32_t n_batch = 512;  // max tokens per forward pass
    std::uint32_t n_seq_max = 16; // backend sequence slots (active + prefix-cache)
    std::int32_t n_gpu_layers = 99;
    bool flash_attention = true;
};

// llama.cpp backend. Uses one unified KV cache shared by every sequence so the
// block manager's prefix sharing maps onto llama_memory_seq_cp without copying data.
class LlamaExecutor final : public engine::Executor {
public:
    [[nodiscard]] static Result<std::unique_ptr<LlamaExecutor>> load(const LlamaConfig& config);

    LlamaExecutor(const LlamaExecutor&) = delete;
    LlamaExecutor& operator=(const LlamaExecutor&) = delete;
    ~LlamaExecutor() override;

    [[nodiscard]] Result<std::vector<Token>> execute(const engine::StepPlan& plan) override;
    void release(RequestId id) override;
    [[nodiscard]] bool is_eog(Token token) const override;
    [[nodiscard]] std::string token_to_piece(Token token) const override;

    [[nodiscard]] Result<std::vector<Token>> tokenize(std::string_view text, bool add_special = true) const;
    // Formats a single-turn chat with the model's built-in template.
    [[nodiscard]] std::string format_chat(std::string_view system, std::string_view user) const;
    [[nodiscard]] std::string description() const;
    [[nodiscard]] const LlamaConfig& config() const { return config_; }

private:
    struct ModelDeleter {
        void operator()(llama_model* m) const;
    };
    struct ContextDeleter {
        void operator()(llama_context* c) const;
    };
    struct SamplerDeleter {
        void operator()(llama_sampler* s) const;
    };
    using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;
    struct Batch;

    explicit LlamaExecutor(LlamaConfig config);
    [[nodiscard]] llama_sampler* sampler_for(const engine::Sequence& seq);
    void apply(const kv::KvOp& op);

    LlamaConfig config_;
    std::unique_ptr<llama_model, ModelDeleter> model_;
    std::unique_ptr<llama_context, ContextDeleter> ctx_;
    const llama_vocab* vocab_ = nullptr;
    std::unique_ptr<Batch> batch_;
    std::unordered_map<RequestId, SamplerPtr> samplers_;
};

} // namespace avis::llama
