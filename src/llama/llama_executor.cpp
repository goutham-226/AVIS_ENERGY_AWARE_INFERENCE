#include "avis/llama/llama_executor.hpp"

#include <llama.h>

#include <array>
#include <cstdio>
#include <format>
#include <mutex>

namespace avis::llama {

namespace {

void quiet_logger(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

void init_backend_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(quiet_logger, nullptr);
        llama_backend_init();
    });
}

} // namespace

struct LlamaExecutor::Batch {
    explicit Batch(std::int32_t max_tokens) : batch(llama_batch_init(max_tokens, 0, 1)), capacity(max_tokens) {}
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;
    ~Batch() { llama_batch_free(batch); }

    void clear() { batch.n_tokens = 0; }
    std::int32_t add(Token token, std::int32_t pos, std::int32_t seq, bool logits) {
        const std::int32_t i = batch.n_tokens++;
        batch.token[i] = token;
        batch.pos[i] = pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = logits;
        return i;
    }

    llama_batch batch;
    std::int32_t capacity;
};

void LlamaExecutor::ModelDeleter::operator()(llama_model* m) const { llama_model_free(m); }
void LlamaExecutor::ContextDeleter::operator()(llama_context* c) const { llama_free(c); }
void LlamaExecutor::SamplerDeleter::operator()(llama_sampler* s) const { llama_sampler_free(s); }

LlamaExecutor::LlamaExecutor(LlamaConfig config) : config_(std::move(config)) {}
LlamaExecutor::~LlamaExecutor() = default;

Result<std::unique_ptr<LlamaExecutor>> LlamaExecutor::load(const LlamaConfig& config) {
    init_backend_once();
    std::unique_ptr<LlamaExecutor> exec(new LlamaExecutor(config));

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.progress_callback = [](float, void*) { return true; }; // no loading dots on stderr
    exec->model_.reset(llama_model_load_from_file(config.model_path.c_str(), model_params));
    if (!exec->model_) {
        return make_error(std::format("failed to load model '{}'", config.model_path), ErrorCode::unavailable);
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.n_ubatch = std::min<std::uint32_t>(config.n_batch, 512);
    ctx_params.n_seq_max = config.n_seq_max;
    ctx_params.kv_unified = true;
    ctx_params.no_perf = true;
    ctx_params.flash_attn_type =
        config.flash_attention ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    exec->ctx_.reset(llama_init_from_model(exec->model_.get(), ctx_params));
    if (!exec->ctx_) {
        return make_error("failed to create llama context (try a smaller n_ctx)", ErrorCode::unavailable);
    }

    exec->vocab_ = llama_model_get_vocab(exec->model_.get());
    exec->batch_ = std::make_unique<Batch>(static_cast<std::int32_t>(config.n_batch));
    return exec;
}

void LlamaExecutor::apply(const kv::KvOp& op) {
    llama_memory_t mem = llama_get_memory(ctx_.get());
    const auto n = static_cast<llama_pos>(op.num_tokens);
    switch (op.kind) {
    case kv::KvOpKind::copy_prefix:
        llama_memory_seq_cp(mem, static_cast<llama_seq_id>(op.src), static_cast<llama_seq_id>(op.dst), 0, n);
        break;
    case kv::KvOpKind::truncate:
        llama_memory_seq_rm(mem, static_cast<llama_seq_id>(op.dst), n, -1);
        break;
    case kv::KvOpKind::clear:
        llama_memory_seq_rm(mem, static_cast<llama_seq_id>(op.dst), -1, -1);
        break;
    }
}

llama_sampler* LlamaExecutor::sampler_for(const engine::Sequence& seq) {
    auto it = samplers_.find(seq.id);
    if (it != samplers_.end()) {
        return it->second.get();
    }
    const auto& p = seq.request.sampling;
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (p.temperature <= 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(p.top_k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(p.top_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(p.seed));
    }
    return samplers_.emplace(seq.id, std::move(chain)).first->second.get();
}

Result<std::vector<Token>> LlamaExecutor::execute(const engine::StepPlan& plan) {
    for (const auto& op : plan.kv_ops) {
        apply(op);
    }

    if (static_cast<std::int32_t>(plan.num_tokens()) > batch_->capacity) {
        return make_error(std::format("step has {} tokens but n_batch is {}", plan.num_tokens(), batch_->capacity),
                          ErrorCode::invalid_argument);
    }

    batch_->clear();
    std::vector<std::int32_t> sample_rows;
    for (const auto& chunk : plan.chunks) {
        const auto& seq = *chunk.seq;
        const auto slot = static_cast<llama_seq_id>(*seq.slot);
        for (std::size_t j = chunk.begin; j < chunk.begin + chunk.count; ++j) {
            const bool last = j + 1 == chunk.begin + chunk.count;
            const auto row = batch_->add(seq.tokens[j], static_cast<llama_pos>(j), slot, chunk.sample && last);
            if (chunk.sample && last) {
                sample_rows.push_back(row);
            }
        }
    }

    if (const int rc = llama_decode(ctx_.get(), batch_->batch); rc != 0) {
        if (rc == 1) {
            return make_error("no KV slot for batch", ErrorCode::kv_exhausted);
        }
        return make_error(std::format("llama_decode failed with code {}", rc));
    }

    std::vector<Token> sampled;
    sampled.reserve(sample_rows.size());
    std::size_t next = 0;
    for (const auto& chunk : plan.chunks) {
        if (chunk.sample) {
            // llama_sampler_sample also accepts the token into the sampler state.
            sampled.push_back(llama_sampler_sample(sampler_for(*chunk.seq), ctx_.get(), sample_rows[next++]));
        }
    }
    return sampled;
}

void LlamaExecutor::release(RequestId id) { samplers_.erase(id); }

void LlamaExecutor::reset() {
    llama_memory_clear(llama_get_memory(ctx_.get()), /*data=*/false);
    samplers_.clear();
}

bool LlamaExecutor::is_eog(Token token) const { return llama_vocab_is_eog(vocab_, token); }

std::string LlamaExecutor::token_to_piece(Token token) const {
    std::array<char, 256> buf{};
    int n = llama_token_to_piece(vocab_, token, buf.data(), static_cast<std::int32_t>(buf.size()), 0, false);
    if (n >= 0) {
        return std::string(buf.data(), static_cast<std::size_t>(n));
    }
    std::string big(static_cast<std::size_t>(-n), '\0');
    n = llama_token_to_piece(vocab_, token, big.data(), static_cast<std::int32_t>(big.size()), 0, false);
    big.resize(static_cast<std::size_t>(std::max(n, 0)));
    return big;
}

Result<std::vector<Token>> LlamaExecutor::tokenize(std::string_view text, bool add_special) const {
    const auto len = static_cast<std::int32_t>(text.size());
    const int needed = -llama_tokenize(vocab_, text.data(), len, nullptr, 0, add_special, true);
    if (needed <= 0) {
        return make_error("tokenization failed", ErrorCode::invalid_argument);
    }
    std::vector<Token> tokens(static_cast<std::size_t>(needed));
    const int n = llama_tokenize(vocab_, text.data(), len, tokens.data(), needed, add_special, true);
    if (n < 0) {
        return make_error("tokenization failed", ErrorCode::invalid_argument);
    }
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

std::string LlamaExecutor::format_chat(std::string_view system, std::string_view user) const {
    const std::string sys(system);
    const std::string usr(user);
    std::vector<llama_chat_message> messages;
    if (!sys.empty()) {
        messages.push_back({"system", sys.c_str()});
    }
    messages.push_back({"user", usr.c_str()});

    const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
    if (tmpl == nullptr) {
        return sys.empty() ? std::format("User: {}\nAssistant:", usr)
                           : std::format("{}\n\nUser: {}\nAssistant:", sys, usr);
    }
    std::string out(2 * (sys.size() + usr.size()) + 256, '\0');
    int n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, out.data(),
                                      static_cast<std::int32_t>(out.size()));
    if (n > static_cast<int>(out.size())) {
        out.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, out.data(), n);
    }
    out.resize(static_cast<std::size_t>(std::max(n, 0)));
    return out;
}

std::string LlamaExecutor::description() const {
    std::array<char, 256> desc{};
    llama_model_desc(model_.get(), desc.data(), desc.size());
    return std::format("{} ({:.2f} GiB, n_ctx={}, slots={})", desc.data(),
                       static_cast<double>(llama_model_size(model_.get())) / (1024.0 * 1024.0 * 1024.0),
                       llama_n_ctx(ctx_.get()), llama_n_seq_max(ctx_.get()));
}

} // namespace avis::llama
