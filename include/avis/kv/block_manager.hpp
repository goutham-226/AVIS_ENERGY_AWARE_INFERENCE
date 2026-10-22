#pragma once

#include "avis/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace avis::kv {

using BlockId = std::uint32_t;

struct BlockManagerConfig {
    std::size_t num_blocks = 1024; // KV capacity in blocks
    std::size_t block_size = 16;   // tokens per block
    std::size_t num_slots = 16;    // backend sequence ids available (active + cached)
    bool enable_prefix_caching = true;
};

// Physical KV operations the backend must apply, in order, before the next forward pass.
enum class KvOpKind : std::uint8_t {
    copy_prefix, // share positions [0, num_tokens) of src with dst
    truncate,    // drop positions >= num_tokens of dst
    clear,       // drop every position of dst
};

struct KvOp {
    KvOpKind kind;
    SlotId src = 0;
    SlotId dst = 0;
    std::size_t num_tokens = 0;

    bool operator==(const KvOp&) const = default;
};

struct Admission {
    SlotId slot = 0;
    std::size_t cached_tokens = 0; // prompt tokens whose KV is reused from the prefix cache
};

struct BlockManagerStats {
    std::size_t prefix_hit_tokens = 0;
    std::size_t prefix_lookups = 0;
    std::size_t prefix_hits = 0;
    std::size_t evictions = 0;
};

// Paged KV-cache accounting in the style of vLLM's block manager.
//
// KV memory is split into fixed-size blocks. Every backend sequence slot owns a
// block table, blocks are reference counted, and full blocks are content-hashed
// (chained with their prefix) so requests that share a prompt prefix share the
// same physical blocks. Finished sequences can be retained as "idle" slots that
// keep their prefix cached until the memory or the slot is needed again.
class BlockManager {
public:
    explicit BlockManager(BlockManagerConfig config);

    // Picks a slot for a new sequence whose tokens are `prompt` and reuses the
    // longest cached block-aligned prefix (always leaving at least one token to
    // compute). Returns nullopt when every slot is busy with an active sequence.
    [[nodiscard]] std::optional<Admission> admit(std::span<const Token> prompt);

    // Makes sure `slot` has blocks for `total_tokens` tokens, evicting idle
    // slots if needed. Returns false, changing nothing, if memory is exhausted.
    [[nodiscard]] bool reserve(SlotId slot, std::size_t total_tokens);

    // Records that the KV of tokens[0, num_computed) of `slot` is materialised,
    // making its full blocks available to the prefix cache.
    void commit(SlotId slot, std::span<const Token> tokens, std::size_t num_computed);

    // Ends a sequence. With retain_cache the slot becomes idle and keeps its
    // full blocks cached; otherwise its memory is released immediately.
    void release(SlotId slot, bool retain_cache);

    // Drains the pending backend operations.
    [[nodiscard]] std::vector<KvOp> take_ops();

    [[nodiscard]] std::size_t blocks_for(std::size_t num_tokens) const;
    [[nodiscard]] std::size_t free_blocks() const { return free_list_.size(); }
    // Free blocks plus blocks that evicting every idle slot would return.
    [[nodiscard]] std::size_t available_blocks() const;
    [[nodiscard]] std::size_t num_blocks() const { return config_.num_blocks; }
    [[nodiscard]] std::size_t block_size() const { return config_.block_size; }
    [[nodiscard]] std::size_t active_slots() const;
    [[nodiscard]] std::size_t idle_slots() const;
    [[nodiscard]] std::span<const BlockId> block_table(SlotId slot) const;
    [[nodiscard]] const BlockManagerStats& stats() const { return stats_; }

private:
    enum class SlotState : std::uint8_t { free, active, idle };

    struct Slot {
        SlotState state = SlotState::free;
        std::vector<BlockId> blocks;
        std::vector<std::uint64_t> hashes; // hashes of the committed (full) blocks
        std::uint64_t last_used = 0;
    };

    struct Block {
        std::uint32_t ref_count = 0;
        std::optional<std::uint64_t> hash;
        std::vector<Token> tokens; // contents of a hashed block, used to reject hash collisions
    };

    struct PrefixMatch {
        SlotId donor = 0;
        std::size_t num_blocks = 0;
    };

    [[nodiscard]] std::uint64_t chain_hash(std::uint64_t parent, std::span<const Token> tokens) const;
    [[nodiscard]] std::optional<PrefixMatch> find_prefix(std::span<const Token> prompt, SlotId exclude) const;
    [[nodiscard]] std::optional<SlotId> find_holder(BlockId block, std::size_t index, SlotId exclude) const;
    [[nodiscard]] std::optional<SlotId> lru_idle_slot() const;
    void truncate_blocks(Slot& slot, std::size_t keep);
    void evict(SlotId slot); // drops an idle slot's cache to make room
    void clear_slot(SlotId slot);
    void unref(BlockId block);

    BlockManagerConfig config_;
    std::vector<Block> blocks_;
    std::vector<BlockId> free_list_;
    std::vector<Slot> slots_;
    std::unordered_map<std::uint64_t, BlockId> cached_;
    std::vector<KvOp> ops_;
    std::uint64_t tick_ = 0;
    BlockManagerStats stats_;
};

} // namespace avis::kv
