#include "avis/kv/block_manager.hpp"

#include <algorithm>
#include <cassert>
#include <ranges>
#include <stdexcept>

namespace avis::kv {

namespace {

constexpr std::uint64_t kHashSeed = 0x9e3779b97f4a7c15ULL;

constexpr std::uint64_t mix(std::uint64_t x) {
    // splitmix64 finaliser
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

} // namespace

BlockManager::BlockManager(BlockManagerConfig config)
    : config_(config), blocks_(config.num_blocks), slots_(config.num_slots) {
    if (config_.block_size == 0 || config_.num_blocks == 0 || config_.num_slots == 0) {
        throw std::invalid_argument("BlockManager: block_size, num_blocks and num_slots must be > 0");
    }
    free_list_.reserve(config_.num_blocks);
    // Reverse order so blocks are handed out as 0, 1, 2, ...
    for (auto id : std::views::iota(BlockId{0}, static_cast<BlockId>(config_.num_blocks)) | std::views::reverse) {
        free_list_.push_back(id);
    }
}

std::size_t BlockManager::blocks_for(std::size_t num_tokens) const {
    return (num_tokens + config_.block_size - 1) / config_.block_size;
}

std::uint64_t BlockManager::chain_hash(std::uint64_t parent, std::span<const Token> tokens) const {
    std::uint64_t h = mix(parent ^ kHashSeed);
    for (const Token t : tokens) {
        h = mix(h ^ static_cast<std::uint32_t>(t));
    }
    return h;
}

std::optional<SlotId> BlockManager::find_holder(BlockId block, std::size_t index, SlotId exclude) const {
    for (SlotId s = 0; s < slots_.size(); ++s) {
        const auto& slot = slots_[s];
        if (s != exclude && slot.state != SlotState::free && slot.hashes.size() > index &&
            slot.blocks[index] == block) {
            return s;
        }
    }
    return std::nullopt;
}

std::optional<BlockManager::PrefixMatch> BlockManager::find_prefix(std::span<const Token> prompt,
                                                                   SlotId exclude) const {
    if (!config_.enable_prefix_caching || prompt.empty()) {
        return std::nullopt;
    }
    const std::size_t bs = config_.block_size;
    const std::size_t max_blocks = (prompt.size() - 1) / bs;

    std::uint64_t parent = 0;
    std::optional<BlockId> last;
    std::size_t matched = 0;
    for (std::size_t i = 0; i < max_blocks; ++i) {
        const auto chunk = prompt.subspan(i * bs, bs);
        const std::uint64_t h = chain_hash(parent, chunk);
        const auto it = cached_.find(h);
        if (it == cached_.end() || !std::ranges::equal(blocks_[it->second].tokens, chunk)) {
            break;
        }
        last = it->second;
        parent = h;
        matched = i + 1;
    }
    if (matched == 0) {
        return std::nullopt;
    }
    // Any holder of the last matched block holds the whole matched prefix, because
    // block hashes are chained with everything that precedes them.
    const auto donor = find_holder(*last, matched - 1, exclude);
    if (!donor) {
        return std::nullopt;
    }
    return PrefixMatch{*donor, matched};
}

std::optional<SlotId> BlockManager::lru_idle_slot() const {
    std::optional<SlotId> best;
    for (SlotId s = 0; s < slots_.size(); ++s) {
        if (slots_[s].state == SlotState::idle &&
            (!best || slots_[s].last_used < slots_[*best].last_used)) {
            best = s;
        }
    }
    return best;
}

std::optional<Admission> BlockManager::admit(std::span<const Token> prompt) {
    constexpr auto kNoSlot = static_cast<SlotId>(-1);
    ++stats_.prefix_lookups;
    const auto match = find_prefix(prompt, kNoSlot);

    // Best case: the prefix lives in an idle slot, so take that slot over and
    // keep its KV in place without copying anything.
    if (match && slots_[match->donor].state == SlotState::idle) {
        auto& slot = slots_[match->donor];
        truncate_blocks(slot, match->num_blocks);
        ops_.push_back({KvOpKind::truncate, match->donor, match->donor, match->num_blocks * config_.block_size});
        slot.state = SlotState::active;
        slot.last_used = ++tick_;
        ++stats_.prefix_hits;
        stats_.prefix_hit_tokens += match->num_blocks * config_.block_size;
        return Admission{match->donor, match->num_blocks * config_.block_size};
    }

    std::optional<SlotId> chosen;
    for (SlotId s = 0; s < slots_.size(); ++s) {
        if (slots_[s].state == SlotState::free) {
            chosen = s;
            break;
        }
    }
    if (!chosen) {
        chosen = lru_idle_slot();
        if (!chosen) {
            return std::nullopt;
        }
        evict(*chosen);
    }

    auto& slot = slots_[*chosen];
    slot.state = SlotState::active;
    slot.last_used = ++tick_;

    std::size_t cached_tokens = 0;
    if (match) {
        // The donor is an active sequence: share its blocks.
        const auto& donor = slots_[match->donor];
        for (std::size_t i = 0; i < match->num_blocks; ++i) {
            ++blocks_[donor.blocks[i]].ref_count;
            slot.blocks.push_back(donor.blocks[i]);
            slot.hashes.push_back(donor.hashes[i]);
        }
        cached_tokens = match->num_blocks * config_.block_size;
        ops_.push_back({KvOpKind::copy_prefix, match->donor, *chosen, cached_tokens});
        ++stats_.prefix_hits;
        stats_.prefix_hit_tokens += cached_tokens;
    }
    return Admission{*chosen, cached_tokens};
}

bool BlockManager::reserve(SlotId slot_id, std::size_t total_tokens) {
    auto& slot = slots_.at(slot_id);
    assert(slot.state == SlotState::active);
    const std::size_t want = blocks_for(total_tokens);
    if (want <= slot.blocks.size()) {
        return true;
    }
    const std::size_t needed = want - slot.blocks.size();
    while (free_list_.size() < needed) {
        const auto victim = lru_idle_slot();
        if (!victim) {
            return false;
        }
        evict(*victim);
    }
    for (std::size_t i = 0; i < needed; ++i) {
        const BlockId id = free_list_.back();
        free_list_.pop_back();
        blocks_[id].ref_count = 1;
        slot.blocks.push_back(id);
    }
    slot.last_used = ++tick_;
    return true;
}

void BlockManager::commit(SlotId slot_id, std::span<const Token> tokens, std::size_t num_computed) {
    auto& slot = slots_.at(slot_id);
    const std::size_t bs = config_.block_size;
    const std::size_t full = std::min(num_computed / bs, slot.blocks.size());
    while (slot.hashes.size() < full) {
        const std::size_t i = slot.hashes.size();
        const auto chunk = tokens.subspan(i * bs, bs);
        const std::uint64_t parent = i == 0 ? 0 : slot.hashes.back();
        const std::uint64_t h = chain_hash(parent, chunk);
        slot.hashes.push_back(h);
        auto& block = blocks_[slot.blocks[i]];
        if (!block.hash) {
            block.hash = h;
            if (config_.enable_prefix_caching) {
                block.tokens.assign(chunk.begin(), chunk.end());
                cached_.try_emplace(h, slot.blocks[i]);
            }
        }
    }
}

void BlockManager::release(SlotId slot_id, bool retain_cache) {
    auto& slot = slots_.at(slot_id);
    if (slot.state != SlotState::active) {
        return;
    }
    if (retain_cache && config_.enable_prefix_caching && !slot.hashes.empty()) {
        // Keep only the full, hashed blocks; the partial tail cannot be shared.
        const std::size_t keep = slot.hashes.size();
        if (slot.blocks.size() > keep) {
            truncate_blocks(slot, keep);
        }
        ops_.push_back({KvOpKind::truncate, slot_id, slot_id, keep * config_.block_size});
        slot.state = SlotState::idle;
        slot.last_used = ++tick_;
        return;
    }
    clear_slot(slot_id);
}

void BlockManager::truncate_blocks(Slot& slot, std::size_t keep) {
    while (slot.blocks.size() > keep) {
        unref(slot.blocks.back());
        slot.blocks.pop_back();
    }
    if (slot.hashes.size() > keep) {
        slot.hashes.resize(keep);
    }
}

void BlockManager::evict(SlotId slot_id) {
    clear_slot(slot_id);
    ++stats_.evictions;
}

void BlockManager::clear_slot(SlotId slot_id) {
    auto& slot = slots_[slot_id];
    truncate_blocks(slot, 0);
    slot.state = SlotState::free;
    ops_.push_back({KvOpKind::clear, slot_id, slot_id, 0});
}

void BlockManager::unref(BlockId id) {
    auto& block = blocks_[id];
    assert(block.ref_count > 0);
    if (--block.ref_count > 0) {
        return;
    }
    if (block.hash) {
        if (const auto it = cached_.find(*block.hash); it != cached_.end() && it->second == id) {
            cached_.erase(it);
        }
        block.hash.reset();
        block.tokens.clear();
    }
    free_list_.push_back(id);
}

std::vector<KvOp> BlockManager::take_ops() { return std::exchange(ops_, {}); }

std::size_t BlockManager::available_blocks() const {
    // A block is reclaimable when every reference to it comes from idle slots.
    std::unordered_map<BlockId, std::uint32_t> idle_refs;
    for (const auto& slot : slots_) {
        if (slot.state == SlotState::idle) {
            for (const BlockId b : slot.blocks) {
                ++idle_refs[b];
            }
        }
    }
    std::size_t reclaimable = 0;
    for (const auto& [block, refs] : idle_refs) {
        if (blocks_[block].ref_count == refs) {
            ++reclaimable;
        }
    }
    return free_list_.size() + reclaimable;
}

std::size_t BlockManager::active_slots() const {
    return static_cast<std::size_t>(
        std::ranges::count(slots_, SlotState::active, &Slot::state));
}

std::size_t BlockManager::idle_slots() const {
    return static_cast<std::size_t>(std::ranges::count(slots_, SlotState::idle, &Slot::state));
}

std::span<const BlockId> BlockManager::block_table(SlotId slot) const { return slots_.at(slot).blocks; }

} // namespace avis::kv
