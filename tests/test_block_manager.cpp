#include "avis/kv/block_manager.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

using namespace avis;
using namespace avis::kv;

namespace {

std::vector<Token> iota_tokens(std::size_t n, Token start = 0) {
    std::vector<Token> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

BlockManagerConfig small_config() {
    return {.num_blocks = 16, .block_size = 4, .num_slots = 4, .enable_prefix_caching = true};
}

// Admits `tokens`, reserves room for all of them and marks them computed.
SlotId admit_and_compute(BlockManager& bm, const std::vector<Token>& tokens) {
    auto adm = bm.admit(tokens);
    EXPECT_TRUE(adm.has_value());
    EXPECT_TRUE(bm.reserve(adm->slot, tokens.size()));
    bm.commit(adm->slot, tokens, tokens.size());
    return adm->slot;
}

} // namespace

TEST(BlockManager, ReserveAllocatesWholeBlocks) {
    BlockManager bm(small_config());
    auto adm = bm.admit(iota_tokens(10));
    ASSERT_TRUE(adm);
    EXPECT_EQ(adm->cached_tokens, 0u);
    ASSERT_TRUE(bm.reserve(adm->slot, 10));
    EXPECT_EQ(bm.block_table(adm->slot).size(), 3u);
    EXPECT_EQ(bm.free_blocks(), 13u);

    // Growing within the last block needs nothing new.
    ASSERT_TRUE(bm.reserve(adm->slot, 12));
    EXPECT_EQ(bm.free_blocks(), 13u);
    ASSERT_TRUE(bm.reserve(adm->slot, 13));
    EXPECT_EQ(bm.free_blocks(), 12u);
}

TEST(BlockManager, ReleaseWithoutCacheFreesEverything) {
    BlockManager bm(small_config());
    const auto slot = admit_and_compute(bm, iota_tokens(10));
    (void)bm.take_ops();
    bm.release(slot, /*retain_cache=*/false);
    EXPECT_EQ(bm.free_blocks(), 16u);
    EXPECT_EQ(bm.active_slots(), 0u);
    const auto ops = bm.take_ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0].kind, KvOpKind::clear);
    EXPECT_EQ(bm.stats().evictions, 0u);
}

TEST(BlockManager, ReserveFailsWithoutSideEffectsWhenFull) {
    BlockManager bm(small_config());
    auto a = bm.admit(iota_tokens(4));
    ASSERT_TRUE(bm.reserve(a->slot, 60)); // 15 blocks
    auto b = bm.admit(iota_tokens(4, 100));
    ASSERT_TRUE(b);
    EXPECT_FALSE(bm.reserve(b->slot, 8)); // needs 2, only 1 left
    EXPECT_EQ(bm.free_blocks(), 1u);
    EXPECT_TRUE(bm.block_table(b->slot).empty());
}

TEST(BlockManager, SharesPrefixWithActiveSequence) {
    BlockManager bm(small_config());
    const auto prefix = iota_tokens(8); // two full blocks
    auto first = prefix;
    first.insert(first.end(), {500, 501});
    const auto donor = admit_and_compute(bm, first);
    (void)bm.take_ops();
    const std::size_t free_before = bm.free_blocks();

    auto second = prefix;
    second.insert(second.end(), {900, 901, 902});
    auto adm = bm.admit(second);
    ASSERT_TRUE(adm);
    EXPECT_NE(adm->slot, donor);
    EXPECT_EQ(adm->cached_tokens, 8u);
    // The two shared blocks are referenced, not copied.
    EXPECT_EQ(bm.free_blocks(), free_before);
    EXPECT_EQ(bm.block_table(adm->slot)[0], bm.block_table(donor)[0]);
    EXPECT_EQ(bm.block_table(adm->slot)[1], bm.block_table(donor)[1]);

    const auto ops = bm.take_ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0], (KvOp{KvOpKind::copy_prefix, donor, adm->slot, 8}));

    // Only the new tail needs a block.
    ASSERT_TRUE(bm.reserve(adm->slot, second.size()));
    EXPECT_EQ(bm.free_blocks(), free_before - 1);
    EXPECT_EQ(bm.stats().prefix_hits, 1u);
    EXPECT_EQ(bm.stats().prefix_hit_tokens, 8u);
}

TEST(BlockManager, AlwaysLeavesOneTokenToCompute) {
    BlockManager bm(small_config());
    const auto tokens = iota_tokens(8);
    admit_and_compute(bm, tokens);
    // Identical prompt that is an exact multiple of the block size: the last
    // block cannot be reused or there would be nothing to produce logits from.
    auto adm = bm.admit(tokens);
    ASSERT_TRUE(adm);
    EXPECT_EQ(adm->cached_tokens, 4u);
}

TEST(BlockManager, TakesOverIdleSlotInPlace) {
    BlockManager bm(small_config());
    auto first = iota_tokens(10);
    const auto slot = admit_and_compute(bm, first);
    bm.release(slot, /*retain_cache=*/true);
    EXPECT_EQ(bm.idle_slots(), 1u);
    // The partial third block is dropped, the two full ones stay cached.
    EXPECT_EQ(bm.block_table(slot).size(), 2u);
    (void)bm.take_ops();

    auto second = iota_tokens(8);
    second.push_back(77);
    auto adm = bm.admit(second);
    ASSERT_TRUE(adm);
    EXPECT_EQ(adm->slot, slot);
    EXPECT_EQ(adm->cached_tokens, 8u);
    const auto ops = bm.take_ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0], (KvOp{KvOpKind::truncate, slot, slot, 8}));
    EXPECT_EQ(bm.idle_slots(), 0u);
}

TEST(BlockManager, EvictsLeastRecentlyUsedIdleSlot) {
    BlockManager bm({.num_blocks = 16, .block_size = 4, .num_slots = 2, .enable_prefix_caching = true});
    const auto a = admit_and_compute(bm, iota_tokens(8, 0));
    const auto b = admit_and_compute(bm, iota_tokens(8, 100));
    bm.release(a, true);
    bm.release(b, true);
    (void)bm.take_ops();

    auto adm = bm.admit(iota_tokens(8, 1000)); // no prefix match, no free slot
    ASSERT_TRUE(adm);
    EXPECT_EQ(adm->slot, a);
    EXPECT_EQ(bm.stats().evictions, 1u);
    const auto ops = bm.take_ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0].kind, KvOpKind::clear);
    EXPECT_EQ(ops[0].dst, a);
}

TEST(BlockManager, ReserveEvictsIdleSlotsForMemory) {
    BlockManager bm(small_config());
    const auto idle = admit_and_compute(bm, iota_tokens(32)); // 8 blocks
    bm.release(idle, true);
    EXPECT_EQ(bm.free_blocks(), 8u);
    EXPECT_EQ(bm.available_blocks(), 16u);

    auto adm = bm.admit(iota_tokens(4, 500));
    ASSERT_TRUE(bm.reserve(adm->slot, 48)); // 12 blocks: must evict the idle slot
    EXPECT_EQ(bm.idle_slots(), 0u);
    EXPECT_EQ(bm.free_blocks(), 4u);
}

TEST(BlockManager, SharedBlocksAreNotReclaimableWhileActive) {
    BlockManager bm(small_config());
    const auto prefix = iota_tokens(8);
    auto first = prefix;
    first.push_back(1);
    const auto donor = admit_and_compute(bm, first); // 3 blocks
    auto second = prefix;
    second.push_back(2);
    auto adm = bm.admit(second);
    ASSERT_TRUE(bm.reserve(adm->slot, second.size())); // shares 2, adds 1
    bm.release(donor, true); // keeps 2 blocks, both also held by the active sequence
    EXPECT_EQ(bm.free_blocks(), 16u - 4u + 1u);
    EXPECT_EQ(bm.available_blocks(), bm.free_blocks());
}

TEST(BlockManager, PrefixCachingCanBeDisabled) {
    auto cfg = small_config();
    cfg.enable_prefix_caching = false;
    BlockManager bm(cfg);
    const auto tokens = iota_tokens(12);
    const auto slot = admit_and_compute(bm, tokens);
    bm.release(slot, true);
    EXPECT_EQ(bm.idle_slots(), 0u);
    auto adm = bm.admit(tokens);
    ASSERT_TRUE(adm);
    EXPECT_EQ(adm->cached_tokens, 0u);
}

TEST(BlockManager, RejectsZeroSizedConfig) {
    EXPECT_THROW(BlockManager({.num_blocks = 0, .block_size = 4, .num_slots = 1}), std::invalid_argument);
}
