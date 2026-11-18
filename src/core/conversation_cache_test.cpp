#include "strata/core/conversation_cache.hpp"

#include "strata/core/conversation_prefix.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
SavedConversation image_with_checkpoint(std::initializer_list<int32_t> live_ids,
                                        std::initializer_list<int32_t> ck_ids) {
    SavedConversation s = image(live_ids);
    ConversationCheckpoint c;
    c.ids = ck_ids;
    c.gdn.resize(64, 7);
    s.checkpoints.push_back(std::move(c));
    return s;
}

void test_root_priority() {
    // The shared system-prompt prefix: parked pinned, deduplicated, evicted after conversations, and
    // preferred at an equal-length tie because it is the smaller image.
    const size_t one = image({1, 2}).bytes();
    {
        ConversationCache cache(one * 8, 8);
        check(cache.put(image({1, 2}), 0, true), "the root prefix parks pinned");
        check(cache.put(image({5, 6, 7})), "a chat parks unpinned");
        check(cache.size() == 2, "pinned and unpinned entries coexist");
        check(cache.put(image({1, 2}), 0, true), "a repeat of the same prefix parks a newer pinned copy");
        check(cache.size() == 3, "a repeat parks again; an equal-length tie prefers the newest");
        check(cache.has({1, 2}, {}, true), "has() finds the pinned root");
        check(!cache.has({1, 2}, {}, false), "has() respects the steering mode");
    }
    {
        ConversationCache cache(one * 3, 8);
        check(cache.put(image({1, 2}), 0, true), "the root parks pinned");
        check(cache.put(image({5, 6, 7})), "a first chat parks");
        check(cache.put(image({9, 9, 9, 9})), "a second chat parks and forces an eviction");
        check(cache.best(std::vector<int64_t>{1, 2, 9}, {}, true).tokens == 2, "the pinned root survives the eviction");
        check(cache.best(std::vector<int64_t>{5, 6, 7, 8}, {}, true).tokens == 0, "the unpinned chat was evicted first");
        check(cache.best(std::vector<int64_t>{9, 9, 9, 9, 1}, {}, true).tokens == 4, "the newest chat still parks");
        check(cache.evictions() == 1, "one eviction, and it was the unpinned conversation");
    }
    {
        ConversationCache cache(one * 6, 8);
        check(cache.put(image_with_checkpoint({1, 2, 3}, {1, 2})), "a conversation with a {1,2} checkpoint parks");
        check(cache.put(image({1, 2}), 0, true), "the pinned {1,2} root parks");
        const ConversationCache::Match m = cache.best(std::vector<int64_t>{1, 2, 9}, {}, true);
        check(m.tokens == 2 && cache.take(m.index).live.ids == std::vector<int32_t>({1, 2}),
              "an equal-length tie prefers the smaller pinned root over the containing conversation");
    }
    {
        ConversationCache cache(one * 4, 2);
        check(cache.put(image({1, 2}), 0, true), "the pinned root parks");
        check(cache.put(image({1, 3434, 3434, 55})), "an unpinned chat parks");
        check(cache.put(image({7, 7, 7})), "a third image against a slot cap evicts the chat, not the root");
        check(cache.best(std::vector<int64_t>{1, 2, 9}, {}, true).tokens == 2, "the pinned root outlives the slot cap");
    }
    {
        ConversationCache cache(one * 8, 2);
        check(cache.put(image({1, 2}), 0, true), "first root parks");
        check(cache.put(image({3, 4}), 0, true), "second root fills the slot cap");
        check(cache.put(image({5, 6})), "a conversation displaces a root when all slots hold roots");
        check(!cache.has({1, 2}, {}, true) && cache.has({3, 4}, {}, true),
              "slot pressure evicts the oldest root");
        const auto match_a = cache.best(std::vector<int32_t>{5, 6, 9}, {}, true);
        check(match_a.tokens == 2, "the first switch finds the conversation in RAM");
        auto a = cache.take(match_a.index);
        check(cache.put(image({7, 8}), a.bytes()), "the other conversation parks during the exchange");
        const auto match_b = cache.best(std::vector<int32_t>{7, 8, 9}, {}, true);
        check(match_b.tokens == 2, "the next switch finds the other conversation in RAM");
        auto b = cache.take(match_b.index);
        check(cache.put(std::move(a), b.bytes()), "the first conversation parks on the return switch");
        check(cache.best(std::vector<int32_t>{5, 6, 9}, {}, true).tokens == 2,
              "the return switch finds the first conversation in RAM");
    }
    {
        ConversationCache cache(one * 2, 8);
        check(cache.put(image({1, 2}), 0, true), "first root fits the byte budget");
        check(cache.put(image({3, 4}), 0, true), "second root fills the byte budget");
        check(cache.put(image({5, 6})), "byte pressure can evict a root");
        check(!cache.has({1, 2}, {}, true) && cache.has({3, 4}, {}, true) &&
              cache.has({5, 6}, {}, true) && cache.bytes() <= one * 2,
              "oldest root leaves and the byte budget remains bounded");
    }
    {
        ConversationCache cache(one * 16, 2);
        auto protected_image = image_with_checkpoint({1, 2, 3}, {1, 2});
        protected_image.checkpoints.back().pinned = true;
        check(cache.put(std::move(protected_image)), "explicit checkpoint pin parks");
        check(cache.put(image({3, 4}), 0, true), "root parks beside the checkpoint pin");
        check(cache.put(image({5, 6})), "root fallback preserves an explicit checkpoint pin");
        check(cache.has({1, 2, 3}, {}, true) && !cache.has({3, 4}, {}, true),
              "only the root is evicted");
        check(cache.evict_oldest() && !cache.evict_oldest(),
              "explicit checkpoint pin remains protected when no victim is available");
    }
}
// A parked conversation and the fork of it share one physical prefix.  These are
// the boundaries the parking path depends on: what a copy costs, what a write
// detaches, what the cache charges, and what a bounded retention keeps.
void test_shared_prefix() {
    {   // ConversationBuffer: a prefix trim adopts pieces; a write copies only what it touches
        ConversationBuffer source;
        source.resize(ConversationBuffer::segment_bytes + 32, 7);
        const size_t cut = ConversationBuffer::segment_bytes + 8;
        ConversationBuffer fork;
        fork.share_prefix(source, cut);
        const uint8_t* full = nullptr;
        const uint8_t* shared = nullptr;
        const ConversationBuffer& shared_source = source;
        const ConversationBuffer& shared_fork = fork;
        shared_source.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { full = p; return true; });
        shared_fork.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { shared = p; return true; });
        check(fork.size() == cut && full == shared, "a prefix trim adopts the allocation instead of copying it");
        check(fork.detach_peak(0, cut) == cut, "the whole trimmed prefix is still shared");
        fork.resize(cut + 8, 9);
        check(fork.detach_peak(0, cut) == cut && fork.detach_peak(cut, 8) == 0,
              "appending to a shared piece allocates a new one and copies no prefix byte");
        std::vector<uint8_t> tail(8, 0);
        check(fork.read(tail.data(), cut, 8) && tail[0] == 9 && tail[7] == 9,
              "the fork appends its own suffix");
        check(source.detach_peak(0, 1) > 0 && source.read(tail.data(), cut, 8) && tail[0] == 7,
              "the fork's append cannot alter the original");
        check(source.size() == ConversationBuffer::segment_bytes + 32 && fork.size() == cut + 8,
              "each side keeps its own extent");
        // a dirty tail inside a shared piece copies that tail alone
        const size_t dirty = 64;
        const size_t at = cut - dirty;                       // inside the shared prefix
        check(fork.detach_peak(at, dirty) == dirty, "the dirty tail is shared before the write");
        check(fork.visit(at, dirty, [](uint8_t* p, size_t n, size_t) {
                  std::memset(p, 9, n);
                  return true;
              }), "the dirty tail is written");
        check(fork.detach_peak(0, at) == at, "the unchanged prefix is still shared, not copied");
        check(fork.detach_peak(at, dirty) == 0, "and the written tail alone is private");
        std::vector<uint8_t> probe(dirty, 0);
        check(fork.read(probe.data(), at, dirty) && probe[0] == 9 && probe[dirty - 1] == 9,
              "the write landed in the tail");
        check(source.read(probe.data(), at, dirty) && probe[0] == 7,
              "the source's own tail is untouched");
    }
    {
        ConversationBuffer source;
        source.resize(509);
        std::vector<uint8_t> original(509);
        for (size_t i = 0; i < original.size(); ++i) original[i] = uint8_t(i % 251);
        source.visit(0, original.size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, original.data() + at, n);
            return true;
        });
        auto branch = source;
        auto expected = original;
        for (const auto [at, count] : {std::pair<size_t, size_t>{101, 23}, {260, 17}}) {
            branch.visit(at, count, [](uint8_t* p, size_t n, size_t) {
                std::memset(p, 42, n);
                return true;
            });
            std::fill(expected.begin() + at, expected.begin() + at + count, 42);
        }
        std::vector<uint8_t> actual(original.size());
        check(branch.read(actual.data(), 0, actual.size()) && actual == expected,
              "middle writes preserve the original bytes on both sides of each dirty range");
        check(source.read(actual.data(), 0, actual.size()) && actual == original,
              "middle writes leave the complete source unchanged");
        size_t allocations = 0, allocation_bytes = 0;
        branch.visit_allocations([&](const void*, size_t bytes) { ++allocations; allocation_bytes += bytes; });
        check(allocations == 3 && allocation_bytes == 509 + 23 + 17 &&
                  branch.allocation_bytes() == allocation_bytes,
              "non-adjacent ranges of one allocation are charged once");
        ConversationBuffer contiguous;
        contiguous.resize(expected.size());
        contiguous.visit(0, expected.size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, expected.data() + at, n);
            return true;
        });
        check(branch == contiguous, "equality reads each split range at its allocation offset");
        branch.resize(264);
        branch.resize(272, 91);
        expected.resize(264);
        expected.resize(272, 91);
        actual.resize(expected.size());
        check(branch.read(actual.data(), 0, actual.size()) && actual == expected,
              "rewind and regrowth preserve split ranges and fill only the new suffix");
        const auto& readonly = branch;
        readonly.visit(0, readonly.size(), [](auto* p, size_t, size_t) {
            static_assert(std::is_const_v<std::remove_pointer_t<decltype(p)>>);
            return true;
        });
        branch.share_prefix(branch, 130);
        expected.resize(130);
        actual.resize(130);
        check(branch.read(actual.data(), 0, actual.size()) && actual == expected,
              "a self prefix trim retains the requested bytes");
    }
    {
        const size_t prefix = ConversationBuffer::segment_bytes;
        ConversationCache cache(prefix + (1 << 20), 4);
        auto source = image({1, 2, 3});
        source.kv.resize(1);
        source.kv[0].k.resize(prefix, 7);
        check(cache.put(std::move(source)), "the original large prefix fits in the cache");
        auto branch = cache.fork(0);
        branch.live.ids.push_back(4);
        branch.kv[0].k.resize(prefix + 8, 9);
        check(cache.put(std::move(branch)) && cache.size() == 2,
              "a short divergent suffix fits without a second prefix-sized allocation or source eviction");
    }
    {
        auto source = image({1, 2, 3});
        source.checkpoints.push_back(source.live);
        const size_t payload = source.live.gdn.capacity();
        const size_t budget = source.private_bytes() + payload;
        ConversationCache cache(budget, 4);
        check(cache.put(std::move(source)) && cache.bytes() == budget,
              "one state shared by live and checkpoint fits a one-copy payload budget");
        const auto branch = cache.fork(0);
        check(cache.shared_bytes(branch) == payload,
              "the shared-byte query counts repeated checkpoint references once");
    }
    for (const bool supersede : {false, true}) {
        const size_t budget = 14 << 10;
        ConversationCache cache(budget, 3);
        auto source = image({1, 2, 3});
        source.checkpoint_archive = supersede;
        source.kv.resize(1);
        source.kv[0].k.resize(6 << 10, 7);
        check(cache.put(std::move(source)), "the shared-prefix owner parks before admission pressure");
        auto unrelated = image({8, 9});
        unrelated.kv.resize(1);
        unrelated.kv[0].k.resize(6 << 10, 8);
        check(cache.put(std::move(unrelated)), "an unrelated prefix fills the remaining cache space");
        auto branch = cache.fork(0);
        if (supersede) branch.checkpoints.push_back(branch.live);
        branch.live.ids.push_back(4);
        branch.live.gdn.resize(3 << 10, 9);
        check(cache.put(std::move(branch)) && cache.bytes() <= budget && cache.size() == 1,
              "removing a shared prefix owner does not free bytes still held by the incoming branch");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true).tokens == 4,
              "the admitted branch remains restorable after its source and unrelated entry leave");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto source = image({});
        for (int32_t i = 0; i < 100; ++i) source.live.ids.push_back(i);
        auto stripped = source.live;
        stripped.ids.resize(20);
        stripped.drop_state();
        source.checkpoints.push_back(std::move(stripped));
        check(cache.put(std::move(source)), "a long conversation parks without an early running-state anchor");
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 20; ++i) prompt.push_back(i);
        prompt.push_back(999);
        check(cache.best(prompt, {}, true).tokens == 0 && cache.shared_prefix(prompt, {}, true).prefix_tokens == 20,
              "a stateless position cannot restore, but its backing still shares the full matching K/V prefix");
        auto shorter = image({});
        shorter.live.ids.assign(prompt.begin(), prompt.begin() + 15);
        check(cache.put(std::move(shorter)), "a separate shorter restorable prefix parks");
        const auto backing = cache.shared_prefix(prompt, {}, true);
        check(cache.best(prompt, {}, true).tokens == 15 && backing.index == 0 && backing.prefix_tokens == 20,
              "the nearest running-state anchor and longest shared K/V backing are selected independently");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto source = image({1, 2, 3});
        source.kv.resize(1);
        source.kv[0].k.resize(1 << 16, 5);
        check(cache.put(std::move(source)), "a prefix parks before retained-buffer reuse");
        const size_t alone = cache.bytes();
        auto restored = cache.fork(0);
        cache.retain(std::move(restored.kv), 3);
        auto reuse = cache.take_reuse();
        check(cache.bytes() == alone, "taking shared retained buffers removes only their private directory charge");
        reuse.kv[0].k.visit(100, 17, [](uint8_t* p, size_t n, size_t) {
            std::memset(p, 9, n);
            return true;
        });
        uint8_t original[17]{};
        const auto preserved = cache.fork(0);
        check(cache.bytes() == alone && preserved.kv[0].k.read(original, 100, sizeof(original)) &&
                  original[0] == 5 && original[16] == 5,
              "an external retained-buffer rewrite leaves the parked prefix and its charge unchanged");
        auto branch = image({1, 2, 3, 4});
        branch.kv = std::move(reuse.kv);
        const size_t expected = alone + branch.private_bytes() + branch.live.gdn.capacity() + 17;
        check(cache.put(std::move(branch)) && cache.bytes() == expected,
              "parking the rewritten retained branch charges its new state and dirty range, not the prefix again");
        check(cache.evict_oldest() && cache.evict_oldest() && cache.bytes() == 0,
              "releasing both retained-buffer branches removes every cache allocation charge");
    }
    {   // ConversationPayload: a checkpoint copy shares its running state; a write detaches it
        ConversationCheckpoint first;
        first.ids = {1, 2, 3};
        first.gdn.assign(64, 7);
        const ConversationCheckpoint& shared_first = first;
        ConversationCheckpoint second = first;
        const ConversationCheckpoint& shared_second = second;
        const uint8_t* payload = shared_first.gdn.data();
        check(shared_second.gdn.data() == payload && second.gdn == first.gdn,
              "a checkpoint copy shares its payload");
        second.gdn.assign(64, 9);
        check(shared_first.gdn.data() == payload && shared_first.gdn.vec()[0] == 7 && second.gdn.vec()[0] == 9,
              "writing the copy leaves the original payload alone");
        // writable_all: a checkpoint save overwrites every byte, so a shared block is replaced, not copied
        ConversationCheckpoint third = first;
        uint8_t* const fresh = third.gdn.writable_all();
        check(fresh != nullptr && fresh != payload && third.gdn.vec().size() == first.gdn.vec().size(),
              "writable_all detaches a shared payload into a block of the same size");
        check(shared_first.gdn.data() == payload && shared_first.gdn.vec()[0] == 7,
              "writable_all leaves the shared original alone");
        check(third.gdn.writable_all() == fresh, "writable_all keeps its own detached block");
        fresh[0] = 11;
        check(first.gdn.vec()[0] == 7 && third.gdn.vec()[0] == 11,
              "a write through writable_all lands in the detached block");
        check(!ConversationCheckpoint{}.has_state() && first.has_state(), "has_state() reports a captured state");
        ConversationCheckpoint stripped = first;
        stripped.drop_state();
        check(!stripped.has_state() && stripped.ids == first.ids, "a dropped state keeps the matching position");
    }
    {   // the cache charges a shared prefix once; the newer branch leaves first
        ConversationCache cache(1 << 20, 4);
        auto source = image({1, 2, 3});
        source.kv.resize(1);
        source.kv[0].k.resize(1 << 16, 5);
        check(cache.put(std::move(source)), "a conversation with a K/V backbone parks");
        const size_t alone = cache.bytes();
        SavedConversation fork = cache.fork(0);
        const size_t metadata = fork.private_bytes();
        check(fork.kv[0].k.size() == (1 << 16) && cache.bytes() == alone,
              "a fork copies no payload and does not change the cache's charge");
        check(cache.shared_bytes(fork) == fork.allocation_bytes() && fork.allocation_bytes() >= (1 << 16),
              "every allocation of the fork is already charged to the parked source");
        check(cache.make_room(fork), "a fork that shares the prefix needs no room of its own");
        check(cache.put(std::move(fork)), "the fork parks beside its source");
        check(cache.bytes() == alone + metadata, "the shared K/V is charged once; only new metadata is added");
        check(cache.take(1).kv[0].k.size() == (1 << 16) && cache.bytes() >= (1 << 16),
              "the shared prefix stays charged to the branch that is left");
        check(cache.take(0).kv[0].k.size() == (1 << 16) && cache.bytes() == 0,
              "the last owner frees the shared prefix");
    }
    {   // the same in the other order: the older branch goes first
        ConversationCache cache(1 << 20, 4);
        auto source = image({1, 2, 3});
        source.kv.resize(1);
        source.kv[0].k.resize(1 << 16, 5);
        cache.put(std::move(source));
        const size_t alone = cache.bytes();
        auto fork = cache.fork(0);
        const size_t metadata = fork.private_bytes();
        check(cache.put(std::move(fork)) && cache.bytes() == alone + metadata, "the fork adds its metadata only");
        check(cache.evict_oldest() && cache.bytes() >= (1 << 16),
              "the survivor still pays for the shared prefix");
        check(cache.evict_oldest() && cache.bytes() == 0, "and frees it when it goes too");
    }
    {   // a long parked conversation shares only the tokens a shorter request has in common
        ConversationCache cache(1 << 20, 4);
        SavedConversation long_one;
        long_one.live.ids.assign(4096, 5);
        long_one.live.gdn.assign(8, 1);
        ConversationCheckpoint anchor;
        anchor.ids.assign(1024, 5);
        anchor.gdn.assign(8, 2);
        long_one.checkpoints.push_back(std::move(anchor));
        check(cache.put(std::move(long_one)), "a long conversation parks");
        std::vector<int32_t> prompt(2048, 5);
        prompt.back() = 9;                       // the request diverges at 2047
        const auto match = cache.best(prompt, {}, true);
        check(match.tokens == 1024 && !match.live, "the restore resumes from the checkpoint at 1024 tokens");
        check(match.prefix_tokens == 2047,
              "while the K/V it can share reaches the divergence, past the state it can restore");
        const auto extension = cache.best(std::vector<int32_t>(4097, 5), {}, true);
        check(extension.tokens == 4096 && extension.live && extension.prefix_tokens == 4096,
              "an extension of the parked conversation shares its whole backing");
    }
    {   // a bounded retention: pinned, earliest, newest, then log-spaced history
        SavedConversation archive;
        archive.live.ids.assign(200, 5);
        archive.live.gdn.assign(8, 1);
        for (int i = 1; i <= 40; ++i) {
            ConversationCheckpoint c;
            c.ids.assign((size_t) i * 2, 5);
            c.gdn.assign(8, 2);
            archive.checkpoints.push_back(std::move(c));
        }
        archive.checkpoints[3].pinned = true;   // 8 tokens: never a victim
        const size_t dropped = archive.compact_checkpoints(4);
        std::vector<size_t> kept;
        for (const auto& c : archive.checkpoints) kept.push_back(c.ids.size());
        check(dropped == 34 && archive.checkpoints.size() == 6, "a bounded retention drops the rest whole");
        check(kept == std::vector<size_t>({2, 8, 74, 76, 78, 80}),
              "pinned, earliest, newest and log-spaced history survive - never a stripped position");
        check(std::all_of(archive.checkpoints.begin(), archive.checkpoints.end(),
                          [](const ConversationCheckpoint& c) { return c.has_state(); }),
              "every survivor can still be restored");
        check(archive.restorable_checkpoint(80) != nullptr && archive.restorable_checkpoint(80)->ids.size() == 80,
              "the longest restorable checkpoint at that extent is the one offered");
        SavedConversation positions = archive;
        const size_t stripped = positions.drop_checkpoint_states(1);
        size_t stateless = 0;
        for (const auto& c : positions.checkpoints)
            if (!c.has_state()) ++stateless;
        check(stripped == stateless && stateless > 0 && positions.checkpoints.size() == archive.checkpoints.size(),
              "the disk tier's state bound keeps the matching positions");
        const ConversationCheckpoint* at74 = positions.restorable_checkpoint(74);
        check(at74 != nullptr && at74->has_state() && at74->ids.size() == 8,
              "a stripped position is never offered for resume");
        SavedConversation long_archive;
        long_archive.live.ids.assign(300, 5);
        for (int i = 1; i <= 300; ++i) {
            ConversationCheckpoint c;
            c.ids.assign((size_t) i, 5);
            c.gdn.assign(8, 2);
            long_archive.checkpoints.push_back(std::move(c));
        }
        long_archive.compact_checkpoints(6);
        check(long_archive.checkpoints.size() <= 8, "three hundred turn boundaries stay within the bound");
    }
    {
        SavedConversation archive;
        archive.live.ids.assign(31, 5);
        archive.live.gdn.assign(8, 31);
        archive.stage_images.resize(1);
        archive.stage_images[0].live = archive.live;
        for (size_t i = 1; i <= 30; ++i) {
            ConversationCheckpoint primary, secondary;
            primary.ids.assign(i, 5);
            primary.gdn.assign(8, uint8_t(i));
            primary.pinned = i == 7;
            secondary.ids = primary.ids;
            secondary.gdn.assign(8, uint8_t(i + 100));
            archive.checkpoints.push_back(std::move(primary));
            archive.stage_images[0].checkpoints.push_back(std::move(secondary));
        }
        archive.checkpoints[3].drop_state();
        archive.stage_images[0].checkpoints[3].drop_state();
        archive.compact_checkpoints(3);
        const auto& other = archive.stage_images[0];
        check(archive.checkpoints.size() <= 5 && other.checkpoints.size() == archive.checkpoints.size(),
              "compacting a layer-split archive bounds every stage with one shared selection");
        bool pin = false, aligned = true;
        for (size_t i = 0; i < archive.checkpoints.size(); ++i) {
            const auto& a = archive.checkpoints[i];
            const auto& b = other.checkpoints[i];
            aligned = aligned && a.ids == b.ids && a.has_state() && b.has_state() &&
                      b.gdn[0] == uint8_t(a.ids.size() + 100);
            pin = pin || (a.pinned && a.ids.size() == 7);
        }
        check(aligned && pin, "a primary-stage pin preserves its matching state on every stage");
    }
}

ModelGeometry prefix_geometry() {
    ModelGeometry g;
    g.n_embd = 4096; g.n_layers = 48; g.qsa_interval = 4;
    g.ssm_state_size = 128; g.ssm_k_heads = 16; g.ssm_v_heads = 32; g.ssm_d_conv = 4;
    g.ssm_conv_channels = 8192; g.ssm_value_dim = 128; g.n_head = 32; g.n_head_kv = 2;
    g.head_dim = 128; g.idx_q_heads = 4; g.idx_key_dim = 128; g.hc = 4; g.hc_lr = 4;
    g.n_expert = 256; g.n_ff = 14336;
    return g;
}

SavedConversation parked_prefix_image(const ModelGeometry& g, int64_t tokens) {
    SavedConversation s;
    s.geometry = conversation_geometry_key(g);
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    const int64_t block = strata::kernels::qsa_real_shapes().idx_block;
    s.live.ids.assign((size_t) tokens, 5);
    s.kv.resize(1);
    ConversationKv& layer = s.kv[0];
    layer.cells = ((tokens + page - 1) / page) * page;
    layer.heads = g.n_head_kv;
    layer.head_dim = g.head_dim;
    layer.page_size = page;
    layer.idx_dim = g.idx_key_dim;
    layer.pooled_rows = tokens / block + 1;
    layer.k.resize((size_t) layer.cells * (size_t) g.n_head_kv * (size_t) g.head_dim * 2, 3);
    layer.pooled.resize((size_t) layer.pooled_rows * (size_t) g.idx_key_dim * sizeof(float), 4);
    return s;
}

// The parked 300k image keeps its long tail; a fork trimmed to an anchor restores
// exactly that anchor's running state and copies no K/V byte to get there.
void test_prefix_trim() {
    const ModelGeometry g = prefix_geometry();
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    const int64_t block = strata::kernels::qsa_real_shapes().idx_block;
    const int64_t anchor = 4 * page, later = 6 * page, live = 8 * page;
    SavedConversation image = parked_prefix_image(g, live);
    ConversationCheckpoint at_anchor;
    at_anchor.ids.assign((size_t) anchor, 5);
    at_anchor.gdn.assign(64, 11);
    at_anchor.tails.assign(32, 12);
    image.checkpoints.push_back(at_anchor);
    ConversationCheckpoint past;
    past.ids.assign((size_t) later, 5);
    past.gdn.assign(64, 13);
    image.checkpoints.push_back(std::move(past));
    const SavedConversation parked = image;    // shares everything, like a cache entry
    const uint8_t* bytes = nullptr;
    { const ConversationBuffer& first = image.kv[0].k;
      first.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { bytes = p; return true; }); }
    const size_t full = image.kv[0].k.size();
    std::string error;
    check(conversation_snapshot_prefix(image, anchor, g, false, error), "trim to the anchor checkpoint");
    check(error.empty(), "a valid trim reports no error");
    check(image.live.ids.size() == (size_t) anchor && image.live.gdn == at_anchor.gdn &&
          image.live.tails == at_anchor.tails,
          "the anchor's own running state becomes the live state, so a restore is exact there");
    check(image.checkpoints.empty(), "every checkpoint past the anchor is dropped");
    const uint8_t* after = nullptr;
    { const ConversationBuffer& first = image.kv[0].k;
      first.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { after = p; return true; }); }
    check(after == bytes && image.kv[0].k.size() == (size_t) anchor * 2 * 128 * 2,
          "the K/V keeps its allocation and only shrinks its extent");
    check(image.kv[0].k.size() < full &&
          image.kv[0].pooled.size() == (size_t) (anchor / block + 1) * 128 * sizeof(float),
          "the indexer extent follows the anchor too");
    check(image.kv[0].cells == anchor && image.kv[0].pooled_rows == anchor / block + 1,
          "the recorded K/V extent follows the anchor");
    check(parked.live.ids.size() == (size_t) live && parked.checkpoints.size() == 2 &&
          parked.kv[0].k.size() == full, "the parked image is untouched by the trim");
    image.kv[0].k.resize(image.kv[0].k.size() + 4, 9);
    std::vector<uint8_t> probe(4, 0);
    check(image.kv[0].k.read(probe.data(), full / 2 - 4, 4) && probe[0] == 3,
          "a write past the anchor leaves the original K/V bytes alone");
    check(parked.kv[0].k.read(probe.data(), full / 2 - 4, 4) && probe[0] == 3,
          "and the parked image still reads its own prefix");
    SavedConversation missing = parked;
    check(!conversation_snapshot_prefix(missing, 5 * page, g, false, error) &&
          missing.live.ids.size() == (size_t) live && missing.checkpoints.size() == 2,
          "no checkpoint at that extent is refused, and the image is left as it was");
    SavedConversation stateless = parked;
    stateless.checkpoints[0].drop_state();
    check(!conversation_snapshot_prefix(stateless, anchor, g, false, error),
          "a matching position without a running state is refused");
    SavedConversation split = parked;
    split.stage_images.push_back(SavedConversation{});
    check(!conversation_snapshot_prefix(split, anchor, g, false, error),
          "a layer split's image is trimmed stage by stage");
    SavedConversation delta = parked;
    delta.kv[0].first_units[0] = anchor;
    check(!conversation_snapshot_prefix(delta, anchor, g, false, error), "a disk delta image is refused");
    SavedConversation malformed = parked;
    malformed.kv[0].k.pop_back();
    check(!conversation_snapshot_prefix(malformed, anchor, g, false, error) &&
              malformed.live.ids.size() == (size_t) live && malformed.checkpoints.size() == 2,
          "an incomplete K/V cell is rejected before checkpoint state or extents change");
    SavedConversation parts = parked;
    parts.live.stage_parts.push_back(ConversationCheckpoint{});
    check(!conversation_snapshot_prefix(parts, anchor, g, false, error),
          "a live checkpoint with stage parts is refused");
    SavedConversation whole = parked;
    check(conversation_snapshot_prefix(whole, live, g, false, error) && whole.checkpoints.size() == 2,
          "an extent equal to the live state is a no-op trim");
    ModelGeometry other = g;
    other.head_dim = 64;
    SavedConversation foreign = parked;
    check(!conversation_snapshot_prefix(foreign, anchor, other, false, error),
          "another model geometry is refused");
    {   // a draft layer rides along: no indexer rows, and its own ring extent follows
        SavedConversation draft_image = parked_prefix_image(g, live);
        draft_image.kv.push_back(ConversationKv{});
        ConversationKv& draft = draft_image.kv.back();
        draft.cells = live;
        draft.heads = g.n_head_kv;
        draft.head_dim = g.head_dim;
        draft.page_size = page;
        draft.idx_dim = g.idx_key_dim;
        draft.pooled_rows = 0;
        draft.k.resize((size_t) live * 2 * 128 * 2, 6);
        ConversationCheckpoint at_anchor;
        at_anchor.ids.assign((size_t) anchor, 5);
        at_anchor.gdn.assign(8, 1);
        draft_image.checkpoints.push_back(std::move(at_anchor));
        check(conversation_snapshot_prefix(draft_image, anchor, g, true, error),
              "an image whose last layer is the draft trims too");
        check(draft_image.kv.size() == 2 && draft_image.kv.back().pooled.empty() &&
              draft_image.kv.back().pooled_rows == 0 &&
              draft_image.kv.back().k.size() == (size_t) anchor * 2 * 128 * 2,
              "the draft layer keeps its ring extent and never grows indexer rows");
        check(draft_image.kv[0].pooled_rows == anchor / block + 1,
              "the indexer layer keeps its own rows");
    }
}

// A retiring checkpoint's capture borrows the unchanged backing a sibling still
// holds.  Bounded sparse checkpoints, a replacement terminal tail and slot churn
// must never make it recapture a K/V the cache already has: the archive's own
// selection refuses a divergence inside its input (a genuine branch, not a reply
// rewrite), and the cache then offers the parked sibling holding that prefix,
// whose buffers the new image shares and the ledger charges exactly once.
void test_retiring_capture_borrows_backing() {
    const size_t section = 4u << 20;   // one K/V section: large enough that a copy would show
    auto conversation = [&](std::initializer_list<int32_t> ids) {
        SavedConversation s = image(ids);
        s.kv.resize(1);
        s.kv[0].k.resize(section, 7);
        return s;
    };
    ConversationCache cache(48u << 20, 4);
    auto snapshot = conversation({1, 2, 3, 4, 5, 6});   // the long parked conversation
    snapshot.terminal_from = 6;
    check(cache.put(std::move(snapshot)), "the long conversation parks");
    auto archive = conversation({1, 2, 3, 4, 9, 9});    // a replaced terminal reply
    archive.checkpoint_archive = true;
    archive.terminal_from = 6;                          // its input ends where its live does
    archive.unchanged_tokens = 6;
    ConversationCheckpoint early;                       // one bounded sparse input anchor
    early.ids = {1, 2};
    early.gdn.resize(8, 1);
    archive.checkpoints.push_back(std::move(early));
    check(cache.put(std::move(archive)), "the archive with a bounded sparse checkpoint parks");
    ConversationCheckpoint retiring;
    retiring.ids = {1, 2, 3, 4, 5, 6, 7};               // continues the long conversation
    check(cache.archive_for(retiring, true).tokens == 0,
          "a divergence before the archive's input boundary is a genuine branch, not a replacement");
    const auto lenders = cache.shared_prefixes(retiring.ids, retiring.imgs, true);
    check(!lenders.empty() && lenders.front().prefix_tokens == 6 && lenders.front().index == 0,
          "the cache offers the sibling that still holds the whole shared prefix");
    const size_t before = cache.bytes();
    auto borrower = cache.fork(lenders.front().index);
    borrower.live.ids = retiring.ids;
    borrower.checkpoints.clear();
    borrower.terminal_from = 6;
    check(cache.put(std::move(borrower)), "the retiring capture parks beside the backing it borrowed");
    check(cache.evictions() == 0, "the capture does not evict the backing it borrows");
    check(cache.bytes() - before <= section / 8,
          "the borrowed backing is charged once, not copied for the capture");
    const size_t both = cache.bytes();
    check(cache.evict_oldest(), "the borrowed source can leave on slot pressure");
    check(cache.bytes() + section / 2 >= both,
          "dropping the source leaves the shared prefix charged to the capture that borrowed it");
}
}

int main() {
    {
        ConversationCheckpoint checkpoint;
        checkpoint.ids.resize(800, 42);
        auto prompt = checkpoint.ids;
        for (size_t at : {0u, 255u, 256u, 257u, 767u, 798u}) {
            prompt[at] = 43;
            check(conversation_common_prefix(checkpoint, prompt, {}) == (int64_t) at,
                  "long prefix ends exactly at the changed token");
            prompt[at] = 42;
        }
        check(conversation_common_prefix(checkpoint, prompt, {}) == 799,
              "long exact match still excludes the last prompt token");
    }
    test_root_priority();
    test_shared_prefix();
    test_prefix_trim();
    test_retiring_capture_borrows_backing();
    {
        ConversationCache cache(1 << 20, 4);
        auto old = image_with_checkpoint({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4});
        old.checkpoint_archive = true;
        old.unchanged_tokens = 6;
        old.terminal_from = 4;
        old.checkpoints.push_back(old.live); // a terminal state can be archived twice
        check(cache.put(std::move(old)), "park generated reply and its earlier branch point");
        ConversationCheckpoint next;
        next.ids = {1, 2, 3, 4, 7, 8};
        auto match = cache.archive_for(next, true);
        check(match.tokens == 6 && match.prefix_tokens == 4,
              "a rewritten terminal reply reuses the existing conversation prefix");
        cache.limit_reuse(0);
        match = cache.archive_for(next, true);
        check(match.tokens == 6 && match.prefix_tokens == 4,
              "full prefill does not hide the terminal-rewrite candidate");
        next.ids = {1, 2, 9};
        check(cache.archive_for(next, true).tokens == 0,
              "an edit before historical checkpoints preserves the independent branch");
        next.ids = {1, 2, 3, 4, 7, 8};
        next.imgs = {{1, 42}};
        check(cache.archive_for(next, true).tokens == 0,
              "rewritten replies cannot reuse an incompatible image prefix");
        auto saved = cache.take(match.index);
        saved.checkpoints.back().pinned = true;
        check(cache.put(std::move(saved)), "park a terminal prefix with an explicit pin");
        next.imgs.clear();
        check(cache.archive_for(next, true).tokens == 0,
              "an explicit terminal pin prevents suffix replacement");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto old = image_with_checkpoint({1, 2, 3}, {1, 2});
        old.checkpoint_archive = true;
        check(cache.put(std::move(old)), "park older archive");
        auto newer = image_with_checkpoint({1, 2, 3, 4}, {1, 2});
        newer.checkpoint_archive = true;
        auto prior = image({1, 2, 3}).live;
        newer.checkpoints.push_back(prior);
        check(cache.put(std::move(newer)), "park archive containing every older branch point");
        check(cache.size() == 1 && cache.superseded() == 1 && cache.evictions() == 0,
              "a fully subsumed archive is removed without a disk spill");
        check(cache.best(std::vector<int32_t>{1, 2, 9}, {}, true).tokens == 2 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 9}, {}, true).tokens == 3,
              "archive supersession retains both historical restore points");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto old = image_with_checkpoint({1, 2, 3}, {1, 2});
        old.checkpoint_archive = true;
        check(cache.put(std::move(old)), "a restorable historical archive parks");
        auto newer = image_with_checkpoint({1, 2, 3, 4}, {1, 2});
        newer.checkpoint_archive = true;
        ConversationCheckpoint position;
        position.ids = {1, 2, 3};
        newer.checkpoints.push_back(std::move(position));
        check(cache.put(std::move(newer)) && cache.size() == 2 && cache.superseded() == 0 &&
                  cache.best(std::vector<int32_t>{1, 2, 3, 9}, {}, true).tokens == 3,
              "a stateless matching position cannot supersede the original historical running state");
    }
    for (const bool archived : {false, true}) {
        ConversationCache cache(1 << 20, 4);
        auto parent = image_with_checkpoint({1, 2, 3, 4, 5, 6}, {1, 2});
        parent.checkpoint_archive = archived;
        parent.terminal_from = 5;
        parent.kv.resize(1);
        parent.kv[0].k.resize(16384, 9);
        check(cache.put(std::move(parent)), "the original long continuation parks");
        cache.seal_branches(std::vector<int32_t>{1, 2, 99}, {}, true);
        auto archive = cache.fork(0);
        archive.checkpoint_archive = true;
        archive.archive_sealed = false;
        archive.checkpoints.push_back(archive.live);
        archive.live.ids.push_back(7);
        auto archive_ids = archive.live.ids;
        archive_ids.push_back(99); // the request's final token is not a resume point
        check(cache.put(std::move(archive)), "a shared archive temporarily covers the original state");
        auto rotated = cache.take(cache.best(archive_ids, {}, true).index);
        rotated.checkpoints.push_back(rotated.live);
        rotated.live.ids.push_back(8);
        rotated.compact_checkpoints(1);
        check(cache.put(std::move(rotated)), "the bounded archive rotates away from the original state");
        const auto original = cache.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 99}, {}, true);
        check(original.tokens == 6 && original.live &&
                  cache.fork(original.index).live.gdn.vec() == std::vector<uint8_t>(64, 7),
              "archive compaction cannot remove the original long continuation's running state");
    }
    {
        // Active-window rotation must not make older branch points disappear.
        auto archive = image_with_checkpoint({1, 2, 3, 4, 5}, {1, 2});
        archive.checkpoint_archive = true;
        archive.unchanged_tokens = 5;
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(archive)), "checkpoint archive parks");
        auto newer = image_with_checkpoint({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4});
        check(cache.put(std::move(newer)), "recent active window parks");
        check(cache.best(std::vector<int32_t>{1, 2, 9, 9}, {}, true).tokens == 2,
              "historical edit restores an archived checkpoint absent from the active window");
        ConversationCheckpoint next;
        next.ids = {1, 2, 3, 4, 5, 6};
        auto match = cache.archive_for(next, true);
        check(match.tokens == 5, "continued branch extends its existing backing");
        next.ids = {1, 2, 3};
        check(cache.archive_for(next, true).tokens == 5, "older checkpoint uses the covering archive");
        next.imgs = {{1, 42}};
        check(cache.archive_for(next, true).tokens == 0, "image changes cannot share archive backing");
        next.imgs.clear();
        check(cache.archive_for(next, false).tokens == 0, "steering changes cannot share archive backing");
        cache.limit_reuse(3);
        auto saved = cache.take(match.index);
        check(saved.unchanged_tokens == 3, "rewrites invalidate archived K/V reuse past the dirty boundary");
        check(saved.live.ids == std::vector<int32_t>({1, 2, 3, 4, 5}) &&
              saved.checkpoints[0].ids == std::vector<int32_t>({1, 2}),
              "invalidating reuse does not destroy the preserved branch");
        saved.archive_sealed = true;
        check(cache.put(std::move(saved)), "a full metadata group remains parked");
        check(cache.archive_for(next, true).tokens == 0 &&
              cache.best(std::vector<int32_t>{1, 2, 9}, {}, true).tokens == 2,
              "sealed groups stop growing without losing historical reuse");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto stale = image_with_checkpoint({1, 2, 3, 4, 5}, {1, 2});
        stale.checkpoint_archive = true;
        stale.unchanged_tokens = 2;
        check(cache.put(std::move(stale)), "rewritten branch remains reusable as an immutable image");
        auto fresh = image_with_checkpoint({1, 2, 3, 4}, {1, 2});
        fresh.checkpoint_archive = true;
        fresh.unchanged_tokens = 4;
        check(cache.put(std::move(fresh)), "current branch backing parks");
        ConversationCheckpoint next;
        next.ids = {1, 2, 3, 4, 5, 6};
        check(cache.archive_for(next, true).tokens == 4,
              "extension prefers proven unchanged backing over a longer rewritten branch");
    }
    {
        auto older = image_with_checkpoint({1, 2, 3, 4, 5}, {1, 2});
        auto deepest = image({1, 2, 3, 4}).live;
        older.checkpoints.push_back(deepest);
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(older)), "conversation with historical checkpoints parks");
        auto recent = image_with_checkpoint({1, 2, 3, 4, 6}, {1, 2, 3, 4});
        check(cache.put(std::move(recent)), "conversation with a rotated checkpoint window parks");
        check(cache.best(std::vector<int32_t>{1, 2, 8}, {}, true).tokens == 2,
              "supersession cannot delete the only historical checkpoint");
    }
    for (const size_t extent : {size_t(700), size_t(1000), ConversationBuffer::segment_bytes + 1000}) {
        ConversationBuffer source;
        source.resize(1000, 7);
        auto branch = source;
        const size_t peak = branch.allocation_peak(extent, 350);
        branch.resize(extent);
        branch.visit(350, extent - 350, [](uint8_t* p, size_t n, size_t) {
            std::memset(p, 9, n);
            return true;
        });
        check(branch.bytes() <= peak, "a resized shared suffix and its split directory fit the admitted peak");
        uint8_t prefix = 0, suffix = 0, original = 0;
        check(branch.read(&prefix, 349, 1) && prefix == 7 &&
                  branch.read(&suffix, extent - 1, 1) && suffix == 9 &&
                  source.read(&original, 999, 1) && original == 7,
              "the admitted suffix write preserves the original prefix through shrink, rewrite, and growth");
    }
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // a parked conversation's K/V buffer (16 MiB, as captured) grows by one turn: the reserve for later turns is
        // an eighth of the buffer at most, not another 16 MiB segment
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 3);
        const size_t captured = bytes.bytes();
        check(captured - bytes.size() < 1024, "a fresh capture reserves nothing");
        const size_t turn = 1 << 20;
        const size_t peak = bytes.allocation_peak(bytes.size() + turn);
        bytes.resize(bytes.size() + turn, 4);
        check(bytes.bytes() <= peak, "a turn's growth stays within the predicted allocation peak");
        check(bytes.bytes() - captured <= turn + ConversationBuffer::segment_bytes / 8 + 1024,
              "the reserve for later turns is at most an eighth of the buffer");
        const size_t reserved = bytes.bytes();
        bytes.resize(bytes.size() + turn / 2, 5);
        check(bytes.bytes() == reserved, "the next small turn fills the reserve instead of allocating");
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments == 2, "the reserve extends the turn's segment");
    }
    {
        // many turns of 1 MiB onto a 16 MiB buffer: the reserve grows with the buffer, so the segments do not
        // multiply one per turn, and the reserve never exceeds an eighth of the buffer
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        for (int k = 0; k < 256; ++k) {
            const size_t before = bytes.size();
            const size_t peak = bytes.allocation_peak(before + (1 << 20));
            bytes.resize(before + (1 << 20), 2);
            check(bytes.bytes() <= peak, "each turn stays within the predicted allocation peak");
            check(bytes.bytes() - bytes.size() <= before / 8 + 4096, "the reserve stays within an eighth");
        }
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments <= 64, "256 turns of 1 MiB take far fewer than 256 segments");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true).tokens == 0 && cache.best(b, {}, true).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        auto cp = image({1, 2}).live;
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { auto c = image({}).live; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { auto c = image({}).live; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        // --conversation-cache-park-gain N: parked_prefix_tokens reports what a park now would supersede, so a
        // conversation that advanced fewer than N tokens since that copy keeps it instead of writing the whole
        // snapshot again for the next turn to drop.  It must name exactly the entries drop_superseded() would remove.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        check(cache.parked_prefix_tokens({1, 2, 3}, {}, {}, true) == 0, "an empty cache supersedes nothing");

        const std::vector<int32_t> live_ids = {1, 2, 3, 4, 60, 70};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        SavedConversation same = image({1, 2, 3, 4, 60, 70});
        same.checkpoints = chain;
        check(cache.put(std::move(same)), "park a conversation with a checkpoint chain");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true) == 5,
              "the entry the outgoing chain supersedes is reported by its deepest checkpoint");

        const std::vector<ConversationCheckpoint> other_chain = {cp({1, 2, 3})};
        check(cache.parked_prefix_tokens({1, 2, 3, 40, 50}, {}, other_chain, true) == 0,
              "a conversation sharing only the root supersedes nothing");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, false) == 0,
              "the other steering mode is a different entry");
        check(cache.parked_prefix_tokens({1, 2, 3, 4, 60, 70, 71}, {}, chain, true) == 5,
              "an appended tail keeps the reported depth at the parked one");
        SavedConversation no_cp = image({1, 2, 3, 9});
        check(cache.put(std::move(no_cp)) && cache.parked_prefix_tokens({1, 2, 3, 9}, {}, {}, true) == 0,
              "an entry without checkpoints is never reported");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
