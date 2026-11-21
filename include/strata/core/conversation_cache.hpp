// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <cstdio>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <deque>
#include <iterator>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    // The captured running state.  A copy shares these payloads and the first
    // write makes it private again, so an archived checkpoint costs nothing to
    // fork.  All five empty means this checkpoint is only a matching position
    // (the disk tier's state bound) and can never be restored.
    ConversationPayload gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // --prefill-pipe on a layer split: the plan a prompt segment [from, to) is read in - its chunk (0: none), the loan
    // its buffers were laid out for (which experts the prompt reads in place and which it streams: that sets bits
    // too), the pipe value it was planned with, whether it is one chunk and whether the plan changed the buffers'
    // chunks.  A periodic checkpoint taken inside such a segment keeps it, so a read that goes on from there reads the
    // rest in the same chunks with the same loan.  Run-time only, like `pinned` (a layer split saves no session files).
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
    } plan;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    /// The running state is present: the only checkpoints a restore may select.
    bool has_state() const {
        return !gdn.empty() || !ple.empty() || !tails.empty() || !dead.empty() || !block_pos.empty();
    }
    /// Drop the running state, keeping ids/imgs/used as a matching position.
    void drop_state() {
        gdn.clear(); ple.clear(); tails.clear(); dead.clear(); block_pos.clear();
        stage_parts.clear();
    }
    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
    /// The physical allocations alone, without ids/imgs/directories: what the ledger charges.
    size_t allocation_bytes() const {
        size_t n = gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity();
        for (const auto& part : stage_parts) n += part.allocation_bytes();
        return n;
    }
    template<class Fn> void visit_allocations(Fn&& fn) const {
        gdn.visit_allocations(fn); ple.visit_allocations(fn); tails.visit_allocations(fn);
        dead.visit_allocations(fn); block_pos.visit_allocations(fn);
        for (const auto& part : stage_parts) part.visit_allocations(fn);
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    // The layout-unit offset where each of the 5 sections' bytes begin in the FINAL image (the
    // L3 disk tier's tail capture: page-capped cells for k/v/scales, pooled rows for the indexer,
    // one page earlier for the draft layer).  0 in a full-capture image.
    std::array<int64_t, 5> first_units{};
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
    size_t allocation_bytes() const {
        return k.allocation_bytes() + v.allocation_bytes() + k_scale.allocation_bytes() +
               v_scale.allocation_bytes() + pooled.allocation_bytes();
    }
    template<class Fn> void visit_allocations(Fn&& fn) const {
        k.visit_allocations(fn); v.visit_allocations(fn); k_scale.visit_allocations(fn);
        v_scale.visit_allocations(fn); pooled.visit_allocations(fn);
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
    size_t allocation_bytes() const {
        size_t n = 0;
        for (const auto& layer : kv) n += layer.allocation_bytes();
        for (const auto& s : stages) n += s.allocation_bytes();
        return n;
    }
    size_t private_bytes() const { return bytes() - allocation_bytes(); }
    template<class Fn> void visit_allocations(Fn&& fn) const {
        for (const auto& layer : kv) layer.visit_allocations(fn);
        for (const auto& s : stages) s.visit_allocations(fn);
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

/// The one retention policy the archive bound and the active restore window
/// share: over `eligible` (ascending) it keeps every pinned checkpoint, the
/// earliest (the shared prefix root every suffix branches from), the newest, and
/// then the rest newest-first with geometrically growing index gaps, so history
/// stays log-spaced instead of only the last few.  `limit` bounds the ordinary
/// states kept and the live checkpoint counts as one of them.
inline std::vector<size_t> conversation_checkpoint_selection(const std::vector<ConversationCheckpoint>& checks,
                                                             const std::vector<size_t>& eligible, size_t limit) {
    std::vector<size_t> keep;
    if (eligible.empty()) return keep;
    if (limit == 0) limit = 1;
    std::vector<size_t> candidates;
    for (const size_t i : eligible) {
        if (checks[i].pinned) keep.push_back(i);
        else if (i != eligible.front() && i != eligible.back()) candidates.push_back(i);
    }
    keep.push_back(eligible.front());
    keep.push_back(eligible.back());
    size_t ordinary = 1;   // the live checkpoint
    for (size_t back = 0; ordinary < limit && back < candidates.size(); back = back == 0 ? 1 : back * 2) {
        keep.push_back(candidates[candidates.size() - 1 - back]);
        ++ordinary;
    }
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return keep;
}

/// Which checkpoints a bounded retention keeps, as a flag per position.  A
/// checkpoint that is not kept goes whole - never as a stateless position a
/// restore could pick.
inline std::vector<bool> conversation_checkpoint_keep(const std::vector<ConversationCheckpoint>& checks, size_t limit) {
    std::vector<size_t> all(checks.size());
    for (size_t i = 0; i < checks.size(); ++i) all[i] = i;
    std::vector<bool> keep(checks.size(), false);
    for (const size_t i : conversation_checkpoint_selection(checks, all, limit)) keep[i] = true;
    return keep;
}

/// The same policy for a restore window: the positions up to `upto` tokens that
/// still carry a running state.  A stripped matching position is never eligible.
inline std::vector<size_t> conversation_checkpoint_indices(const std::vector<ConversationCheckpoint>& checks,
                                                           size_t limit, int64_t upto) {
    std::vector<size_t> eligible;
    for (size_t i = 0; i < checks.size(); ++i)
        if (checks[i].has_state() && !checks[i].ids.empty() && (int64_t) checks[i].ids.size() <= upto)
            eligible.push_back(i);
    return conversation_checkpoint_selection(checks, eligible, limit);
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    // An active checkpoint archive shares one authoritative K/V image across
    // its checkpoint chain. Runtime-only: disk records already retain the chain.
    bool checkpoint_archive = false;
    bool archive_sealed = false; // a forked source or full metadata group remains reusable, but no longer grows
    int64_t unchanged_tokens = 0;
    // The input-prompt token count this image's live state was captured at: the
    // boundary the client's input ended on, so every position at or after it is
    // the model's generated reply.  0: the live state is still input (a capture
    // made while reading a prompt) or its provenance is unknown, and no part of
    // its tail is ever treated as a replaceable reply.  A request that shares at
    // least `terminal_from` tokens is rewriting that reply, so this image's tail
    // may be superseded; a request that diverges earlier is a genuine branch and
    // leaves the image whole.  A token distance can never stand in for it: with
    // a bounded sparse checkpoint chain the deepest anchor may sit before the
    // reply, and a long-to-short rewind would then look like a rewrite.
    int64_t terminal_from = 0;

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    /// The longest checkpoint that still carries a running state and whose ids
    /// are a prefix of `upto` tokens: what a restore may select.  A stripped
    /// matching position (has_state() false) is never offered.  nullptr: none.
    const ConversationCheckpoint* restorable_checkpoint(int64_t upto) const {
        const ConversationCheckpoint* best = nullptr;
        for (const auto& c : checkpoints) {
            if (!c.has_state() || c.ids.empty() || (int64_t) c.ids.size() > upto) continue;
            if (!best || c.ids.size() > best->ids.size()) best = &c;
        }
        return best;
    }

    /// Bounds the running states this archive keeps to `limit` ordinary states
    /// (see conversation_checkpoint_keep).  Returns how many were dropped.
    size_t compact_checkpoints(size_t limit) {
        std::vector<bool> keep(checkpoints.size(), false);
        for (size_t i : conversation_checkpoint_indices(checkpoints, limit, INT64_MAX)) keep[i] = true;
        const size_t before = checkpoints.size();
        // Stage checkpoint lists have the same order. Apply one selection so a
        // pin on the primary stage also retains every other stage's state.
        auto compact = [&](auto&& self, SavedConversation& image) -> void {
            for (auto& stage : image.stage_images) self(self, stage);
            size_t w = 0;
            for (size_t i = 0; i < image.checkpoints.size(); ++i) {
                if (i >= keep.size() || !keep[i]) continue;
                if (w != i) image.checkpoints[w] = std::move(image.checkpoints[i]);
                ++w;
            }
            image.checkpoints.resize(w);
        };
        compact(compact, *this);
        return before - checkpoints.size();
    }

    /// The same selection, but a victim keeps its ids/imgs/used as a matching
    /// position and only its running state goes.  A caller that offers positions
    /// for resume MUST skip the stripped ones (has_state(), restorable_checkpoint).
    size_t drop_checkpoint_states(size_t limit) {
        const std::vector<bool> keep = conversation_checkpoint_keep(checkpoints, limit);
        size_t dropped = 0;
        for (size_t i = 0; i < checkpoints.size(); ++i)
            if (!keep[i]) { checkpoints[i].drop_state(); ++dropped; }
        return dropped;
    }

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
    /// The physical allocations alone, without ids/imgs/directories: what the ledger charges.
    size_t allocation_bytes() const {
        size_t n = live.allocation_bytes();
        for (const auto& s : stage_images) n += s.allocation_bytes();
        for (const auto& c : checkpoints) n += c.allocation_bytes();
        for (const auto& k : kv) n += k.allocation_bytes();
        return n;
    }
    size_t private_bytes() const { return bytes() - allocation_bytes(); }
    template<class Fn> void visit_allocations(Fn&& fn) const {
        live.visit_allocations(fn);
        for (const auto& s : stage_images) s.visit_allocations(fn);
        for (const auto& c : checkpoints) c.visit_allocations(fn);
        for (const auto& k : kv) k.visit_allocations(fn);
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

// Skip equal stretches with libc, then locate the first difference in
// the one block that did not match.
inline size_t common_token_prefix(const int32_t* a, const int32_t* b, size_t limit) {
    if (limit == 0 || a[0] != b[0]) return 0;
    constexpr size_t block = 256;
    size_t n = 0;
    while (limit - n >= block) {
        if (std::memcmp(a + n, b + n, block * sizeof(*a)) != 0) {
            limit = n + block;
            break;
        }
        n += block;
    }
    while (n < limit && a[n] == b[n]) ++n;
    return n;
}

/// The K/V extent this checkpoint's backing shares with `prompt`: the longest
/// common token prefix below the last prompt token, with the image keys below it
/// equal.  Unlike conversation_prefix it does not require the checkpoint to end
/// there, so a 300k parked image shares its first 20k tokens with a 20k request.
/// Bytes only: the caller confirms them on the device before reusing.
template<class Token>
int64_t conversation_common_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                                   const std::vector<ConversationImageKey>& images) {
    const size_t limit = std::min(c.ids.size(), prompt.empty() ? size_t(0) : prompt.size() - 1);
    size_t n = 0;
    if constexpr (std::is_same_v<Token, int32_t>)
        n = common_token_prefix(c.ids.data(), prompt.data(), limit);
    else
        while (n < limit && c.ids[n] == prompt[n]) ++n;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;        // the running state a restore resumes from: the matched checkpoint's extent
        bool live = false;
        // The K/V this entry shares with the prompt: the common prefix of its own
        // backing (conversation_common_prefix), which may be longer than `tokens`
        // when the request diverges before the parked conversation ends.  Bytes
        // only - the caller confirms them on the device before reusing them.
        int64_t prefix_tokens = 0;
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    // Spill only capacity victims. Superseded entries and active restores do not leave for L3.
    using EvictionHandler = void (*)(SavedConversation&&, bool root, void*) noexcept;
    void on_eviction(EvictionHandler handler, void* context) {
        eviction_handler_ = handler;
        eviction_context_ = context;
    }
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    /// The cache's own retention: every physical allocation charged once however
    /// many entries share it, plus each entry's private metadata.  The ledger and
    /// the private total are maintained on the mutations above, so a query never
    /// scans a payload, and a view the caller keeps outside the cache (a fork)
    /// neither adds nor removes a charge.
    size_t bytes() const { return ledger_.bytes() + private_; }
    /// The physical bytes of `reuse` this cache already charges, because those
    /// buffers are shared with a parked entry (or with its own retained reuse):
    /// a scalar capture estimate minus this is the additional RAM the capture
    /// needs, so pre-admission never charges a retained shared prefix twice and
    /// never evicts the long source a short branch shares.
    size_t shared_bytes(const ConversationKvReuse& reuse) const { return charged_shared(reuse); }
    /// The same for a whole image: the covered archive path sums it over the
    /// images that already hold the K/V and checkpoints, without building a
    /// temporary reuse candidate just to account for the shared prefix.
    size_t shared_bytes(const SavedConversation& image) const { return charged_shared(image); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    /// Evictions of a pinned conversation (evict_oldest's last resort, older pins than the newest).
    size_t pinned_evictions() const { return pinned_evictions_; }
    bool pinned(size_t index) const { return entries_.at(index).pinned; }
    // A parked system-prompt prefix (the chain's root) is the same ids for every chat of a client.
    bool has(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec) const {
        for (const auto& e : entries_)
            if (e.image.cvec == cvec && e.image.live.ids == ids && e.image.live.imgs == imgs) return true;
        return false;
    }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.image.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    // Buffers shared with a parked entry are charged once, not twice.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        release_reuse();
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (!enabled()) return;
        adopt_reuse(candidate);
        if (bytes() > budget_) {
            uncharge_reuse(candidate);
            return;
        }
        reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) release_reuse();
        for (auto& e : entries_)
            e.image.unchanged_tokens = std::min(e.image.unchanged_tokens, first_dirty);
    }
    ConversationKvReuse take_reuse() {
        ConversationKvReuse out = std::exchange(reuse_, {});
        uncharge_reuse(out);   // the caller now holds these bytes outside the cache
        return out;
    }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    // Find an archive on this branch, or replace its rewritten terminal reply
    // when every earlier checkpoint remains on the incoming branch. Historical
    // edits before that chain keep an independent archive.
    Match archive_for(const ConversationCheckpoint& checkpoint, bool cvec) const {
        Match match;
        int best_rank = 0;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& image = entries_[i].image;
            if (!image.checkpoint_archive || image.cvec != cvec) continue;
            const auto& a = image.live;
            const auto& b = checkpoint;
            const auto& shorter = a.ids.size() <= b.ids.size() ? a : b;
            const auto& longer = a.ids.size() <= b.ids.size() ? b : a;
            const int64_t common = common_token_prefix(shorter.ids.data(), longer.ids.data(), shorter.ids.size());
            const bool terminal_rewrite = common < (int64_t) shorter.ids.size();
            if (terminal_rewrite) {
                // Only a generated reply may be replaced, and this image's own
                // input boundary says where its reply starts (0: unknown - never
                // a replacement).  An explicit pin, on the terminal state or on
                // any checkpoint past the shared prefix, refuses it too: that
                // state is a promised prefix and stays where it is.
                bool compatible = !a.pinned && image.terminal_from > 0 && common >= image.terminal_from;
                for (const auto& c : image.checkpoints)
                    if (c.pinned && (c.ids == a.ids || (int64_t) c.ids.size() > common)) compatible = false;
                if (!compatible) continue;
            }
            size_t j = 0;
            bool compatible = true;
            for (const auto& key : longer.imgs) {
                if (key.start >= common) break;
                if (j == shorter.imgs.size() || !(shorter.imgs[j++] == key)) { compatible = false; break; }
            }
            if (!compatible || (j < shorter.imgs.size() && shorter.imgs[j].start < common)) continue;
            const bool present = a.ids == b.ids || std::any_of(
                image.checkpoints.begin(), image.checkpoints.end(), [&](const auto& c) { return c.ids == b.ids; });
            if (image.archive_sealed && !present) continue;
            const int rank = present ? 4 : terminal_rewrite ? 1 : image.unchanged_tokens >= common ? 3 : 2;
            if (rank > best_rank || (rank == best_rank && common > match.prefix_tokens)) {
                best_rank = rank;
                match = {i, (int64_t) a.ids.size(), true, common};
            }
        }
        return match;
    }
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // An equal-length tie prefers the SMALLER image: a parked system-prompt prefix over a whole
        // conversation that contains it reads back faster with identical output.  At equal bytes the
        // most recently parked branch wins (the iteration direction); the caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.image.cvec != cvec) continue;
            const int64_t backing = conversation_common_prefix(e.image.live, prompt, images);
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                if (!live && !c.has_state()) return;
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n == 0) return;
                const bool longer = n > best.tokens;
                const bool equal_smaller = n == best.tokens && n != 0 &&
                                           e.image.bytes() < entries_[best.index].image.bytes();
                if (longer || equal_smaller) best = {i, n, live, backing};
            };
            consider(e.image.live, true);
            for (const auto& c : e.image.checkpoints) consider(c, false);
        }
        return best;
    }

    // K/V sharing does not require a recurrent-state anchor before the cut.
    template<class Token>
    Match shared_prefix(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                        bool cvec) const {
        Match match;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& image = entries_[i].image;
            if (image.cvec != cvec) continue;
            const int64_t common = conversation_common_prefix(image.live, prompt, images);
            if (common > match.prefix_tokens) match = {i, 0, false, common};
        }
        return match;
    }

    /// The same selection for every entry at once, longest common prefix first
    /// (at equal length the most recently parked branch): a caller that has to
    /// confirm a candidate against the device - parking a conversation, or a
    /// retiring checkpoint borrowing the backing it still shares - takes the
    /// first the device agrees with and tries the next when it refuses, instead
    /// of recapturing a K/V another entry already holds.
    template<class Token>
    std::vector<Match> shared_prefixes(const std::vector<Token>& prompt,
                                       const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::vector<Match> out;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& image = entries_[i].image;
            if (image.cvec != cvec) continue;
            const int64_t common = conversation_common_prefix(image.live, prompt, images);
            if (common > 0) out.push_back({i, 0, false, common});
        }
        std::sort(out.begin(), out.end(), [](const Match& left, const Match& right) {
            return left.prefix_tokens > right.prefix_tokens ||
                   (left.prefix_tokens == right.prefix_tokens && left.index > right.index);
        });
        return out;
    }
    /// Freeze the bounded source archive when a request branches inside its
    /// input. Its original resume point must survive compaction of the new
    /// branch; subsequent archives can still borrow the same K/V allocations.
    template<class Token>
    void seal_branches(const std::vector<Token>& prompt,
                       const std::vector<ConversationImageKey>& images, bool cvec) {
        for (auto& entry : entries_) {
            auto& image = entry.image;
            if (image.checkpoint_archive && !image.archive_sealed && image.cvec == cvec && image.terminal_from > 0 &&
                conversation_common_prefix(image.live, prompt, images) < image.terminal_from)
                image.archive_sealed = true;
        }
    }


    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index).image);
        uncharge(out);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        return out;
    }

    /// A snapshot that shares the parked entry's buffers, leaving the entry in
    /// place: the source keeps its own long tail, neither side copies the shared
    /// prefix until it writes into it, and the cache's charge does not move.  The
    /// caller may trim it (conversation_snapshot_prefix) and then modify it.
    SavedConversation fork(size_t index) const { return entries_.at(index).image; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) release_reuse();
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes() > budget_ - held - incoming)) {
            // Evict ordinary conversations first, then roots, then older checkpoint pins if necessary.
            if (!evict_oldest()) return false;
        }
        return true;
    }

    /// make_room for an image that is already built: reserve the unique bytes it
    /// adds (a prefix another entry already holds costs nothing) instead of its
    /// whole reachable footprint, so a shared branch never evicts its own source.
    bool make_room(const SavedConversation& incoming, size_t held = 0) {
        if (!enabled() || held > budget_) return false;
        Reservation reservation(*this, incoming);
        if (incoming.private_bytes() + ledger_.shared_bytes(incoming) > budget_ - held) return false;
        return make_room(size_t(0), held);
    }

    // Evict the oldest ordinary conversation, or the oldest root if no ordinary victim remains.
    // The parking path can also free RAM one entry at a time on demand (see the physical-RAM admission
    // gate in generate.cpp).  A ConversationBuffer holds pieces of 16 MiB, each its own allocation, far
    // above glibc's mmap threshold, so dropping an entry returns the whole footprint to the kernel at
    // once - the next admission check reads it back from /proc/meminfo.  A piece another entry still
    // shares stays charged to that entry.
    // If only conversations with pinned checkpoints remain, evict the oldest parked one, but keep the newest.
    // A pin belongs to a conversation, so abandoned pinned conversations can otherwise fill the cache indefinitely.
    // Park order is not a liveness signal: an older pin may still be in use by a client.
    // False when none can go (empty, or only the newest pinned entry is left).
    bool evict_oldest() {
        auto victim = std::find_if(entries_.begin(), entries_.end(),
                                   [](const Entry& e) { return !e.pinned && !e.image.pinned(); });
        if (victim == entries_.end())
            victim = std::find_if(entries_.begin(), entries_.end(),
                                  [](const Entry& e) { return e.pinned && !e.image.pinned(); });
        bool leaked_pin = false;
        if (victim == entries_.end()) {
            auto newest = std::find_if(entries_.rbegin(), entries_.rend(),
                                       [](const Entry& e) { return e.image.pinned(); });
            if (newest != entries_.rend()) {
                const auto keep = std::prev(newest.base());
                victim = std::find_if(entries_.begin(), keep, [](const Entry& e) { return e.image.pinned(); });
                if (victim == keep) victim = entries_.end();
                leaked_pin = victim != entries_.end();
            }
        }
        if (victim == entries_.end()) return false;
        if (leaked_pin) {
            ++pinned_evictions_;
            std::fprintf(stderr, "strata serve: conversation cache: evicting an older pinned conversation (%zu tokens) "
                                 "to make room; pinned_evictions=%zu\n",
                         victim->image.live.ids.size(), pinned_evictions_);
        }
        uncharge(victim->image);
        if (eviction_handler_)
            eviction_handler_(std::move(victim->image), victim->pinned, eviction_context_);
        entries_.erase(victim);
        ++evictions_;
        return true;
    }
    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        private_ = 0;
        ledger_ = ConversationLedger{};
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    // An archive also requires its live checkpoint in the replacement. Keep
    // every historical restore point; discard a fully covered group without
    // treating it as a capacity victim or writing it to L3.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                           bool incoming_archive = false) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.has_state() && k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            // A bounded checkpoint archive can drop these states on its next
            // compaction. It must not replace the full parked branch that keeps
            // the original continuation. Both images already share their K/V.
            if (incoming_archive && (!e.image.checkpoint_archive || e.image.archive_sealed)) { ++i; continue; }
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.image.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            const bool subsumed = e.image.checkpoint_archive
                ? held(e.image.live)
                : deepest && !deepest->ids.empty() && held(*deepest);
            if (e.image.cvec == cvec && subsumed &&
                std::all_of(e.image.checkpoints.begin(), e.image.checkpoints.end(), held)) {
                uncharge(e.image);
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none.  It is the entry drop_superseded() would remove, so the caller can tell whether a
    /// park now would only rewrite what a turn from now drops: the prefix such an entry holds is this conversation's
    /// prefix, and the conversation only appends, so the copy stays usable for the reuse it was made for.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) const {
        size_t longest = 0;
        for (const auto& e : entries_) {
            if (e.image.cvec != cvec) continue;
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.image.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (!deepest || deepest->ids.empty()) continue;
            bool holds = deepest->ids == ids && deepest->imgs == images;
            for (const auto& k : checkpoints)
                if (!holds && k.ids == deepest->ids && k.imgs == deepest->imgs) holds = true;
            if (holds) longest = std::max(longest, deepest->ids.size());
        }
        return longest;
    }

    bool put(SavedConversation&& image, size_t held = 0, bool pinned = false) {
        if (!enabled() || held > budget_) return false;
        Reservation reservation(*this, image);
        if (image.private_bytes() + ledger_.shared_bytes(image) > budget_ - held) return false;
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec, image.checkpoint_archive);
        if (!make_room(size_t(0), held)) return false;
        // Allocate the deque entry before moving the reservation's rollback source.
        entries_.emplace_back();
        entries_.back().image = std::move(image);
        entries_.back().pinned = pinned;
        reservation.committed = true;
        return true;
    }

private:
    // Keep the candidate's shared allocations charged during eviction. Removing
    // its old cache owner must not make that still-live prefix look like free RAM.
    struct Reservation {
        ConversationCache& owner;
        const SavedConversation& image;
        bool committed = false;
        Reservation(ConversationCache& owner, const SavedConversation& image) : owner(owner), image(image) {
            owner.charge(image);
        }
        ~Reservation() { if (!committed) owner.uncharge(image); }
    };
    size_t budget_ = 0, slots_ = 0, private_ = 0, evictions_ = 0, superseded_ = 0, pinned_evictions_ = 0;
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    struct Entry {
        SavedConversation image;
        bool pinned = false;   // a shared system-prompt prefix: evicted last (--conversation-cache-keep-root)
    };
    std::deque<Entry> entries_; // least recently active first
    ConversationKvReuse reuse_;
    ConversationLedger ledger_;
    EvictionHandler eviction_handler_ = nullptr;
    void* eviction_context_ = nullptr;

    /// The physical bytes of `image` this cache already charges, through its ledger.
    template<class Image> size_t charged_shared(const Image& image) const {
        return ledger_.shared_bytes(image);
    }
    template<class Image> void charge(const Image& image) {
        ledger_.adopt(image);
        private_ += image.private_bytes();
    }
    template<class Image> void uncharge(const Image& image) {
        ledger_.release(image);
        private_ -= image.private_bytes();
    }
    void adopt_reuse(const ConversationKvReuse& reuse) { charge(reuse); }
    void uncharge_reuse(const ConversationKvReuse& reuse) { uncharge(reuse); }
    void release_reuse() {
        if (reuse_.kv.empty() && reuse_.stages.empty()) { reuse_ = {}; return; }
        uncharge_reuse(reuse_);
        reuse_ = {};
    }
};

} // namespace strata::core
