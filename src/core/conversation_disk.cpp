// src/core/conversation_disk.cpp - the L3 disk tier for parked conversations (append-only chains).
//
// One conversation = one file, a fixed 128-byte master header followed by one PANEL per park.
// A park writes only the delta since the chain's last panel - the token ids/imgs it gained, new
// checkpoints, the current running state, and per-layer K/V tails - so a growing conversation
// never rewrites its own history.  A chain may BORROW a prefix from another chain instead of
// storing it: base kind 1 is a shared system-prompt ('p') chain that covers exactly the borrowed
// prefix, base kind 2 is a conversation ('c') chain that covers more and lends only its first
// base_tokens tokens.  A base may itself have a base; the graph is acyclic and bounded by
// ConversationDiskStore::max_base_depth.
//
// File layout, from offset zero:
//   chain header   a fixed 128-byte block: magic, version, coverage, panel count, identity, cvec,
//                  base reference (kind/tokens/name/prefix fingerprint)
//   panel 1        header + metadata + payload
//   panel 2        header + metadata + payload
//   ...            (appended at EOF; the chain header is patched in place)
//
// Panel header   a fixed 56-byte block: magic, base/end tokens, metadata size + checksum,
//                payload size + checksum.
// Panel metadata the delta's shape: every scalar, every id and image key, and every section length.
// Panel payload  the delta's variable-length arrays back to back, in the metadata's section order.
//
// The seed (first panel) writes to a temporary file and renames it over the target, exactly like
// the old full-snapshot records.  Appends write in place at EOF: a crash mid-append leaves a torn
// tail whose payload checksum fails on get(), and the store truncates back to the last complete
// panel - the conversation survives to its previous park.
//
// The store is CPU only.  It makes no CUDA call; the caller captures and restores delta images
// with conversation_snapshot.hpp.  Reading merges the base and every panel back into full stage
// images whose K/V buffers are pre-sized and streamed into place, so the engine's restore path
// never changes (peak host RAM is one merged conversation plus the small base record).  A prefix
// read (get_prefix) sizes those buffers to the requested cut, so a short branch of a long chain
// never allocates or holds the long chain's K/V.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>
#include "strata/kernels/qsa.hpp"

namespace strata::core {
namespace {

// "STRCDISK", little-endian.
constexpr uint64_t kMagic = 0x4B53494443525453ull;
// "PANELKDS": the panel block magic.
constexpr uint64_t kPanelMagic = 0x53444B4C454E4150ull;
constexpr uint32_t kChainHeaderBytes = 128;
constexpr uint32_t kPanelHeaderBytes = 56;
constexpr uint64_t kMaxMetadataBytes = ConversationDiskStore::max_metadata_bytes;
constexpr uint32_t kMaxStages = 1024;
constexpr uint32_t kMaxCheckpoints = 1u << 20;
constexpr uint32_t kMaxParts = 64;
constexpr uint32_t kMaxPartDepth = 4;
constexpr uint32_t kMaxKv = 1u << 16;
constexpr uint64_t kMaxPanels = 1u << 24;
// The base reference kinds: none, a shared system-prompt root, a longer conversation's prefix.
constexpr uint8_t kBaseNone = 0;
constexpr uint8_t kBaseRoot = 1;
constexpr uint8_t kBaseChain = 2;

// ---- K/V layout units.  The final extent of each section is measured in units: page-capped
// cells for the four K/V buffers, pooled indexer ROWS (idx_block-aligned, with the moving spare
// row) for the pooled buffer.  Each panel's kv metadata carries its own tail's START unit per
// section (`ConversationKv::first_units`), so the merge places every tail at that offset - later
// tails overwrite the previous content's partial page/row.
int64_t kv_end_units(int64_t end, int64_t page, int e) {
    if (e == 4) return end / strata::kernels::qsa_real_shapes().idx_block + 1;
    return (end + page - 1) / page * page;
}

// The units a tail covers: from its stored start unit to the panel's end extent.
int64_t kv_units(int64_t first, int64_t end, int64_t page, int e) {
    return kv_end_units(end, page, e) - first;
}
constexpr size_t kMaxNameBytes = 96;

struct ChainHeader {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t reserved = 0;
    uint64_t total_tokens = 0;              // the chain's coverage after its last panel
    uint64_t panel_count = 0;
    uint8_t identity[ConversationDiskIdentity::size] = {};
    uint8_t cvec = 0;
    uint8_t base_kind = 0;                  // kBaseNone, kBaseRoot, or kBaseChain
    uint8_t padding0[6] = {};
    uint64_t base_tokens = 0;               // how many tokens of the base this chain borrows
    char base_name[40] = {};
    // The borrowed prefix's content (its ids and its images below base_tokens).  A chain that
    // borrows nothing writes 0; a file from an older build also holds 0, so the check is skipped.
    uint64_t base_prefix_checksum = 0;
};
static_assert(sizeof(ChainHeader) == kChainHeaderBytes, "the chain header is a fixed 128 bytes");

struct PanelHeader {
    uint64_t magic = 0;
    uint64_t base_tokens = 0;               // the chain's coverage BEFORE this panel
    uint64_t end_tokens = 0;                // the chain's coverage AFTER this panel
    uint64_t metadata_bytes = 0;
    uint64_t metadata_checksum = 0;
    uint64_t payload_bytes = 0;
    uint64_t payload_checksum = 0;
};
static_assert(sizeof(PanelHeader) == kPanelHeaderBytes, "the panel header is a fixed 56 bytes");

bool add_overflow(uint64_t& total, uint64_t extra) {
    if (extra > UINT64_MAX - total) return false;
    total += extra;
    return true;
}

bool read_exact(std::FILE* file, void* destination, uint64_t bytes) {
    uint8_t* out = static_cast<uint8_t*>(destination);
    while (bytes != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, 1u << 20));
        if (std::fread(out, 1, chunk, file) != chunk) return false;
        out += chunk;
        bytes -= chunk;
    }
    return true;
}

// The base name a chain header holds.  The field is fixed width and may be unterminated in a
// corrupt file, so the read is bounded.
std::string base_name_of(const ChainHeader& header) {
    size_t n = 0;
    while (n < sizeof(header.base_name) && header.base_name[n] != '\0') ++n;
    return std::string(header.base_name, n);
}

// One variable-length payload section.  Every reader of it here SERIALIZES, so both pointers are
// const: a running state and a K/V buffer are shared between forked images (copy on write), and a
// writable access would unshare - and copy - the whole payload.
struct Section {
    const std::vector<uint8_t>* bytes = nullptr;
    const ConversationBuffer* buffer = nullptr;
};

template<class Fn>
void visit_checkpoint(const ConversationCheckpoint& checkpoint, Fn& fn) {
    fn(Section{&checkpoint.gdn.vec(), nullptr});
    fn(Section{&checkpoint.ple.vec(), nullptr});
    fn(Section{&checkpoint.tails.vec(), nullptr});
    fn(Section{&checkpoint.dead.vec(), nullptr});
    fn(Section{&checkpoint.block_pos.vec(), nullptr});
    for (const auto& part : checkpoint.stage_parts) visit_checkpoint(part, fn);
}

// Visits every payload section in file order. The metadata writer and reader use the same order.
template<class Fn>
void visit_payload(const ConversationDiskRecord& record, Fn& fn) {
    for (const auto& stage : record.stages) {
        visit_checkpoint(stage.live, fn);
        for (const auto& checkpoint : stage.checkpoints) visit_checkpoint(checkpoint, fn);
        for (const auto& kv : stage.kv) {
            fn(Section{nullptr, &kv.k});
            fn(Section{nullptr, &kv.v});
            fn(Section{nullptr, &kv.k_scale});
            fn(Section{nullptr, &kv.v_scale});
            fn(Section{nullptr, &kv.pooled});
        }
    }
}

uint64_t count_sections(const ConversationDiskRecord& record) {
    uint64_t count = 0;
    auto fn = [&](Section) { ++count; };
    visit_payload(record, fn);
    return count;
}

// The payload bytes of one checkpoint and its stage parts, in file order.
uint64_t checkpoint_payload_bytes(const ConversationCheckpoint& checkpoint) {
    uint64_t total = checkpoint.gdn.size() + checkpoint.ple.size() + checkpoint.tails.size() +
                     checkpoint.dead.size() + checkpoint.block_pos.size();
    for (const auto& part : checkpoint.stage_parts) total += checkpoint_payload_bytes(part);
    return total;
}

// The bytes a record occupies in a chain file: its panel header, its metadata, and its payload.
uint64_t record_bytes(const ConversationDiskRecord& record, size_t metadata_bytes) {
    uint64_t total = kChainHeaderBytes + kPanelHeaderBytes + metadata_bytes;
    for (const auto& stage : record.stages) {
        total += checkpoint_payload_bytes(stage.live);
        for (const auto& checkpoint : stage.checkpoints) total += checkpoint_payload_bytes(checkpoint);
        for (const auto& kv : stage.kv)
            total += kv.k.size() + kv.v.size() + kv.k_scale.size() + kv.v_scale.size() + kv.pooled.size();
    }
    return total;
}

// Writes the record's shape and collects the section lengths in payload order.
struct MetaWriter {
    std::vector<uint8_t> data;
    std::vector<uint64_t> lengths;
    bool measuring = false;
    size_t measured_bytes = 0;

    void raw(const void* source, size_t bytes) {
        if (measuring) { measured_bytes += bytes; return; }
        const uint8_t* first = static_cast<const uint8_t*>(source);
        data.insert(data.end(), first, first + bytes);
    }
    void u8(uint8_t value) { raw(&value, sizeof(value)); }
    void u32(uint32_t value) { raw(&value, sizeof(value)); }
    void u64(uint64_t value) { raw(&value, sizeof(value)); }
    void i32(int32_t value) { raw(&value, sizeof(value)); }
    void i64(int64_t value) { raw(&value, sizeof(value)); }
    void section(uint64_t bytes) { u64(bytes); if (!measuring) lengths.push_back(bytes); }

    void write_checkpoint(const ConversationCheckpoint& value) {
        u32(static_cast<uint32_t>(value.ids.size()));
        if (!value.ids.empty()) raw(value.ids.data(), value.ids.size() * sizeof(int32_t));
        u32(static_cast<uint32_t>(value.imgs.size()));
        for (const auto& key : value.imgs) {
            i64(key.start);
            u64(key.hash);
        }
        u64(value.used);
        section(value.gdn.size());
        section(value.ple.size());
        section(value.tails.size());
        section(value.dead.size());
        section(value.block_pos.size());
        u32(static_cast<uint32_t>(value.stage_parts.size()));
        for (const auto& part : value.stage_parts) write_checkpoint(part);
    }

    void write_kv(const ConversationKv& layer) {
        i32(layer.format);
        i64(layer.cells);
        i64(layer.heads);
        i64(layer.head_dim);
        i64(layer.page_size);
        i64(layer.pooled_rows);
        i64(layer.idx_dim);
        for (int64_t first : layer.first_units) i64(first);
        section(layer.k.size());
        section(layer.v.size());
        section(layer.k_scale.size());
        section(layer.v_scale.size());
        section(layer.pooled.size());
    }

    void write_stage(const SavedConversation& image) {
        i64(image.layer_lo);
        i64(image.layer_hi);
        for (int64_t value : image.geometry) i64(value);
        u8(image.cvec ? 1 : 0);
        write_checkpoint(image.live);
        u32(static_cast<uint32_t>(image.checkpoints.size()));
        for (const auto& checkpoint : image.checkpoints) write_checkpoint(checkpoint);
        u32(static_cast<uint32_t>(image.kv.size()));
        for (const auto& layer : image.kv) write_kv(layer);
    }
};

// Reads the record's shape and collects the section lengths in payload order. It allocates only
// ids and image keys, never a payload buffer.
struct MetaReader {
    const uint8_t* cursor = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;
    std::vector<uint64_t> lengths;

    size_t remaining() const { return ok ? static_cast<size_t>(end - cursor) : 0; }
    bool need(size_t bytes) {
        if (!ok || remaining() < bytes) {
            ok = false;
            return false;
        }
        return true;
    }
    uint8_t u8() { return need(1) ? *cursor++ : 0; }
    uint32_t u32() {
        uint32_t value = 0;
        if (need(4)) { std::memcpy(&value, cursor, 4); cursor += 4; }
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        if (need(8)) { std::memcpy(&value, cursor, 8); cursor += 8; }
        return value;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    bool section(uint64_t& bytes) {
        bytes = u64();
        if (!ok) return false;
        lengths.push_back(bytes);
        return true;
    }

    bool read_checkpoint(ConversationCheckpoint& out, uint32_t depth) {
        if (depth > kMaxPartDepth) { ok = false; return false; }
        const uint32_t ids = u32();
        if (!ok || ids > remaining() / sizeof(int32_t)) { ok = false; return false; }
        out.ids.resize(ids);
        if (ids != 0) {
            std::memcpy(out.ids.data(), cursor, ids * sizeof(int32_t));
            cursor += ids * sizeof(int32_t);
        }
        const uint32_t imgs = u32();
        if (!ok || imgs > remaining() / 16) { ok = false; return false; }
        out.imgs.resize(imgs);
        for (uint32_t i = 0; i < imgs; ++i) {
            out.imgs[i].start = i64();
            out.imgs[i].hash = u64();
        }
        out.used = u64();
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        const uint32_t parts = u32();
        if (!ok || parts > kMaxParts || parts > remaining() / 8) { ok = false; return false; }
        out.stage_parts.resize(parts);
        for (uint32_t i = 0; i < parts; ++i)
            if (!read_checkpoint(out.stage_parts[i], depth + 1)) return false;
        return ok;
    }

    bool read_kv(ConversationKv& out) {
        out.format = i32();
        out.cells = i64();
        out.heads = i64();
        out.head_dim = i64();
        out.page_size = i64();
        out.pooled_rows = i64();
        out.idx_dim = i64();
        for (int i = 0; i < 5; ++i) out.first_units[(size_t) i] = i64();
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        return ok;
    }

    bool read_stage(SavedConversation& out) {
        out.layer_lo = i64();
        out.layer_hi = i64();
        for (int i = 0; i < 18; ++i) out.geometry[i] = i64();
        out.cvec = u8() != 0;
        if (!read_checkpoint(out.live, 0)) return false;
        const uint32_t checkpoints = u32();
        if (!ok || checkpoints > kMaxCheckpoints || checkpoints > remaining() / 24) { ok = false; return false; }
        out.checkpoints.resize(checkpoints);
        for (uint32_t i = 0; i < checkpoints; ++i)
            if (!read_checkpoint(out.checkpoints[i], 0)) return false;
        const uint32_t kv = u32();
        if (!ok || kv > kMaxKv || kv > remaining() / 40) { ok = false; return false; }
        out.kv.resize(kv);
        for (uint32_t i = 0; i < kv; ++i)
            if (!read_kv(out.kv[i])) return false;
        return ok;
    }

    bool parse(ConversationDiskRecord& out) {
        const uint32_t stages = u32();
        if (!ok || stages == 0 || stages > kMaxStages) { ok = false; return false; }
        out.stages.resize(stages);
        for (uint32_t i = 0; i < stages; ++i)
            if (!read_stage(out.stages[i])) return false;
        return ok;
    }
};

std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// Validates the parsed metadata against the payload size. No allocation.
bool valid_lengths(ConversationDiskRecord& record, const std::vector<uint64_t>& lengths,
                   uint64_t payload_bytes) {
    uint64_t total = 0;
    for (const uint64_t bytes : lengths) {
        if (total > payload_bytes || bytes > payload_bytes - total) return false;
        total += bytes;
    }
    if (total != payload_bytes) return false;
    return count_sections(record) == lengths.size();
}

// One harness parsed from the metadata: shapes/ids/checkpoints/lengths, no payload bytes.
struct ParsedPanel {
    uint64_t base_tokens = 0;
    uint64_t end_tokens = 0;
    uint64_t payload_bytes = 0;
    uint64_t payload_checksum = 0;
    uint64_t payload_at = 0;                // file offset of the panel's payload
    ConversationDiskRecord delta;
    std::vector<uint64_t> lengths;
};

// The 5 running-state byte vectors of a checkpoint, in payload order.
std::array<std::vector<uint8_t>*, 5> live_arrays(ConversationCheckpoint& c) {
    return {&c.gdn.vec(), &c.ple.vec(), &c.tails.vec(), &c.dead.vec(), &c.block_pos.vec()};
}

// The K/V section lengths of a parsed panel, in payload order: [stage][layer] = the 5 buffer sizes.
// (The meta parse records the section lengths in `lengths` but never sizes the buffers themselves.)
std::vector<std::vector<std::array<size_t, 5>>> panel_kv_lens(const ConversationDiskRecord& delta,
                                                              const std::vector<uint64_t>& lengths) {
    std::vector<std::vector<std::array<size_t, 5>>> out(delta.stages.size());
    size_t i = 0;
    for (size_t si = 0; si < delta.stages.size(); ++si) {
        const SavedConversation& stage = delta.stages[si];
        i += 5;                                  // the live checkpoint's run-state
        i += 5 * stage.checkpoints.size();
        out[si].resize(stage.kv.size());
        for (size_t m = 0; m < stage.kv.size(); ++m)
            for (int e = 0; e < 5; ++e) out[si][m][(size_t) e] = (size_t) lengths[i++];
    }
    return out;
}

// True when a parsed panel's five-length section group at `at` holds any bytes: the position's
// running state is present.  A group past the parsed lengths is no state at all.
bool panel_sections_present(const ParsedPanel& panel, size_t at) {
    if (at + 5 > panel.lengths.size()) return false;
    for (size_t e = 0; e < 5; ++e)
        if (panel.lengths[at + e] != 0) return true;
    return false;
}

// True when the panel carries the chain's end state inline in EVERY stage (a panel of a chain with a
// sidecar writes none).
bool panel_live_has_state(const ParsedPanel& panel) {
    size_t at = 0;
    for (const auto& stage : panel.delta.stages) {
        if (!panel_sections_present(panel, at)) return false;
        at += 5 + 5 * stage.checkpoints.size() + 5 * stage.kv.size();
    }
    return true;
}

// True when the panel's checkpoint at `index` carries a running state in EVERY stage.
bool panel_checkpoint_has_state(const ParsedPanel& panel, size_t index) {
    size_t at = 0;
    for (const auto& stage : panel.delta.stages) {
        if (!panel_sections_present(panel, at + 5 + 5 * index)) return false;
        at += 5 + 5 * stage.checkpoints.size() + 5 * stage.kv.size();
    }
    return true;
}

// A byte-range comparison of two buffers, in chunks (no allocation).
bool buffer_range_equal(const ConversationBuffer& a, const ConversationBuffer& b, size_t offset,
                        size_t count) {
    std::array<uint8_t, 1u << 16> left{};
    std::array<uint8_t, 1u << 16> right{};
    size_t done = 0;
    while (done < count) {
        const size_t chunk = std::min(count - done, left.size());
        if (!a.read(left.data(), offset + done, chunk) || !b.read(right.data(), offset + done, chunk))
            return false;
        if (std::memcmp(left.data(), right.data(), chunk) != 0) return false;
        done += chunk;
    }
    return true;
}

// Compares the K/V a chain stores for [0, tokens) with an image's own K/V: EVERY retained page and
// completed indexer row, in the engine's reuse order.  The unused page padding and the moving spare
// row are excluded - the core reuse invariant - so a prefill that rounded a different cell cannot
// pass on its last page alone.  A DRAFT layer (one without an indexer, pooled_rows == 0 on both
// sides) is verified only through max(tokens-1, 0): its final cell may not have been computed when
// the output cap was reached, and the tail written after this proof re-copies that page - the same
// boundary the engine's own tail save keeps.  Both sides must share the stage and layer layout.
bool kv_prefix_equal(const std::vector<const SavedConversation*>& stored,
                     const std::vector<const SavedConversation*>& image, uint64_t tokens,
                     std::string& error) {
    if (stored.size() != image.size() || stored.empty()) {
        error = "the stored chain has a different stage count";
        return false;
    }
    const int64_t idx_block = strata::kernels::qsa_real_shapes().idx_block;
    for (size_t si = 0; si < stored.size(); ++si) {
        const SavedConversation& a = *stored[si];
        const SavedConversation& b = *image[si];
        if (a.kv.size() != b.kv.size()) {
            error = "the stored chain has a different layer count";
            return false;
        }
        for (size_t m = 0; m < a.kv.size(); ++m) {
            const ConversationKv& x = a.kv[m];
            const ConversationKv& y = b.kv[m];
            if (x.format != y.format || x.heads != y.heads || x.head_dim != y.head_dim ||
                x.page_size != y.page_size || x.idx_dim != y.idx_dim) {
                error = "the stored chain has a different K/V geometry";
                return false;
            }
            const ConversationBuffer* xs[5] = {&x.k, &x.v, &x.k_scale, &x.v_scale, &x.pooled};
            const ConversationBuffer* ys[5] = {&y.k, &y.v, &y.k_scale, &y.v_scale, &y.pooled};
            const int64_t page = std::max<int64_t>(1, x.page_size);
            const int64_t heads = std::max<int64_t>(1, x.heads);
            // The verified extent: one token shorter on a draft layer both sides agree is a draft.
            const uint64_t shared = x.pooled_rows == 0 && y.pooled_rows == 0 && tokens != 0
                ? tokens - 1
                : tokens;
            for (int e = 0; e < 5; ++e) {
                if (e == 4) {
                    if (x.pooled_rows == 0) continue;               // a layer without an indexer
                    const size_t rows = (size_t) (shared / (uint64_t) idx_block);
                    const size_t bytes = rows * (size_t) x.idx_dim * 4;
                    if (xs[4]->size() < bytes || ys[4]->size() < bytes) {
                        error = "the stored indexer is shorter than the shared prefix";
                        return false;
                    }
                    if (!buffer_range_equal(*xs[4], *ys[4], 0, bytes)) {
                        error = "the stored indexer rows differ";
                        return false;
                    }
                    continue;
                }
                if (x.cells < (int64_t) tokens || x.cells <= 0 ||
                    xs[e]->size() % (size_t) (x.cells * heads) != 0) {
                    error = "the stored K/V is shorter than the shared prefix";
                    return false;
                }
                const size_t per_cell = xs[e]->size() / (size_t) (x.cells * heads);
                if (ys[e]->size() != (size_t) y.cells * (size_t) heads * per_cell) {
                    error = "the K/V extents do not match";
                    return false;
                }
                const int64_t whole = ((int64_t) shared / page) * page;
                const size_t whole_bytes = (size_t) whole * (size_t) heads * per_cell;
                if (!buffer_range_equal(*xs[e], *ys[e], 0, whole_bytes)) {
                    error = "the stored K/V pages differ";
                    return false;
                }
                const int64_t partial = (int64_t) shared % page;
                if (partial == 0) continue;
                // The last page is not a contiguous token prefix: each head holds its own slice.
                for (int64_t head = 0; head < heads; ++head)
                    if (!buffer_range_equal(*xs[e], *ys[e],
                                            whole_bytes + (size_t) (head * page) * per_cell,
                                            (size_t) partial * per_cell)) {
                        error = "the stored K/V page differs";
                        return false;
                    }
            }
        }
    }
    return true;
}

// The common token prefix of a chain's stored CONTENT and a prompt, with the images below it.  Unlike
// conversation_common_prefix this is the content match, not the engine's reuse rule: it reaches the
// whole of the shorter side, so a chain that already holds an image reports it as contained.
uint64_t content_common_prefix(const ConversationCheckpoint& content, const std::vector<int32_t>& ids,
                               const std::vector<ConversationImageKey>& images) {
    const size_t limit = std::min(content.ids.size(), ids.size());
    const size_t n = common_token_prefix(content.ids.data(), ids.data(), limit);
    if (n == 0) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if ((uint64_t) image.start >= n) continue;
        if (j == content.imgs.size() || !(content.imgs[j++] == image)) return 0;
    }
    if (j != content.imgs.size()) return 0;
    return n;
}

// The index of the checkpoint whose ids are `tokens` long, or SIZE_MAX.
size_t find_checkpoint_index(const std::vector<ConversationCheckpoint>& checkpoints, size_t tokens) {
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (checkpoints[i].ids.size() == tokens) return i;
    return SIZE_MAX;
}

// The 5 K/V buffers of a layer, in payload order.
std::array<ConversationBuffer*, 5> kv_arrays(ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}

// ---- The bounded anchor-state sidecar ----
//
// A chain file stores the ids/imgs, the checkpoint positions and the K/V diffs.  The RUNNING STATES
// are large (a dense recurrent state per position) and unbounded if every panel keeps its own, so
// they live in <name>.states: one fixed slot per retained anchor.  A new anchor overwrites the slot
// the retention policy drops, so the state bytes on disk are bounded and no K/V byte is ever
// rewritten.  A panel of a chain that has a sidecar writes zero-length state sections; a panel that
// predates the sidecar keeps its inline states, which stay readable.
constexpr uint64_t kStateMagic = 0x5441545343525453ull;   // "STRCSTAT", little-endian
constexpr uint64_t kSlotMagic = 0x54415453544F4C53ull;    // "SLOTSTAT", little-endian
constexpr uint32_t kStateVersion = 1;
constexpr uint32_t kStateHeaderBytes = 128;
constexpr uint32_t kSlotHeaderBytes = 64;
constexpr uint32_t kMaxStateSlots = 1024;                 // a sanity bound, never a policy cap
constexpr uint64_t kSlotPinned = 1;
constexpr const char* kStateSuffix = ".states";

struct StateHeader {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t slot_count = 0;
    uint8_t identity[ConversationDiskIdentity::size] = {};
    uint8_t cvec = 0;
    uint8_t stage_count = 0;
    uint8_t padding0[6] = {};
    uint64_t slot_bytes = 0;          // the payload bytes of one slot
    uint64_t chain_cover = 0;         // the chain coverage the file was last published for
    uint64_t generation = 0;
    uint64_t checksum = 0;            // FNV over this header with the checksum field zeroed
    // The chain's terminal boundary: the input-prompt length its live state was captured at, so
    // positions at or after it are generated.  0 means unknown (an older build), and a replacement
    // of the tail then branches instead.  Carved out of the header's reserved bytes, so the layout
    // and the checksum still cover every byte.
    uint64_t terminal_from = 0;
    uint64_t origin_tokens = 0;       // first generated end later consumed as input; one ordinary slot
    uint8_t reserved[24] = {};
};
static_assert(sizeof(StateHeader) == kStateHeaderBytes, "the state header is a fixed 128 bytes");

struct SlotHeader {
    uint64_t magic = 0;
    uint64_t tokens = 0;
    uint64_t generation = 0;
    uint64_t payload_checksum = 0;
    uint64_t content_checksum = 0;    // the anchor's ids/images: the state binds to its content
    uint64_t used = 0;
    uint64_t flags = 0;               // bit 0: pinned - never a victim
    uint64_t reserved = 0;
};
static_assert(sizeof(SlotHeader) == kSlotHeaderBytes, "the slot header is a fixed 64 bytes");

uint64_t state_header_checksum(const StateHeader& header) {
    StateHeader copy = header;
    copy.checksum = 0;
    return conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&copy), sizeof(copy));
}

// The anchors a policy retains out of `positions`, mirroring conversation_checkpoint_keep: the
// earliest, the newest, and up to limit-1 more taken newest-first with doubling gaps.  A chain with
// a known terminal boundary (`terminal_from`) spends those slots on its HISTORICAL anchors - the
// positions at or below the boundary - first: the anchors inside the terminal region describe a
// reply the next rewrite replaces, while the historical ones stay the resume points of the
// conversation the client can ask for again after many rewritten replies.
std::vector<bool> anchor_keep(const std::vector<uint64_t>& positions, size_t limit, uint64_t terminal_from,
                              uint64_t origin_tokens) {
    const size_t n = positions.size();
    std::vector<bool> keep(n, false);
    if (n == 0) return keep;
    if (limit == 0) limit = 1;
    keep[0] = true;        // the earliest: the shared root every suffix branches from
    keep[n - 1] = true;    // the newest state the chain reached
    size_t ordinary = 1;
    const auto origin = std::lower_bound(positions.begin(), positions.end(), origin_tokens);
    if (limit > 1 && origin != positions.end() && *origin == origin_tokens) {
        const size_t at = (size_t) (origin - positions.begin());
        if (!keep[at]) { keep[at] = true; ++ordinary; }
    }
    std::vector<size_t> candidates;
    if (terminal_from != 0)
        for (size_t i = 0; i < n; ++i)
            if (!keep[i] && positions[i] <= terminal_from) candidates.push_back(i);
    if (candidates.empty())
        for (size_t i = 0; i < n; ++i)
            if (!keep[i]) candidates.push_back(i);
    for (size_t back = 0; ordinary < limit && back < candidates.size(); back = back == 0 ? 1 : back * 2) {
        keep[candidates[candidates.size() - 1 - back]] = true;
        ++ordinary;
    }
    return keep;
}

// The state bytes of one anchor: its position, the content it belongs to, and per stage the 5
// captured sections.
struct AnchorState {
    uint64_t tokens = 0;
    uint64_t used = 0;
    uint64_t content_checksum = 0;    // the anchor's ids/images: the state binds to its content
    bool pinned = false;
    std::vector<std::array<ConversationPayload, 5>> stages;
};

struct StateFile {
    std::FILE* file = nullptr;
    StateHeader header;
    std::vector<uint64_t> section_sizes;    // stage_count * 5
    std::vector<SlotHeader> slots;
    StateFile() = default;
    StateFile(const StateFile&) = delete;
    StateFile& operator=(const StateFile&) = delete;
    ~StateFile() {
        if (file != nullptr) std::fclose(file);
    }
};

uint64_t slot_offset(const StateHeader& header, size_t index) {
    return (uint64_t) kStateHeaderBytes + (uint64_t) header.stage_count * 5 * 8 +
           (uint64_t) index * (kSlotHeaderBytes + header.slot_bytes);
}

// The sidecar that belongs to a chain file: the same name in the same directory, with the state
// suffix.  The ONE derivation the writer (state_open), the index (state_path_for) and the discard
// path share, so the anchors a chain publishes are the anchors it later reads, charges and drops.
std::filesystem::path state_path_of_chain(const std::filesystem::path& chain_path) {
    const std::string filename = chain_path.filename().string();
    const size_t suffix = std::strlen(ConversationDiskStore::file_suffix);
    const std::string name =
        filename.size() > suffix &&
                filename.compare(filename.size() - suffix, suffix,
                                 ConversationDiskStore::file_suffix) == 0
            ? filename.substr(0, filename.size() - suffix)
            : filename;
    return chain_path.parent_path() / (name + kStateSuffix);
}

// The content fingerprint of a borrowed prefix: its ids and the images that start below it.
uint64_t prefix_checksum(const std::vector<int32_t>& ids,
                         const std::vector<ConversationImageKey>& imgs, size_t upto = SIZE_MAX) {
    uint64_t hash = conversation_disk_checksum_seed;
    const uint64_t count = std::min(ids.size(), upto);
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&count), sizeof(count), hash);
    if (!ids.empty())
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(ids.data()),
                                          count * sizeof(int32_t), hash);
    const auto image_end = upto == SIZE_MAX ? imgs.end() :
        std::find_if(imgs.begin(), imgs.end(), [&](const auto& key) { return key.start >= (int64_t) count; });
    const uint64_t images = (uint64_t) (image_end - imgs.begin());
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&images), sizeof(images), hash);
    for (auto it = imgs.begin(); it != image_end; ++it) {
        const auto& key = *it;
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.start), sizeof(key.start), hash);
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.hash), sizeof(key.hash), hash);
    }
    return hash;
}

// Lifts the running states out of a record: the panel's live state at `end` and every new
// checkpoint's, one anchor per token position.  A payload copy SHARES its bytes, so the lift itself
// costs nothing; the record is left with zero-length state sections, so the panel stores no state.
// The live anchor's POSITION is `end` while its ids are only the panel's slice, so its content
// fingerprint comes from `content_ids`/`content_imgs` - the chain's whole content at `end`, which
// is what a read compares a slot against.
std::vector<AnchorState> lift_anchors(ConversationDiskRecord& record, uint64_t end,
                                      const std::vector<int32_t>& content_ids,
                                      const std::vector<ConversationImageKey>& content_imgs) {
    std::vector<AnchorState> anchors;
    const size_t stages = record.stages.size();
    anchors.reserve(1 + record.stages.front().checkpoints.size());
    auto add = [&](uint64_t tokens, uint64_t used, bool pinned,
                   const std::vector<ConversationCheckpoint*>& parts, bool live) {
        AnchorState anchor;
        anchor.tokens = tokens;
        anchor.used = used;
        anchor.pinned = pinned;
        anchor.content_checksum = live ? prefix_checksum(content_ids, content_imgs)
                                       : prefix_checksum(parts.front()->ids, parts.front()->imgs);
        for (ConversationCheckpoint* part : parts) {
            std::array<ConversationPayload, 5> payloads{part->gdn, part->ple, part->tails, part->dead,
                                                       part->block_pos};
            anchor.stages.push_back(std::move(payloads));
            part->drop_state();
        }
        anchors.push_back(std::move(anchor));
    };
    {
        std::vector<ConversationCheckpoint*> parts;
        parts.reserve(stages);
        for (auto& stage : record.stages) parts.push_back(&stage.live);
        add(end, record.stages.front().live.used, record.stages.front().live.pinned, parts, true);
    }
    for (size_t j = 0; j < record.stages.front().checkpoints.size(); ++j) {
        std::vector<ConversationCheckpoint*> parts;
        parts.reserve(stages);
        bool complete = true;
        for (size_t si = 0; si < stages; ++si) {
            auto& stage = record.stages[si];
            const auto& live = anchors.front().stages[si];
            const bool needs_state = std::any_of(live.begin(), live.end(),
                [](const auto& bytes) { return !bytes.empty(); });
            if (needs_state && !stage.checkpoints[j].has_state()) complete = false;
            parts.push_back(&stage.checkpoints[j]);
        }
        if (!complete) continue;
        add((uint64_t) parts.front()->ids.size(), parts.front()->used, parts.front()->pinned, parts,
            false);
    }
    return anchors;
}


// The fixed layout of one stage of a record, from its own image.
ConversationDiskStore::StageShape stage_shape_of(const SavedConversation& stage) {
    ConversationDiskStore::StageShape shape;
    shape.layer_lo = stage.layer_lo;
    shape.layer_hi = stage.layer_hi;
    shape.geometry = stage.geometry;
    shape.cvec = stage.cvec;
    if (!stage.kv.empty()) {
        shape.format = stage.kv.front().format;
        shape.heads = stage.kv.front().heads;
        shape.head_dim = stage.kv.front().head_dim;
        shape.page_size = stage.kv.front().page_size;
        shape.idx_dim = stage.kv.front().idx_dim;
    }
    return shape;
}

// One shape per stage of a record, in stage order.
bool stage_shapes(const ConversationDiskRecord& record,
                  std::vector<ConversationDiskStore::StageShape>& out, std::string& error) {
    out.clear();
    if (record.stages.empty()) {
        error = "the record has no stage";
        return false;
    }
    out.reserve(record.stages.size());
    for (const auto& stage : record.stages) out.push_back(stage_shape_of(stage));
    return true;
}

// Reads and validates the chain header and every panel header (skipping payloads).  The metadata of
// a panel is parsed only when that panel can contribute to [0, stop_tokens) - the first panel is
// always parsed because it carries the stage layout - so a prefix read never loads a later panel's
// metadata.  Every panel HEADER is still walked, so a chain whose panels do not run in order (a
// rewind rewrote the tail) is validated the same way.
bool read_chain_meta(std::FILE* file, const ConversationDiskIdentity& identity,
                     uint64_t budget_bytes, ChainHeader& header, std::vector<ParsedPanel>& panels,
                     std::string& error, uint64_t stop_tokens = UINT64_MAX) {
    auto bad = [&](const char* why) {
        error = std::string("corrupt conversation record: ") + why;
        return false;
    };
    if (!read_exact(file, &header, sizeof(header))) return bad("header");
    if (header.magic != kMagic || header.version != ConversationDiskStore::format_version ||
        std::memcmp(header.identity, identity.bytes.data(), ConversationDiskIdentity::size) != 0 ||
        header.base_kind > kBaseChain || header.panel_count > kMaxPanels)
        return bad("header");
    uint64_t expected = header.base_kind != kBaseNone ? header.base_tokens : 0;
    if (header.total_tokens < expected) return bad("lengths");
    panels.clear();
    panels.reserve(static_cast<size_t>(header.panel_count));
    uint64_t payload_at = kChainHeaderBytes;
    uint64_t last_end = expected;
    for (uint64_t k = 0; k < header.panel_count; ++k) {
        PanelHeader p{};
        if (!read_exact(file, &p, sizeof(p))) return bad("panel header");
        if (p.magic != kPanelMagic || p.base_tokens > expected ||
            p.base_tokens < (header.base_kind != kBaseNone ? header.base_tokens : 0) ||
            (k == 0 && p.base_tokens != expected) || p.end_tokens <= p.base_tokens)
            return bad("panel header");
        if (p.metadata_bytes > kMaxMetadataBytes || p.payload_bytes > budget_bytes)
            return bad("panel sizes");
        if (k == 0 || p.base_tokens < stop_tokens) {
            std::vector<uint8_t> meta(static_cast<size_t>(p.metadata_bytes));
            if (!read_exact(file, meta.data(), meta.size())) return bad("metadata read");
            if (conversation_disk_checksum(meta.data(), meta.size()) != p.metadata_checksum)
                return bad("metadata checksum");
            ParsedPanel panel;
            panel.base_tokens = p.base_tokens;
            panel.end_tokens = p.end_tokens;
            panel.payload_bytes = p.payload_bytes;
            panel.payload_checksum = p.payload_checksum;
            panel.payload_at = payload_at + sizeof(p) + p.metadata_bytes;
            MetaReader reader{meta.data(), meta.data() + meta.size(), true, {}};
            if (!reader.parse(panel.delta) || !reader.ok) return bad("metadata layout");
            if (panel.delta.stages.empty() || panel.delta.stages.front().cvec != (header.cvec != 0))
                return bad("cvec");
            if (!valid_lengths(panel.delta, reader.lengths, p.payload_bytes)) return bad("section sizes");
            // Every K/V layer's cells must be the panel's end rounded up to its page size.
            for (const auto& stage : panel.delta.stages)
                for (const auto& kv : stage.kv) {
                    const int64_t page = std::max<int64_t>(1, kv.page_size);
                    const int64_t aligned = ((int64_t) p.end_tokens + page - 1) / page * page;
                    if (kv.cells != aligned) return bad("kv cells");
                }
            panel.lengths = std::move(reader.lengths);
            panels.push_back(std::move(panel));
            if (k + 1 < header.panel_count &&
                std::fseek(file, static_cast<long>(p.payload_bytes), SEEK_CUR) != 0)
                return bad("panel sizes");
        } else if (std::fseek(file, static_cast<long>(p.metadata_bytes + p.payload_bytes), SEEK_CUR) != 0) {
            return bad("panel sizes");
        }
        expected = p.end_tokens;
        last_end = p.end_tokens;
        payload_at += sizeof(p) + p.metadata_bytes + p.payload_bytes;
    }
    if (panels.empty()) return bad("no panels");
    if (last_end != header.total_tokens) return bad("lengths");
    return true;
}


// True when two deltas (or the base and a delta) describe the same stage layout.
bool stages_compatible(const ConversationDiskRecord& a, const ConversationDiskRecord& b,
                       std::string& error) {
    if (a.stages.size() != b.stages.size()) {
        error = "incompatible stage count";
        return false;
    }
    for (size_t i = 0; i < a.stages.size(); ++i) {
        if (a.stages[i].geometry != b.stages[i].geometry ||
            a.stages[i].layer_lo != b.stages[i].layer_lo ||
            a.stages[i].layer_hi != b.stages[i].layer_hi || a.stages[i].cvec != b.stages[i].cvec ||
            a.stages[i].kv.size() != b.stages[i].kv.size()) {
            error = "incompatible stage shape";
            return false;
        }
        for (size_t m = 0; m < a.stages[i].kv.size(); ++m) {
            const ConversationKv& x = a.stages[i].kv[m];
            const ConversationKv& y = b.stages[i].kv[m];
            // pooled_rows scales with the panel's extent (the pooled row count at its end), so it
            // is not part of the fixed geometry: only the newest panel's value is merged later.
            if (x.format != y.format || x.heads != y.heads || x.head_dim != y.head_dim ||
                x.page_size != y.page_size || x.idx_dim != y.idx_dim) {
                error = "incompatible K/V geometry across panels";
                return false;
            }
        }
    }
    return true;
}

// A byte-exact copy of [offset, offset + count) of a segmented buffer.
bool buffer_slice(const ConversationBuffer& src, size_t offset, size_t count,
                  ConversationBuffer& out) {
    out.resize(0);
    if (count == 0) return true;
    if (offset > src.size() || count > src.size() - offset) return false;
    std::vector<uint8_t> bytes(count);
    if (!src.read(bytes.data(), offset, count)) return false;
    out.resize(count);
    size_t done = 0;
    return out.visit(0, count, [&](uint8_t* p, size_t chunk, size_t) {
        std::memcpy(p, bytes.data() + done, chunk);
        done += chunk;
        return true;
    });
}

// Trims a full stage image IN PLACE to the delta [from, end): the token ids and the image keys,
// the K/V tails from the page that holds `from` (the pooled rows from the completed row at
// `from`), and only the checkpoints above `from` - plus a PINNED one at or below it, whose state
// then travels in the chain's own bounded sidecar instead of a panel's inline metadata.  The
// running state stays: it describes the image's own end.  A full capture's buffers are page-capped
// at its own end, which is what makes the tail offsets derivable.  No CUDA call.
bool slice_stage_from(SavedConversation& stage, uint64_t from, std::string& error) {
    const uint64_t end = (uint64_t) stage.live.ids.size();
    if (from >= end) {
        error = "the appended delta is empty";
        return false;
    }
    std::vector<ConversationCheckpoint> kept;
    kept.reserve(stage.checkpoints.size());
    for (auto& checkpoint : stage.checkpoints) {
        if ((uint64_t) checkpoint.ids.size() > from) {
            kept.push_back(std::move(checkpoint));
            continue;
        }
        // A pinned anchor the delta shares with the content it continues: keep it only when its ids
        // and images are exactly this image's own prefix, so a pin can never carry another
        // prefix's state into the chain.
        if (!checkpoint.pinned || checkpoint.ids.empty() || checkpoint.ids.size() > end ||
            !std::equal(checkpoint.ids.begin(), checkpoint.ids.end(), stage.live.ids.begin()))
            continue;
        size_t j = 0;
        bool same_imgs = true;
        for (const auto& image : stage.live.imgs) {
            if ((uint64_t) image.start >= checkpoint.ids.size()) break;
            if (j == checkpoint.imgs.size() || !(checkpoint.imgs[j++] == image)) {
                same_imgs = false;
                break;
            }
        }
        if (!same_imgs || j != checkpoint.imgs.size()) continue;
        kept.push_back(std::move(checkpoint));
    }
    stage.checkpoints = std::move(kept);
    stage.live.ids.erase(stage.live.ids.begin(), stage.live.ids.begin() + (std::ptrdiff_t) from);
    stage.live.imgs.erase(std::remove_if(stage.live.imgs.begin(), stage.live.imgs.end(),
        [&](const ConversationImageKey& image) { return (uint64_t) image.start < from; }),
        stage.live.imgs.end());
    for (ConversationKv& kv : stage.kv) {
        const int64_t page = std::max<int64_t>(1, kv.page_size);
        std::array<ConversationBuffer*, 5> buffers = kv_arrays(kv);
        std::array<int64_t, 5> first{};
        // A DRAFT layer (one without an indexer) is copied from one token earlier: its cell at
        // `from - 1` may not have been computed when the output cap was reached, so the overlap page
        // is re-copied whole - the same boundary the engine's own tail save keeps
        // (conversation_kv_tail_save).  The indexer rows are unaffected.
        const bool draft = kv.pooled_rows == 0;
        const uint64_t cell_from = draft && from != 0 ? from - 1 : from;
        for (int e = 0; e < 5; ++e) {
            const int64_t total_units = kv_end_units((int64_t) end, page, e);
            const bool indexed = kv.pooled_rows != 0;
            const int64_t start_unit = e == 4
                ? (indexed ? (int64_t) (from / strata::kernels::qsa_real_shapes().idx_block) : 0)
                : (int64_t) ((cell_from / page) * page);
            if (total_units <= start_unit || total_units <= 0) {
                error = "the image extent does not hold the appended tail";
                return false;
            }
            if (buffers[(size_t) e]->size() % (size_t) total_units != 0) {
                error = "the image is not a full capture of its own extent";
                return false;
            }
            const size_t per_unit = buffers[(size_t) e]->size() / (size_t) total_units;
            const size_t offset = (size_t) start_unit * per_unit;
            ConversationBuffer tail;
            if (!buffer_slice(*buffers[(size_t) e], offset, buffers[(size_t) e]->size() - offset, tail)) {
                error = "the appended tail cannot be sliced";
                return false;
            }
            *buffers[(size_t) e] = std::move(tail);
            first[(size_t) e] = start_unit;
        }
        kv.first_units = first;
    }
    return true;
}

} // namespace

// Defined below, after the store's own helpers: declared here because index_file and the read path
// use them before their definitions.
static bool state_open(const std::filesystem::path& chain_path, const ConversationDiskIdentity& identity,
                       bool cvec, const std::vector<std::array<uint64_t, 5>>* sizes, size_t slots,
                       bool create, StateFile& out, std::string& error);
static bool state_read(const StateFile& file, uint64_t tokens, uint64_t content,
                       const std::vector<std::array<std::vector<uint8_t>*, 5>>& out, std::string& error);

size_t ConversationDiskStore::metadata_bytes(const SavedConversation& image) {
    MetaWriter writer;
    writer.measuring = true;
    writer.u32(static_cast<uint32_t>(image.stage_images.size() + 1));
    writer.write_stage(image);
    for (const auto& stage : image.stage_images) writer.write_stage(stage);
    return writer.measured_bytes;
}

size_t ConversationDiskStore::metadata_bytes(const ConversationCheckpoint& checkpoint) {
    MetaWriter writer;
    writer.measuring = true;
    writer.write_checkpoint(checkpoint);
    return writer.measured_bytes;
}

size_t ConversationDiskStore::metadata_bytes(const ConversationDiskRecord& record) {
    MetaWriter writer;
    writer.measuring = true;
    writer.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) writer.write_stage(stage);
    return writer.measured_bytes;
}

ConversationDiskIdentity ConversationDiskIdentity::from_string(std::string_view text) {
    ConversationDiskIdentity identity;
    uint64_t lanes[4] = {};
    for (uint64_t lane = 0; lane < 4; ++lane) {
        const uint64_t seed = conversation_disk_checksum_seed + lane * 0x9e3779b97f4a7c15ull;
        lanes[lane] = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(text.data()), text.size(), seed);
    }
    std::memcpy(identity.bytes.data(), lanes, sizeof(lanes));
    return identity;
}

// One name class per purpose: 'c' for a conversation's own chain, 'p' for the shared system-prompt
// (chain root) prefix chain.  The rest of the name is the FNV-1a over the token prefix (with its
// image keys and steering mode), so the same prefix makes the same name in every conversation that
// shares it.  For a conversation chain the engine picks the name ONCE at seed time from the seed's
// content; appends keep it, so the file key is stable while the chain grows.
std::string disk_name(char cls, const std::vector<int32_t>& ids,
                      const std::vector<ConversationImageKey>& images, bool cvec) {
    uint64_t hash = conversation_disk_checksum_seed;
    const uint64_t count = ids.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&count), sizeof(count), hash);
    if (!ids.empty())
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(ids.data()),
                                          ids.size() * sizeof(int32_t), hash);
    const uint64_t image_count = images.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&image_count), sizeof(image_count), hash);
    for (const auto& key : images) {
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.start), sizeof(key.start), hash);
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.hash), sizeof(key.hash), hash);
    }
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&cvec), sizeof(cvec), hash);
    return std::string(1, cls) + hex64(hash) + "-" + hex64(count);
}

std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    return disk_name('c', ids, images, cvec);
}

std::string conversation_disk_prefix_name(const std::vector<int32_t>& ids,
                                          const std::vector<ConversationImageKey>& images, bool cvec) {
    return disk_name('p', ids, images, cvec);
}

bool ConversationDiskStore::valid_name(const std::string& name) {
    if (name.empty() || name.size() > kMaxNameBytes || name == "." || name == "..") return false;
    for (const char c : name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')) return false;
    return true;
}

std::filesystem::path ConversationDiskStore::path_for(const std::string& name) const {
    return options_.directory / (name + file_suffix);
}

const ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) const {
    for (const auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}

ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) {
    for (auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}

uint64_t ConversationDiskStore::borrowed_prefix(const std::string& name) const {
    uint64_t longest = 0;
    for (const auto& entry : entries_)
        if (entry.base == name && entry.base_tokens > longest) longest = entry.base_tokens;
    return longest;
}


size_t ConversationDiskStore::lru_victim(const std::string& skip, const std::string& keep) const {
    // tier 0: ordinary conversation chains.  tier 1: system-prompt chains nobody references.
    // tier 2: chains another chain borrows a prefix from (only evicted when the budget must hold).
    for (int tier = 0; tier < 3; ++tier) {
        size_t victim = SIZE_MAX;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!skip.empty() && entries_[i].name == skip) continue;
            if (!keep.empty() && entries_[i].name == keep) continue;
            const bool referenced = entries_[i].refs != 0;
            const int tier_of = referenced ? 2 : (entries_[i].root ? 1 : 0);
            if (tier_of != tier) continue;
            if (entries_[i].stamp < oldest) {
                oldest = entries_[i].stamp;
                victim = i;
            }
        }
        if (victim != SIZE_MAX) return victim;
    }
    return SIZE_MAX;
}

void ConversationDiskStore::evict(size_t index) {
    // A branch chain cannot load without the chain it borrows from: remove the dependents first,
    // then the chain itself.  The graph is acyclic, so the recursion is bounded by max_base_depth.
    const std::string name = entries_[index].name;
    for (size_t i = 0; i < entries_.size();) {
        if (i != index && entries_[i].base == name) {
            evict(i);
            if (i < index) --index;
            continue;
        }
        ++i;
    }
    std::error_code error;
    std::filesystem::remove(path_for(name), error);
    state_discard(name, 0, true);
    bytes_ -= entries_[index].bytes;
    if (!entries_[index].base.empty())
        if (Entry* base = find(entries_[index].base)) --base->refs;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    ++evictions_;
}

ConversationDiskStatus ConversationDiskStore::make_room(uint64_t incoming, const std::string& skip,
                                                        const std::string& keep, bool adds_record,
                                                        std::string& error) {
    auto over_budget = [&]() {
        return bytes_ + incoming > options_.budget_bytes ||
               (options_.max_records != 0 && adds_record && entries_.size() + 1 > options_.max_records);
    };
    while (over_budget()) {
        const size_t victim = lru_victim(skip, keep);
        if (victim == SIZE_MAX) {
            error = "the disk cache cannot hold the record within its budget";
            return ConversationDiskStatus::invalid;
        }
        evict(victim);
    }
    std::error_code space_error;
    const std::filesystem::space_info space = std::filesystem::space(options_.directory, space_error);
    if (!space_error && space.available < incoming + options_.min_free_bytes) {
        uint64_t free_bytes = space.available;
        while (free_bytes < incoming + options_.min_free_bytes) {
            const size_t victim = lru_victim(skip, keep);
            if (victim == SIZE_MAX) {
                error = "the disk cache has no free space for the record";
                return ConversationDiskStatus::failed;
            }
            free_bytes += entries_[victim].bytes;
            evict(victim);
        }
    }
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::open(const ConversationDiskOptions& options, std::string& error) {
    options_ = options;
    entries_.clear();
    bytes_ = 0;
    stamp_ = 0;
    recoveries_ = 0;
    if (options_.budget_bytes == 0) return ConversationDiskStatus::disabled;
    if (options_.directory.empty()) {
        error = "the disk cache has no directory";
        return ConversationDiskStatus::invalid;
    }
    std::error_code error_code;
    std::filesystem::create_directories(options_.directory, error_code);
    if (error_code) {
        error = "cannot create " + options_.directory.string();
        return ConversationDiskStatus::failed;
    }
    const size_t suffix = std::strlen(file_suffix);
    const std::string temporary_suffix = std::string(file_suffix) + ".tmp";
    struct Pending {
        std::filesystem::path path;
        std::string name;
        uint8_t base_kind = kBaseNone;
        std::string base;
    };
    std::vector<Pending> pending;
    std::vector<std::filesystem::file_time_type> written;
    for (std::filesystem::directory_iterator it(options_.directory, error_code), end; !error_code && it != end;
         it.increment(error_code)) {
        std::error_code type_error;
        if (!it->is_regular_file(type_error)) continue;
        const std::string filename = it->path().filename().string();
        if (filename.size() > temporary_suffix.size() &&
            filename.compare(filename.size() - temporary_suffix.size(), temporary_suffix.size(),
                             temporary_suffix) == 0) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        if (filename.size() <= suffix || filename.compare(filename.size() - suffix, suffix, file_suffix) != 0)
            continue;
        const std::string name = filename.substr(0, filename.size() - suffix);
        if (!valid_name(name)) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        // The chain header first: a chain whose base is another chain must be indexed after it.
        std::FILE* file = std::fopen(it->path().string().c_str(), "rb");
        ChainHeader header;
        const bool readable = file != nullptr && read_exact(file, &header, sizeof(header));
        if (file != nullptr) std::fclose(file);
        if (!readable || header.magic != kMagic || header.version != format_version ||
            header.base_kind > kBaseChain || header.panel_count == 0 || header.panel_count > kMaxPanels) {
            std::filesystem::remove(it->path(), type_error);
            ++corruptions_;
            continue;
        }
        Pending item;
        item.path = it->path();
        item.name = name;
        item.base_kind = header.base_kind;
        if (header.base_kind != kBaseNone) item.base = base_name_of(header);
        pending.push_back(std::move(item));
        written.push_back(it->last_write_time(type_error));
    }
    std::vector<size_t> order(pending.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (written[a] != written[b]) return written[a] < written[b];
        return pending[a].name < pending[b].name;
    });
    // Index in dependency order: a chain waits until the chain it borrows from is indexed.  A chain
    // that no pass can index has a missing, too short, cyclic, or incompatible base: it cannot
    // load, so it is dropped (the engine re-reads the prompt).
    std::vector<bool> done(pending.size(), false);
    size_t remaining = pending.size();
    while (remaining != 0) {
        bool progress = false;
        for (const size_t i : order) {
            if (done[i]) continue;
            if (pending[i].base_kind != kBaseNone && find(pending[i].base) == nullptr) continue;
            Entry entry;
            const ConversationDiskStatus status = index_file(pending[i].path, pending[i].name, entry, error);
            done[i] = true;
            --remaining;
            progress = true;
            if (status == ConversationDiskStatus::ok) {
                entry.stamp = ++stamp_;
                bytes_ += entry.bytes;
                entries_.push_back(std::move(entry));
                ++recoveries_;
            } else if (status == ConversationDiskStatus::corrupt) {
                std::filesystem::remove(pending[i].path, error_code);
                ++corruptions_;
            }
            // A read failure keeps the file and skips it, so a transient error loses no data.
        }
        if (!progress) break;
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        if (done[i]) continue;
        std::filesystem::remove(pending[i].path, error_code);
        ++corruptions_;
    }
    // Reference accounting: every branch chain pins the chain it borrows from.
    for (auto& entry : entries_)
        if (!entry.base.empty())
            if (Entry* base = find(entry.base)) ++base->refs;
    // A sidecar without its chain is garbage: drop it, and any leftover temporary.
    for (std::filesystem::directory_iterator it(options_.directory, error_code), end; !error_code && it != end;
         it.increment(error_code)) {
        std::error_code type_error;
        if (!it->is_regular_file(type_error)) continue;
        const std::string filename = it->path().filename().string();
        const size_t suffix_size = std::strlen(kStateSuffix);
        if (filename.size() > suffix_size + 4 &&
            filename.compare(filename.size() - 4, 4, ".tmp") == 0)
            std::filesystem::remove(it->path(), type_error);
        if (filename.size() <= suffix_size ||
            filename.compare(filename.size() - suffix_size, suffix_size, kStateSuffix) != 0)
            continue;
        const std::string owner = filename.substr(0, filename.size() - suffix_size);
        if (find(owner) == nullptr) std::filesystem::remove(it->path(), type_error);
    }
    while (bytes_ > options_.budget_bytes ||
           (options_.max_records != 0 && entries_.size() > options_.max_records)) {
        const size_t victim = lru_victim(std::string(), std::string());
        if (victim == SIZE_MAX) break;
        evict(victim);
    }
    error.clear();
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::index_file(const std::filesystem::path& path,
                                                         const std::string& name, Entry& entry,
                                                         std::string& error) const {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + path.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        if (file != nullptr) {
            std::fclose(file);
            file = nullptr;
        }
        error = std::string("invalid conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    ChainHeader header;
    std::vector<ParsedPanel> panels;
    if (!read_chain_meta(file, options_.identity, options_.budget_bytes, header, panels, error))
        return bad(error.empty() ? "header" : error.c_str());
    std::fclose(file);
    file = nullptr;
    // Build the merged ids/points from the metadata (no payload reads).
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> all_imgs;
    const uint64_t base_tokens = header.base_kind != kBaseNone ? header.base_tokens : 0;
    const Entry* base = nullptr;
    if (header.base_kind != kBaseNone) {
        base = find(base_name_of(header));
        if (base == nullptr) return bad("base chain");
        if (base->points.empty() || base->cover < base_tokens) return bad("base coverage");
        if (base->depth + 1 > max_base_depth) return bad("base depth");
        if (base->shape.size() != panels.front().delta.stages.size()) return bad("base stages");
        std::vector<StageShape> shape;
        if (!stage_shapes(panels.front().delta, shape, error)) return bad("stage shape");
        if (base->shape != shape) return bad("base shape");
        std::vector<int32_t> prefix(base->points.front().ids.begin(),
                                    base->points.front().ids.begin() + (std::ptrdiff_t) base_tokens);
        std::vector<ConversationImageKey> prefix_imgs;
        for (const auto& image : base->points.front().imgs) {
            if ((uint64_t) image.start >= base_tokens) continue;
            prefix_imgs.push_back(image);
        }
        if (header.base_prefix_checksum != 0 &&
            header.base_prefix_checksum != prefix_checksum(prefix, prefix_imgs))
            return bad("base prefix");
        ids = std::move(prefix);
        all_imgs = std::move(prefix_imgs);
        if ((uint64_t) ids.size() != base_tokens) return bad("base tokens");
    }
    ids.reserve((size_t) header.total_tokens);
    for (size_t k = 0; k < panels.size(); ++k) {
        if (k != 0 && !stages_compatible(panels.front().delta, panels[k].delta, error))
            return bad(error.c_str());
        for (size_t si = 0; si < panels[k].delta.stages.size(); ++si) {
            const SavedConversation& stage = panels[k].delta.stages[si];
            if ((uint64_t) stage.live.ids.size() != panels[k].end_tokens - panels[k].base_tokens)
                return bad("token slice");
            if (si == 0) {
                ids.resize((size_t) panels[k].base_tokens);
                all_imgs.erase(std::remove_if(all_imgs.begin(), all_imgs.end(),
                    [&](const ConversationImageKey& image) { return image.start >= (int64_t) panels[k].base_tokens; }), all_imgs.end());
                ids.insert(ids.end(), stage.live.ids.begin(), stage.live.ids.end());
                all_imgs.insert(all_imgs.end(), stage.live.imgs.begin(), stage.live.imgs.end());
            }
        }
    }
    if ((uint64_t) ids.size() != header.total_tokens) return bad("lengths");
    entry.name = name;
    entry.cover = header.total_tokens;
    entry.cvec = header.cvec != 0;
    entry.root = !name.empty() && name[0] == 'p';
    entry.base = base != nullptr ? base_name_of(header) : std::string();
    entry.base_tokens = base_tokens;
    entry.depth = base != nullptr ? base->depth + 1 : 0;
    entry.refs = 0;
    // The chain's own bytes plus its sidecar: the store's budget charges the anchors it keeps.
    entry.bytes = std::filesystem::file_size(path) + state_file_bytes(name);
    entry.points.clear();
    // The inline state of each point: whether a PANEL carries it.  The retained sidecar slots are
    // settled in below, because the retention rotates them - a position whose state the policy
    // dropped must not stay restorable, and one it kept must not be missing until the next reopen.
    std::vector<bool> states;
    if (!stage_shapes(panels.front().delta, entry.shape, error)) return bad("stage shape");
    {
        ConversationCheckpoint final_point;
        final_point.ids = std::move(ids);
        final_point.imgs = std::move(all_imgs);
        final_point.pinned = panels.back().delta.stages.front().live.pinned;
        // The chain's end: the last panel's own inline state, or the sidecar's slot there (below).
        states.push_back(panel_live_has_state(panels.back()));
        entry.points.push_back(std::move(final_point));
    }
    for (auto& panel : panels) {
        // Drop the points a later panel overwrote, with their state flags.
        size_t kept = 1;
        for (size_t i = 1; i < entry.points.size(); ++i) {
            if (entry.points[i].ids.size() > panel.base_tokens) continue;
            if (kept != i) {
                entry.points[kept] = std::move(entry.points[i]);
                states[kept] = states[i];
            }
            ++kept;
        }
        entry.points.resize(kept);
        states.resize(kept);
        const std::vector<ConversationCheckpoint>& checkpoints = panel.delta.stages.front().checkpoints;
        for (size_t j = 0; j < checkpoints.size(); ++j) {
            const ConversationCheckpoint& c = checkpoints[j];
            if (c.ids.size() > entry.cover ||
                !std::equal(c.ids.begin(), c.ids.end(), entry.points.front().ids.begin())) continue;
            ConversationCheckpoint point;
            point.ids = c.ids;
            point.imgs = c.imgs;
            point.used = c.used;
            point.pinned = c.pinned;
            states.push_back(panel_checkpoint_has_state(panel, j));
            entry.points.push_back(std::move(point));
        }
    }
    // The chain's anchors: their positions live in the sidecar, so a panel does not repeat them.
    // The sidecar also carries the chain's terminal boundary, which a replacement of the tail needs.
    StateFile anchor_file;
    std::string side_error;
    if (state_open(path, options_.identity, header.cvec != 0, nullptr, 0, false, anchor_file, side_error))
        entry.terminal_from = anchor_file.header.terminal_from;
    settle_entry_states(entry, std::move(states));
    return ConversationDiskStatus::ok;
}

bool ConversationDiskStore::best(const std::vector<int32_t>& ids,
                                 const std::vector<ConversationImageKey>& images, bool cvec,
                                 ConversationDiskMatch& match) const {
    const Entry* chosen = nullptr;
    int64_t chosen_tokens = 0;
    for (const auto& entry : entries_) {
        if (entry.cvec != cvec) continue;
        for (size_t i = 0; i < entry.points.size(); ++i) {
            // A point whose running state is nowhere cannot resume a restore: offering it would make
            // the caller read the record only to discard the whole chain.
            if (i < entry.point_states.size() && !entry.point_states[i]) continue;
            const ConversationCheckpoint& point = entry.points[i];
            const int64_t tokens = conversation_prefix(point, ids, images);
            const bool longer = tokens > chosen_tokens;
            // An equal-length tie prefers the SMALLER record: a 24k system-prompt record reads back
            // far less than the 80k conversation that contains the same prefix, for the same answer.
            const bool equal_smaller = tokens == chosen_tokens && tokens > 0 && chosen != nullptr &&
                                       (entry.bytes < chosen->bytes ||
                                        (entry.bytes == chosen->bytes && entry.stamp > chosen->stamp));
            if (longer || equal_smaller) {
                chosen_tokens = tokens;
                chosen = &entry;
            }
        }
    }
    if (chosen == nullptr || chosen_tokens <= 0) return false;
    match.name = chosen->name;
    match.tokens = chosen_tokens;
    // The K/V extent the chain shares with the prompt: the common prefix of its stored CONTENT,
    // which can reach past the matched point (the prompt diverges inside the chain) and can stop
    // before it (the point sits above a divergence the images make unusable).  The caller proves
    // those bytes before it reuses them.
    match.prefix_tokens = chosen->points.empty()
        ? chosen_tokens
        : conversation_common_prefix(chosen->points.front(), ids, images);
    return true;
}

bool ConversationDiskStore::shared_prefix(const std::vector<int32_t>& ids,
                                          const std::vector<ConversationImageKey>& images, bool cvec,
                                          ConversationDiskMatch& match) const {
    const Entry* chosen = nullptr;
    uint64_t chosen_common = 0;
    for (const auto& entry : entries_) {
        if (entry.cvec != cvec || entry.points.empty()) continue;
        // The chain's stored CONTENT, not a restore point: a chain whose only point is longer than
        // the prompt still shares its whole prefix with it, and the extent is what a borrow needs.
        const uint64_t common = content_common_prefix(entry.points.front(), ids, images);
        if (common == 0) continue;
        const bool longer = common > chosen_common;
        // An equal extent prefers the SMALLER record: it reads back less for the same prefix.
        const bool equal_smaller = common == chosen_common && chosen != nullptr &&
                                   (entry.bytes < chosen->bytes ||
                                    (entry.bytes == chosen->bytes && entry.stamp > chosen->stamp));
        if (longer || equal_smaller) {
            chosen_common = common;
            chosen = &entry;
        }
    }
    if (chosen == nullptr) return false;
    match.name = chosen->name;
    match.tokens = 0;                  // this call selects no restore point
    match.prefix_tokens = (int64_t) chosen_common;
    return true;
}

bool ConversationDiskStore::can_append(const std::string& name, const std::vector<int32_t>& ids,
                                       const std::vector<ConversationImageKey>& images, bool cvec,
                                       uint64_t tokens) const {
    const Entry* entry = find(name);
    if (entry == nullptr || entry->rewrite_from != UINT64_MAX) return false;
    if (tokens == 0 || entry->cover != tokens || entry->cvec != cvec) return false;
    if (entry->points.empty() || (uint64_t) entry->points.front().ids.size() != tokens) return false;
    if (ids.size() < tokens) return false;
    const ConversationCheckpoint& point = entry->points.front();
    if (!std::equal(point.ids.begin(), point.ids.end(), ids.begin())) return false;
    size_t j = 0;
    for (const auto& image : images) {
        if ((uint64_t) image.start >= tokens) continue;
        if (j == point.imgs.size() || !(point.imgs[j++] == image)) return false;
    }
    return j == point.imgs.size();
}

// Validates that a seed/delta's checkpoints are prefixes of the chain content.  Returns the point
// list (final point first, then the checkpoints, ids/imgs only, with the caller's pin flags).  A
// checkpoint at or below the delta's base is accepted only when it is PINNED and its ids and
// images are exactly the chain's own prefix there: the caller keeps a shared-prefix pin alive
// across a borrow, and its state goes to the chain's bounded sidecar, never to a second copy of
// the base's archive.  A non-pinned pre-base checkpoint is still refused.
std::vector<ConversationCheckpoint> chain_points(const std::vector<int32_t>& full_ids,
                                                 const std::vector<ConversationImageKey>& full_imgs,
                                                 const SavedConversation& stage, uint64_t base_tokens,
                                                 uint64_t end_tokens, std::string& error) {
    std::vector<ConversationCheckpoint> points;
    points.reserve(1 + stage.checkpoints.size());
    ConversationCheckpoint final_point;
    final_point.ids = full_ids;
    final_point.imgs = full_imgs;
    final_point.used = stage.live.used;
    final_point.pinned = stage.live.pinned;
    points.push_back(std::move(final_point));
    for (const auto& c : stage.checkpoints) {
        if (c.ids.empty() || (uint64_t) c.ids.size() > end_tokens ||
            !std::equal(c.ids.begin(), c.ids.end(), points.front().ids.begin())) {
            error = "an invalid checkpoint in the delta";
            return {};
        }
        if ((uint64_t) c.ids.size() <= base_tokens) {
            size_t j = 0;
            bool same_imgs = c.pinned;
            for (const auto& image : points.front().imgs) {
                if ((uint64_t) image.start >= c.ids.size()) break;
                if (j == c.imgs.size() || !(c.imgs[j++] == image)) {
                    same_imgs = false;
                    break;
                }
            }
            if (!same_imgs || j != c.imgs.size()) {
                error = "an invalid checkpoint in the delta";
                return {};
            }
        }
        ConversationCheckpoint point;
        point.ids = c.ids;
        point.imgs = c.imgs;
        point.used = c.used;
        point.pinned = c.pinned;
        points.push_back(std::move(point));
    }
    return points;
}

ConversationDiskStatus ConversationDiskStore::spill(SavedConversation&& image, bool root, std::string& error,
                                                    uint64_t* written_tokens, size_t* written_bytes) {
    if (written_tokens != nullptr) *written_tokens = 0;
    if (written_bytes != nullptr) *written_bytes = 0;
    if (!enabled()) return ConversationDiskStatus::disabled;
    const uint64_t end = (uint64_t) image.live.ids.size();
    if (end == 0) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    std::vector<const SavedConversation*> stage_views;
    std::vector<StageShape> shape;
    stage_views.reserve(image.stage_images.size() + 1);
    shape.reserve(image.stage_images.size() + 1);
    stage_views.push_back(&image);
    shape.push_back(stage_shape_of(image));
    for (const auto& part : image.stage_images) {
        stage_views.push_back(&part);
        shape.push_back(stage_shape_of(part));
    }
    auto available_name = [&](std::string name) {
        if (has(name)) {
            // Equal token IDs can have different K/V arithmetic. Keep the
            // existing lineage and its dependents instead of claiming a hit.
            const std::string stem = std::move(name);
            uint64_t variant = 0;
            do {
                name = stem + "-" + std::to_string(++variant);
            } while (has(name));
        }
        return name;
    };
    // Which stored chain this image can lean on: the longest common token prefix it shares with the
    // chain's content.  A chain that already CONTAINS the image needs no write; a chain whose own
    // content is the image's prefix takes the delta as an append; any other common prefix becomes
    // the base of a branch chain, so the image stores only its divergent suffix instead of a second
    // copy of the prefix.  Every case is byte-verified below: equal token ids never authorise reuse.
    struct Candidate {
        const Entry* entry = nullptr;
        uint64_t common = 0;
        int priority = 2;      // 0: already stored, 1: append, 2: branch
    };
    std::vector<Candidate> candidates;
    for (const auto& entry : entries_) {
        if (entry.root != root || entry.cvec != image.cvec || entry.shape != shape) continue;
        if (entry.points.empty()) continue;
        const uint64_t common = content_common_prefix(entry.points.front(), image.live.ids,
                                                      image.live.imgs);
        if (common == 0) continue;
        Candidate candidate;
        candidate.entry = &entry;
        candidate.common = common;
        candidate.priority = common == end ? 0 : (common == entry.cover ? 1 : 2);
        candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        if (a.common != b.common) return a.common > b.common;
        return a.entry->stamp > b.entry->stamp;
    });
    for (const Candidate& candidate : candidates) {
        const Entry& entry = *candidate.entry;
        uint64_t common = candidate.common;
        // The byte proof: the chain's stored K/V for [0, common) must equal the image's, every
        // retained page and indexer row.  Reading it costs the common extent, never the chain's own.
        ConversationDiskRecord stored;
        std::string check_error;
        if (get_prefix(entry.name, stored, common, check_error) != ConversationDiskStatus::ok) continue;
        std::vector<const SavedConversation*> stored_views;
        stored_views.reserve(stored.stages.size());
        for (const auto& stage : stored.stages) stored_views.push_back(&stage);
        if (!kv_prefix_equal(stored_views, stage_views, common, check_error)) {
            std::vector<uint64_t> boundaries;
            auto add = [&](uint64_t tokens) {
                if (tokens > 0 && tokens < common) boundaries.push_back(tokens);
            };
            add(entry.terminal_from);
            add((uint64_t) std::max<int64_t>(0, image.terminal_from));
            for (const auto& point : entry.points) add(point.ids.size());
            for (const auto& point : image.checkpoints) add(point.ids.size());
            std::sort(boundaries.begin(), boundaries.end(), std::greater<uint64_t>());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            bool verified = false;
            for (const uint64_t boundary : boundaries)
                if (kv_prefix_equal(stored_views, stage_views, boundary, check_error)) {
                    common = boundary;
                    verified = true;
                    break;
                }
            if (!verified) continue;
        }
        const int priority = common == end ? 0 : (common == entry.cover ? 1 : 2);
        if (priority == 0) return ConversationDiskStatus::ok;   // the chain holds the image
        // The name and the borrowed prefix are read before the image is moved and sliced.
        std::string name;
        if (priority == 2)
            name = available_name(root
                ? conversation_disk_prefix_name(image.live.ids, image.live.imgs, image.cvec)
                : conversation_disk_name(image.live.ids, image.live.imgs, image.cvec));
        ConversationDiskBase borrowed;
        if (priority == 2) {
            borrowed.name = entry.name;
            borrowed.tokens = common;
            borrowed.ids.assign(image.live.ids.begin(),
                                image.live.ids.begin() + (std::ptrdiff_t) common);
            for (const auto& key : image.live.imgs)
                if ((uint64_t) key.start < common) borrowed.imgs.push_back(key);
        }
        ConversationDiskRecord record;
        record.stages.reserve(image.stage_images.size() + 1);
        auto parts = std::move(image.stage_images);
        image.stage_images.clear();
        record.stages.push_back(std::move(image));
        for (auto& part : parts) record.stages.push_back(std::move(part));
        for (auto& stage : record.stages)
            if (!slice_stage_from(stage, common, error)) return ConversationDiskStatus::invalid;
        const size_t bytes = (size_t) record_bytes(record, metadata_bytes(record));
        // The chain's name is copied: an eviction inside append() may move the index entry this
        // reference points into.
        const std::string chain = entry.name;
        ConversationDiskStatus status = ConversationDiskStatus::ok;
        if (priority == 1) {
            status = append(chain, std::move(record), error);
        } else {
            // The image diverges inside the chain's terminal reply: replace that tail in place, so a
            // rewritten reply keeps ONE chain and one bounded set of recurrent states per
            // conversation instead of a fresh full sidecar per turn.  The store refuses when the
            // chain's own terminal boundary, its pins, its retained anchors or a borrowed extent do
            // not prove the replacement safe - and then the record is still intact for the genuine
            // branch below, which keeps the source's continuation and shares the prefix by reference.
            status = replace_terminal_tail(chain, common, std::move(record), error);
            if (status == ConversationDiskStatus::invalid) {
                error.clear();
                status = seed(name, std::move(record), borrowed, error);
            }
        }
        if (status == ConversationDiskStatus::ok) {
            if (written_tokens != nullptr) *written_tokens = end - common;
            if (written_bytes != nullptr) *written_bytes = bytes;
        }
        return status;
    }
    // Nothing to lean on: publish the image self-contained.
    const std::string name = available_name(root
        ? conversation_disk_prefix_name(image.live.ids, image.live.imgs, image.cvec)
        : conversation_disk_name(image.live.ids, image.live.imgs, image.cvec));
    ConversationDiskRecord record;
    record.stages.reserve(image.stage_images.size() + 1);
    auto parts = std::move(image.stage_images);
    image.stage_images.clear();
    record.stages.push_back(std::move(image));
    for (auto& part : parts) record.stages.push_back(std::move(part));
    const size_t bytes = (size_t) record_bytes(record, metadata_bytes(record));
    const ConversationDiskStatus status = seed(name, std::move(record), std::string(), 0, error);
    if (status == ConversationDiskStatus::ok) {
        if (written_tokens != nullptr) *written_tokens = end;
        if (written_bytes != nullptr) *written_bytes = bytes;
    }
    return status;
}

ConversationDiskStatus ConversationDiskStore::seed(const std::string& name, ConversationDiskRecord&& record,
                                                   std::string base_name, uint64_t base_tokens,
                                                   std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (base_name.empty()) {
        if (base_tokens != 0) {
            error = "base tokens without a base chain";
            return ConversationDiskStatus::invalid;
        }
        return seed(name, std::move(record), ConversationDiskBase{}, error);
    }
    if (!valid_name(base_name) || base_tokens == 0) {
        error = "an invalid base chain";
        return ConversationDiskStatus::invalid;
    }
    const Entry* base = find(base_name);
    if (base == nullptr) {
        error = "the base chain is not indexed";
        return ConversationDiskStatus::invalid;
    }
    if (base->points.empty() || (uint64_t) base->points.front().ids.size() < base_tokens) {
        error = "the base chain does not cover base_tokens";
        return ConversationDiskStatus::invalid;
    }
    ConversationDiskBase borrowed;
    borrowed.name = base_name;
    borrowed.tokens = base_tokens;
    borrowed.ids.assign(base->points.front().ids.begin(),
                        base->points.front().ids.begin() + (std::ptrdiff_t) base_tokens);
    for (const auto& image : base->points.front().imgs) {
        if ((uint64_t) image.start >= base_tokens) continue;
        borrowed.imgs.push_back(image);
    }
    return seed(name, std::move(record), borrowed, error);
}

ConversationDiskStatus ConversationDiskStore::seed(const std::string& name, ConversationDiskRecord&& record,
                                                   const ConversationDiskBase& base, std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    if (record.stages.empty()) {
        error = "the record has no stage";
        return ConversationDiskStatus::invalid;
    }
    const bool cvec = record.stages.front().cvec;
    for (const auto& stage : record.stages)
        if (stage.cvec != cvec) {
            error = "the stages disagree on the cvec flag";
            return ConversationDiskStatus::invalid;
        }
    const uint64_t slice = (uint64_t) record.stages.front().live.ids.size();
    if (slice == 0) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    for (const auto& stage : record.stages)
        if ((uint64_t) stage.live.ids.size() != slice) {
            error = "the stages disagree on the token chain";
            return ConversationDiskStatus::invalid;
        }
    std::vector<StageShape> shape;
    if (!stage_shapes(record, shape, error)) return ConversationDiskStatus::invalid;
    const bool has_base = !base.name.empty();
    const Entry* base_entry = nullptr;
    uint64_t base_tokens = 0;
    if (has_base) {
        if (base.tokens == 0 || !valid_name(base.name) || base.name.size() >= sizeof(ChainHeader::base_name)) {
            error = "an invalid base chain";
            return ConversationDiskStatus::invalid;
        }
        base_entry = find(base.name);
        if (base_entry == nullptr) {
            error = "the base chain is not indexed";
            return ConversationDiskStatus::invalid;
        }
        // Collapse a chain of branches: the base to store is the highest ancestor that still covers
        // the whole borrowed prefix, so a run of branches on one conversation does not nest one
        // level deeper per branch.  Every ancestor on the way holds the same prefix bytes.
        {
            const Entry* chosen = base_entry;
            for (size_t steps = 0; steps < max_base_depth && !chosen->base.empty(); ++steps) {
                const Entry* parent = find(chosen->base);
                if (parent == nullptr || chosen->base_tokens < base.tokens || parent->cover < base.tokens)
                    break;
                chosen = parent;
            }
            base_entry = chosen;
        }
        // The graph must stay acyclic: a chain may not be its own ancestor.
        {
            const Entry* walk = base_entry;
            size_t steps = 0;
            while (walk != nullptr && steps <= max_base_depth) {
                if (walk->name == name) {
                    error = "a chain cannot be its own base";
                    return ConversationDiskStatus::invalid;
                }
                walk = walk->base.empty() ? nullptr : find(walk->base);
                ++steps;
            }
        }
        if (base_entry->points.empty() || base_entry->cover < base.tokens ||
            (uint64_t) base_entry->points.front().ids.size() < base.tokens) {
            error = "the base chain does not cover the borrowed prefix";
            return ConversationDiskStatus::invalid;
        }
        if (base_entry->cvec != cvec) {
            error = "the base chain's cvec flag does not match the record";
            return ConversationDiskStatus::invalid;
        }
        if (base_entry->depth + 1 > max_base_depth) {
            error = "the branch chain nesting is too deep";
            return ConversationDiskStatus::invalid;
        }
        if ((uint64_t) base.ids.size() != base.tokens) {
            error = "the borrowed prefix does not match its token count";
            return ConversationDiskStatus::invalid;
        }
        const ConversationCheckpoint& point = base_entry->points.front();
        if (!std::equal(base.ids.begin(), base.ids.end(), point.ids.begin())) {
            error = "the borrowed prefix does not match the base chain";
            return ConversationDiskStatus::invalid;
        }
        size_t j = 0;
        for (const auto& image : point.imgs) {
            if ((uint64_t) image.start >= base.tokens) continue;
            if (j == base.imgs.size() || !(base.imgs[j++] == image)) {
                error = "the borrowed prefix images do not match the base chain";
                return ConversationDiskStatus::invalid;
            }
        }
        if (j != base.imgs.size()) {
            error = "the borrowed prefix images do not match the base chain";
            return ConversationDiskStatus::invalid;
        }
        if (base_entry->shape != shape) {
            error = "the base chain's stage layout does not match the record";
            return ConversationDiskStatus::invalid;
        }
        base_tokens = base.tokens;
    }
    // The chain that actually stores the borrowed prefix (after the collapse).
    const std::string storage_base = base_entry != nullptr ? base_entry->name : std::string();
    const uint64_t end = base_tokens + slice;
    if (end < base_tokens) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    // The full chain ids for checkpoint validation (and the index points).
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> all_imgs;
    if (has_base) {
        ids = base.ids;
        all_imgs = base.imgs;
    }
    ids.insert(ids.end(), record.stages.front().live.ids.begin(), record.stages.front().live.ids.end());
    all_imgs.insert(all_imgs.end(), record.stages.front().live.imgs.begin(),
                    record.stages.front().live.imgs.end());
    const std::vector<ConversationCheckpoint> points =
        chain_points(ids, all_imgs, record.stages.front(), base_tokens, end, error);
    if (points.empty()) return ConversationDiskStatus::invalid;
    // The panel stores no state of its own: the sidecar takes every one of them below.  Which of
    // these points a restore can actually resume from is settled from the sidecar's slots after the
    // write, because the retention rotates them (a point the policy dropped must not stay offered).
    std::vector<bool> inline_states(points.size(), false);
    // The kind marks the base chain's class: a 'p' system-prompt root keeps the kind an older
    // build already understands, a 'c' conversation base is kind 2 (an older build refuses it as
    // corrupt instead of misreading it).  Both resolve the same way.
    const uint8_t base_kind = !has_base ? kBaseNone
                             : (base_entry->name[0] == 'p' ? kBaseRoot : kBaseChain);
    const size_t base_depth = base_entry != nullptr ? base_entry->depth + 1 : 0;
    // A re-seed replaces the chain with the same name; its references transfer to the new chain.
    // The old sidecar describes the content being replaced, so it goes BEFORE the new anchors are
    // published below - state_store creates the new sidecar under the same name.
    size_t old_refs = 0;
    if (Entry* old = find(name)) {
        if (borrowed_prefix(name) > end) {
            error = "the chain is the base of a longer branch chain";
            return ConversationDiskStatus::invalid;
        }
        state_discard(name, 0, true);
        old_refs = old->refs;
        if (!old->base.empty())
            if (Entry* b = find(old->base)) --b->refs;
        std::error_code ignored;
        std::filesystem::remove(path_for(name), ignored);
        bytes_ -= old->bytes;
        entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(old - entries_.data()));
    }
    // The running states and their positions go to the chain's sidecar: the panel then stores
    // neither a state nor a repeated checkpoint position, so its metadata is the delta alone.
    size_t state_bytes = 0;
    if (!state_store(name, record, end, ids, all_imgs, &state_bytes, error)) {
        state_discard(name, 0, true);
        return ConversationDiskStatus::failed;
    }
    for (auto& stage : record.stages) stage.checkpoints.clear();
    MetaWriter metadata;
    metadata.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) metadata.write_stage(stage);
    uint64_t payload_bytes = 0;
    for (const uint64_t bytes : metadata.lengths)
        if (!add_overflow(payload_bytes, bytes)) {
            error = "the record size overflows";
            return ConversationDiskStatus::invalid;
        }
    uint64_t record_bytes = kChainHeaderBytes + kPanelHeaderBytes;
    if (!add_overflow(record_bytes, metadata.data.size()) || !add_overflow(record_bytes, payload_bytes)) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    if (record_bytes > options_.budget_bytes) {
        error = "the record exceeds the byte budget";
        return ConversationDiskStatus::invalid;
    }
    const std::string keep = storage_base;
    const ConversationDiskStatus room = make_room(record_bytes, name, keep, true, error);
    if (room != ConversationDiskStatus::ok) return room;
    // Publish atomically through a temporary file, like the old full-snapshot records.
    ChainHeader header;
    header.magic = kMagic;
    header.version = format_version;
    header.total_tokens = end;
    header.panel_count = 1;
    std::memcpy(header.identity, options_.identity.bytes.data(), ConversationDiskIdentity::size);
    header.cvec = cvec ? 1 : 0;
    header.base_kind = base_kind;
    header.base_tokens = base_tokens;
    std::snprintf(header.base_name, sizeof(header.base_name), "%s", storage_base.c_str());
    header.base_prefix_checksum = has_base ? prefix_checksum(base.ids, base.imgs) : 0;
    const std::filesystem::path target = path_for(name);
    const std::filesystem::path temporary = target.string() + ".tmp";
    std::error_code error_code;
    std::filesystem::remove(temporary, error_code);
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        error = "cannot create " + temporary.string();
        return ConversationDiskStatus::failed;
    }
    bool ok = std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
    if (ok) {
        PanelHeader panel;
        panel.magic = kPanelMagic;
        panel.base_tokens = base_tokens;
        panel.end_tokens = end;
        panel.metadata_bytes = metadata.data.size();
        panel.metadata_checksum = conversation_disk_checksum(metadata.data.data(), metadata.data.size());
        panel.payload_bytes = payload_bytes;
        ok = std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
             (metadata.data.empty() ||
              std::fwrite(metadata.data.data(), 1, metadata.data.size(), file) == metadata.data.size());
        if (ok) {
            uint64_t checksum = conversation_disk_checksum_seed;
            auto write_section = [&](Section section) {
                if (!ok) return;
                if (section.bytes != nullptr) {
                    if (!section.bytes->empty() &&
                        std::fwrite(section.bytes->data(), 1, section.bytes->size(), file) != section.bytes->size())
                        ok = false;
                    checksum = conversation_disk_checksum(section.bytes->data(), section.bytes->size(), checksum);
                } else {
                    section.buffer->visit(0, section.buffer->size(),
                                          [&](const uint8_t* data, size_t bytes, size_t) {
                                              if (ok && std::fwrite(data, 1, bytes, file) != bytes) ok = false;
                                              checksum = conversation_disk_checksum(data, bytes, checksum);
                                              return ok;
                                          });
                }
            };
            visit_payload(record, write_section);
            if (ok) {
                panel.payload_checksum = checksum;
                uint64_t total = sizeof(panel) + metadata.data.size() + payload_bytes;
                ok = std::fseek(file, -static_cast<long>(total), SEEK_CUR) == 0 &&
                     std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
                     std::fseek(file, static_cast<long>(total), SEEK_CUR) == 0;
            }
        }
    }
    if (std::fclose(file) != 0) ok = false;
    if (ok) {
        std::filesystem::rename(temporary, target, error_code);
        if (error_code) ok = false;
    }
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "cannot write " + name;
        return ConversationDiskStatus::failed;
    }
    Entry entry;
    entry.name = name;
    entry.cover = end;
    entry.stamp = ++stamp_;
    entry.cvec = cvec;
    entry.root = !name.empty() && name[0] == 'p';
    entry.refs = old_refs;
    entry.base = storage_base;
    entry.base_tokens = base_tokens;
    entry.depth = base_depth;
    entry.bytes = std::filesystem::file_size(target) + state_file_bytes(name);
    entry.shape = std::move(shape);
    entry.points = points;
    entry.terminal_from = (uint64_t) std::max<int64_t>(0, record.stages.front().terminal_from);
    bytes_ += entry.bytes;
    entries_.push_back(std::move(entry));
    settle_entry_states(entries_.back(), std::move(inline_states));
    if (has_base)
        if (Entry* base_found = find(storage_base)) ++base_found->refs;
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::append(const std::string& name, ConversationDiskRecord&& record,
                                                     std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    if (record.stages.empty() || record.stages.front().live.ids.empty()) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    if (record.stages.front().cvec != entry->cvec) {
        error = "the record's cvec flag does not match the chain";
        return ConversationDiskStatus::invalid;
    }
    const uint64_t slice = (uint64_t) record.stages.front().live.ids.size();
    for (const auto& stage : record.stages)
        if ((uint64_t) stage.live.ids.size() != slice) {
            error = "the stages disagree on the token chain";
            return ConversationDiskStatus::invalid;
        }
    {
        std::vector<StageShape> shape;
        if (!stage_shapes(record, shape, error)) return ConversationDiskStatus::invalid;
        if (shape != entry->shape) {
            error = "the record's stage layout does not match the chain";
            return ConversationDiskStatus::invalid;
        }
    }
    const uint64_t base = entry->rewrite_from == UINT64_MAX ? entry->cover : entry->rewrite_from;
    const uint64_t end = base + slice;
    if (end < base) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    // The chain's whole content at `end`: the delta's own slice plus the prefix it continues.  The
    // checkpoints are validated against it and the live anchor's content fingerprint is taken from
    // it (the delta's ids alone are only the slice).
    std::vector<ConversationCheckpoint> points;
    std::vector<bool> inline_states;
    std::vector<int32_t> content_ids;
    std::vector<ConversationImageKey> content_imgs;
    {
        content_ids = entry->points.front().ids;
        content_ids.resize((size_t) base);
        content_ids.insert(content_ids.end(), record.stages.front().live.ids.begin(),
                           record.stages.front().live.ids.end());
        content_imgs = entry->points.front().imgs;
        content_imgs.erase(std::remove_if(content_imgs.begin(), content_imgs.end(),
            [&](const ConversationImageKey& image) { return image.start >= (int64_t) base; }), content_imgs.end());
        content_imgs.insert(content_imgs.end(), record.stages.front().live.imgs.begin(),
                            record.stages.front().live.imgs.end());
        points = chain_points(content_ids, content_imgs, record.stages.front(), base, end, error);
        if (points.empty()) return ConversationDiskStatus::invalid;
        // The delta's own points carry no inline state: the sidecar takes the delta's states below
        // and the panel writes no checkpoint at all.  A retained old point keeps whatever its own
        // panel holds inline; its sidecar slot, if any, is settled after the write.
        inline_states.assign(points.size(), false);
        for (size_t i = 1; i < entry->points.size(); ++i) {
            if (entry->points[i].ids.size() > base) continue;
            points.push_back(entry->points[i]);
            inline_states.push_back(i < entry->point_inline_states.size()
                                        ? entry->point_inline_states[i]
                                        : false);
        }
    }
    // The running states and their positions go to the sidecar; the panel stores the K/V delta only.
    size_t state_bytes = 0;
    if (!state_store(name, record, end, content_ids, content_imgs, &state_bytes, error))
        return ConversationDiskStatus::failed;
    for (auto& stage : record.stages) stage.checkpoints.clear();
    MetaWriter metadata;
    metadata.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) metadata.write_stage(stage);
    uint64_t payload_bytes = 0;
    for (const uint64_t bytes : metadata.lengths)
        if (!add_overflow(payload_bytes, bytes)) {
            error = "the record size overflows";
            return ConversationDiskStatus::invalid;
        }
    uint64_t panel_bytes = kPanelHeaderBytes;
    if (!add_overflow(panel_bytes, metadata.data.size()) || !add_overflow(panel_bytes, payload_bytes)) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    const ConversationDiskStatus room = make_room(panel_bytes, name, entry->base, false, error);
    if (room != ConversationDiskStatus::ok) return room;
    entry = find(name); // eviction can move entries in the index
    if (entry == nullptr) return ConversationDiskStatus::miss;
    const std::filesystem::path target = path_for(name);
    // "r+b": appends must be able to seek back and patch the panel/chain headers in place (an
    // append-mode stream would force EVERY write to the end of the file).
    std::FILE* file = std::fopen(target.string().c_str(), "r+b");
    if (file == nullptr) {
        error = "cannot open " + target.string();
        return ConversationDiskStatus::failed;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        error = "cannot seek " + target.string();
        return ConversationDiskStatus::failed;
    }
    const uint64_t start_bytes = std::filesystem::file_size(target);
    bool ok = true;
    {
        PanelHeader panel;
        panel.magic = kPanelMagic;
        panel.base_tokens = base;
        panel.end_tokens = end;
        panel.metadata_bytes = metadata.data.size();
        panel.metadata_checksum = conversation_disk_checksum(metadata.data.data(), metadata.data.size());
        panel.payload_bytes = payload_bytes;
        ok = std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
             (metadata.data.empty() ||
              std::fwrite(metadata.data.data(), 1, metadata.data.size(), file) == metadata.data.size());
        if (ok) {
            uint64_t checksum = conversation_disk_checksum_seed;
            auto write_section = [&](Section section) {
                if (!ok) return;
                if (section.bytes != nullptr) {
                    if (!section.bytes->empty() &&
                        std::fwrite(section.bytes->data(), 1, section.bytes->size(), file) != section.bytes->size())
                        ok = false;
                    checksum = conversation_disk_checksum(section.bytes->data(), section.bytes->size(), checksum);
                } else {
                    section.buffer->visit(0, section.buffer->size(),
                                          [&](const uint8_t* data, size_t bytes, size_t) {
                                              if (ok && std::fwrite(data, 1, bytes, file) != bytes) ok = false;
                                              checksum = conversation_disk_checksum(data, bytes, checksum);
                                              return ok;
                                          });
                }
            };
            visit_payload(record, write_section);
            if (ok) {
                panel.payload_checksum = checksum;
                uint64_t total = sizeof(panel) + metadata.data.size() + payload_bytes;
                ok = std::fseek(file, -static_cast<long>(total), SEEK_CUR) == 0 &&
                     std::fwrite(&panel, 1, sizeof(panel), file) == sizeof(panel) &&
                     std::fseek(file, static_cast<long>(total), SEEK_CUR) == 0;
            }
        }
    }
    if (ok) {
        // Patch the chain header: one more panel, the new coverage.
        ChainHeader header;
        ok = std::fseek(file, 0, SEEK_SET) == 0 &&
             std::fread(&header, 1, sizeof(header), file) == sizeof(header) &&
             header.magic == kMagic && header.version == format_version;
        if (ok) {
            header.panel_count += 1;
            header.total_tokens = end;
            ok = std::fseek(file, 0, SEEK_SET) == 0 &&
                 std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
        }
    }
    if (std::fclose(file) != 0) ok = false;
    if (!ok) {
        // Roll the torn tail back so the chain stays at its previous coverage.
        std::error_code ignored;
        std::filesystem::resize_file(target, start_bytes, ignored);
        return ConversationDiskStatus::failed;
    }
    const uint64_t new_bytes = std::filesystem::file_size(target) + state_file_bytes(name);
    bytes_ += new_bytes - entry->bytes;
    entry->bytes = new_bytes;
    entry->cover = end;
    entry->rewrite_from = UINT64_MAX;
    entry->points = std::move(points);
    entry->terminal_from = (uint64_t) std::max<int64_t>(0, record.stages.front().terminal_from);
    settle_entry_states(*entry, std::move(inline_states));
    return ConversationDiskStatus::ok;
}

bool ConversationDiskStore::cover(const std::string& name, uint64_t& tokens) const {
    const Entry* entry = find(name);
    if (entry == nullptr) return false;
    tokens = entry->cover;
    return true;
}

ConversationDiskStatus ConversationDiskStore::remove(const std::string& name, std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    std::error_code error_code;
    std::filesystem::remove(path_for(name), error_code);
    state_discard(name, 0, true);
    if (error_code) {
        error = "cannot remove " + path_for(name).string();
        return ConversationDiskStatus::failed;
    }
    if (!entry->base.empty())
        if (Entry* base = find(entry->base)) --base->refs;
    bytes_ -= entry->bytes;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(entry - entries_.data()));
    // A branch chain cannot load without the chain it borrows from: remove it with the base.
    for (size_t i = 0; i < entries_.size();) {
        if (entries_[i].base != name) {
            ++i;
            continue;
        }
        const std::string child = entries_[i].name;
        std::string ignored;
        if (remove(child, ignored) != ConversationDiskStatus::ok) ++i;
    }
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::rewind_chain(const std::string& name, uint64_t at,
                                                           uint64_t& new_cover, std::string& error) {
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    // Keep the existing payload intact and replace only the tail on the next
    // append when `at` is a saved checkpoint inside a panel. A failed append
    // leaves the previous valid record available.
    if (at < entry->cover && std::any_of(entry->points.begin() + 1, entry->points.end(),
            [&](const ConversationCheckpoint& point) { return point.ids.size() == at; })) {
        // A branch chain borrows this chain's prefix: never cut below the borrowed length.
        if (borrowed_prefix(name) > at) {
            error = "the chain is the base of a longer branch chain";
            return ConversationDiskStatus::invalid;
        }
        state_discard(name, at, false);     // the anchors above the cut describe the replaced tail
        // The dropped slots are gone, so the index must stop offering their states before the
        // append that replaces the tail lands (and stays correct if that append never comes).
        settle_entry_states(*entry, entry->point_inline_states);
        entry->rewrite_from = at;
        new_cover = at;
        return ConversationDiskStatus::ok;
    }
    if (at >= entry->cover) {
        new_cover = entry->cover;
        return ConversationDiskStatus::ok;
    }
    const std::filesystem::path target = path_for(name);
    std::FILE* file = std::fopen(target.string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + target.string();
        return ConversationDiskStatus::failed;
    }
    ChainHeader header;
    if (!read_exact(file, &header, sizeof(header))) {
        std::fclose(file);
        error = "corrupt conversation record: header";
        return ConversationDiskStatus::corrupt;
    }
    bool ok = header.magic == kMagic && header.version == format_version && header.base_kind <= kBaseChain;
    uint64_t pos = kChainHeaderBytes;
    uint64_t kept = 0;
    uint64_t kept_end = header.base_kind != kBaseNone ? header.base_tokens : 0;
    for (uint64_t k = 0; ok && k < header.panel_count; ++k) {
        PanelHeader p{};
        if (std::fseek(file, static_cast<long>(pos), SEEK_SET) != 0 ||
            !read_exact(file, &p, sizeof(p)) || p.magic != kPanelMagic ||
            p.base_tokens > kept_end) {
            ok = false;
            break;
        }
        if (p.end_tokens > at) break;               // drop this panel and everything after it
        kept = k + 1;
        kept_end = p.end_tokens;
        pos += sizeof(p) + p.metadata_bytes + p.payload_bytes;
    }
    std::fclose(file);
    if (!ok) {
        error = "corrupt conversation record: panels";
        return ConversationDiskStatus::corrupt;
    }
    if (kept == header.panel_count) {
        new_cover = kept_end;
        return ConversationDiskStatus::ok;
    }
    if (kept == 0) {
        // A header without a panel is not a valid record. Leave the old chain
        // intact and let the caller seed the shorter branch as a new record.
        new_cover = 0;
        return ConversationDiskStatus::miss;
    }
    // The retained panels still cover every borrowed prefix, or the cut would strand a branch.
    if (borrowed_prefix(name) > kept_end) {
        error = "the chain is the base of a longer branch chain";
        return ConversationDiskStatus::invalid;
    }
    std::error_code ignored;
    std::filesystem::resize_file(target, pos, ignored);
    file = std::fopen(target.string().c_str(), "r+b");
    if (file == nullptr) {
        error = "cannot reopen " + target.string();
        return ConversationDiskStatus::failed;
    }
    ok = std::fseek(file, 0, SEEK_SET) == 0 && std::fread(&header, 1, sizeof(header), file) == sizeof(header);
    if (ok) {
        header.panel_count = kept;
        header.total_tokens = kept_end;
        ok = std::fseek(file, 0, SEEK_SET) == 0 && std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
    }
    std::fclose(file);
    if (!ok) {
        error = "cannot patch " + target.string();
        return ConversationDiskStatus::failed;
    }
    const uint64_t new_bytes = std::filesystem::file_size(target) + state_file_bytes(name);
    bytes_ += new_bytes - entry->bytes;
    entry->bytes = new_bytes;
    entry->cover = kept_end;
    // The anchors above the cut are gone with the panels; rebuild the points from what is left.
    state_discard(name, kept_end, false);
    Entry rebuilt;
    const ConversationDiskStatus status = index_file(target, entry->name, rebuilt, error);
    if (status != ConversationDiskStatus::ok) return status;
    entry->points = std::move(rebuilt.points);
    entry->point_states = std::move(rebuilt.point_states);
    entry->point_inline_states = std::move(rebuilt.point_inline_states);
    entry->terminal_from = rebuilt.terminal_from;
    new_cover = kept_end;
    return ConversationDiskStatus::ok;
}

// A proven terminal-reply replacement.  The caller has byte-verified [0, from) against the chain and
// captured `delta` for [from, end); this call re-uses the chain file, so a rewritten terminal reply
// neither adds a chain nor a full recurrent state per turn.  The checks below are the store's own -
// the caller's word alone never authorises dropping a long tail.
ConversationDiskStatus ConversationDiskStore::replace_terminal_tail(const std::string& name, uint64_t from,
                                                                  ConversationDiskRecord&& delta,
                                                                  std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    // The name is copied: an eviction inside append() may move the index entry a caller's reference
    // points into, and this call uses the name after that append.
    const std::string chain = name;
    // Validate the delta without consuming it: on `invalid` the caller still owns it and seeds a
    // genuine branch chain from it.
    if (delta.stages.empty() || delta.stages.front().live.ids.empty()) {
        error = "the record has no live token slice";
        return ConversationDiskStatus::invalid;
    }
    if (delta.stages.front().cvec != entry->cvec) {
        error = "the record's cvec flag does not match the chain";
        return ConversationDiskStatus::invalid;
    }
    const uint64_t slice = (uint64_t) delta.stages.front().live.ids.size();
    for (const auto& stage : delta.stages)
        if ((uint64_t) stage.live.ids.size() != slice) {
            error = "the stages disagree on the token chain";
            return ConversationDiskStatus::invalid;
        }
    {
        std::vector<StageShape> shape;
        if (!stage_shapes(delta, shape, error)) return ConversationDiskStatus::invalid;
        if (shape != entry->shape) {
            error = "the record's stage layout does not match the chain";
            return ConversationDiskStatus::invalid;
        }
    }
    // The delta's terminal boundary is an absolute position in the chain it continues, so it may sit
    // past `from` (a rewritten reply usually grows its input) but never past the new content's end.
    if (delta.stages.front().terminal_from < 0 ||
        (uint64_t) delta.stages.front().terminal_from > from + slice) {
        error = "the record's terminal boundary is past its own content";
        return ConversationDiskStatus::invalid;
    }
    if (from == 0) {
        error = "the replacement has no earlier prefix";
        return ConversationDiskStatus::invalid;
    }
    if (from > entry->cover) {
        error = "the replacement starts past the chain";
        return ConversationDiskStatus::invalid;
    }
    if (from == entry->cover)
        return append(name, std::move(delta), error);   // nothing to replace: the delta continues it
    // The original input must be known and unchanged. Replay may change the
    // last input cells as well as the reply, but a historical edit must branch.
    if (entry->terminal_from == 0) {
        error = "the chain has no known input boundary";
        return ConversationDiskStatus::invalid;
    }
    if (entry->terminal_from > from) {
        const uint64_t input_end = entry->terminal_from;
        const auto& old = entry->points.front();
        const auto& incoming = delta.stages.front().live;
        const auto old_images_begin = std::find_if(old.imgs.begin(), old.imgs.end(),
            [&](const auto& key) { return (uint64_t) key.start >= from; });
        const auto old_images_end = std::find_if(old_images_begin, old.imgs.end(),
            [&](const auto& key) { return (uint64_t) key.start >= input_end; });
        const auto new_images_end = std::find_if(incoming.imgs.begin(), incoming.imgs.end(),
            [&](const auto& key) { return (uint64_t) key.start >= input_end; });
        if (input_end > from + slice || input_end > old.ids.size() ||
            !std::equal(old.ids.begin() + (std::ptrdiff_t) from,
                        old.ids.begin() + (std::ptrdiff_t) input_end, incoming.ids.begin()) ||
            !std::equal(old_images_begin, old_images_end, incoming.imgs.begin(), new_images_end)) {
            error = "the replacement changes the stored input";
            return ConversationDiskStatus::invalid;
        }
    }
    // A retained restore anchor at or below the cut must survive the rewrite: a chain whose only
    // state is its live tail must never be cut down to a short prefix.
    bool grounded = false;
    for (size_t i = 0; i < entry->points.size(); ++i) {
        if (entry->points[i].ids.size() > from) continue;
        if (i < entry->point_states.size() && entry->point_states[i]) {
            grounded = true;
            break;
        }
    }
    if (!grounded) {
        error = "the chain has no earlier restore anchor for the replaced tail";
        return ConversationDiskStatus::invalid;
    }
    // A branch chain borrows this chain's prefix: never cut below the borrowed length.
    if (borrowed_prefix(name) > from) {
        error = "the chain is the base of a longer branch chain";
        return ConversationDiskStatus::invalid;
    }
    // A PINNED anchor is the caller's explicit retention contract: never drop one, wherever above
    // the cut it sits (a pinned live tail included).  The sidecar's slots carry the pins.
    StateFile states;
    std::string side_error;
    if (state_open(path_for(chain), options_.identity, entry->cvec, nullptr, 0, false, states, side_error)) {
        for (const SlotHeader& slot : states.slots)
            if (slot.magic == kSlotMagic && slot.tokens > from && (slot.flags & kSlotPinned) != 0) {
                error = "the chain holds a pinned anchor above the replaced tail";
                return ConversationDiskStatus::invalid;
            }
    }
    for (const auto& point : entry->points)
        if (point.pinned && (uint64_t) point.ids.size() > from) {
            error = "the chain holds a pinned anchor above the replaced tail";
            return ConversationDiskStatus::invalid;
        }
    // Commit: the anchors above the cut describe the superseded reply and go; the delta's own states
    // are published into the chain's sidecar by the append below.  A failure leaves the file at its
    // previous coverage and clears the scheduling, so the chain stays valid and appendable.
    state_discard(chain, from, false);
    entry->rewrite_from = from;
    const ConversationDiskStatus status = append(chain, std::move(delta), error);
    if (status != ConversationDiskStatus::ok) {
        if (Entry* current = find(chain)) {
            current->rewrite_from = UINT64_MAX;
            settle_entry_states(*current, current->point_inline_states);
        }
    }
    return status;
}

void ConversationDiskStore::bound_checkpoint_states(ConversationDiskRecord& record) const {
    if (options_.max_checkpoint_states == 0) return;
    // The same selection the RAM archive uses: the positions stay (a stripped checkpoint still
    // matches a prompt), the states outside the bound go. The caller promotes the state at the
    // restore point it asked for BEFORE this runs, so the bound never removes that state.
    for (auto& stage : record.stages) {
        stage.drop_checkpoint_states(options_.max_checkpoint_states);
        for (auto& image : stage.stage_images) image.drop_checkpoint_states(options_.max_checkpoint_states);
    }
}

// The anchor-state sidecar of a chain: <name>.states, beside <name>.conversation.
std::filesystem::path ConversationDiskStore::state_path_for(const std::string& name) const {
    return state_path_of_chain(path_for(name));
}

uint64_t ConversationDiskStore::state_file_bytes(const std::string& name) const {
    std::error_code error;
    const std::filesystem::path path = state_path_for(name);
    const uint64_t bytes = std::filesystem::is_regular_file(path, error)
        ? (uint64_t) std::filesystem::file_size(path, error)
        : 0;
    return error ? 0 : bytes;
}

// The ordinary anchor slots: the caller's state bound plus the protected earliest anchor.  Pinned
// anchors grow the file beyond this, because a pin is the caller's explicit retention contract.
size_t ConversationDiskStore::anchor_slots() const {
    return std::min<size_t>(kMaxStateSlots, ordinary_slots() + 1);
}
// The ordinary anchors the retention policy keeps: the caller's bound, or the minimum (the earliest
// anchor and the newest) when the caller asked for none.  A PINNED anchor is never a victim and may
// grow the file past this, because a pin is the caller's explicit retention contract.
size_t ConversationDiskStore::ordinary_slots() const {
    return options_.max_checkpoint_states != 0 ? options_.max_checkpoint_states : 1;
}

static bool state_open(const std::filesystem::path& chain_path, const ConversationDiskIdentity& identity,
                       bool cvec, const std::vector<std::array<uint64_t, 5>>* sizes, size_t slots,
                       bool create, StateFile& out, std::string& error) {
    const std::filesystem::path path = state_path_of_chain(chain_path);
    out.file = std::fopen(path.string().c_str(), "r+b");
    if (out.file == nullptr && create && sizes != nullptr) {
        // A fresh sidecar through a temporary file: a reader never sees a half-written header.
        std::FILE* fresh = std::fopen((path.string() + ".tmp").c_str(), "wb");
        if (fresh == nullptr) {
            error = "cannot create " + path.string();
            return false;
        }
        StateHeader header;
        header.magic = kStateMagic;
        header.version = kStateVersion;
        header.slot_count = static_cast<uint32_t>(slots);
        std::memcpy(header.identity, identity.bytes.data(), ConversationDiskIdentity::size);
        header.cvec = cvec ? 1 : 0;
        header.stage_count = static_cast<uint8_t>(sizes->size());
        uint64_t slot_bytes = 0;
        for (const auto& stage : *sizes)
            for (const uint64_t bytes : stage) slot_bytes += bytes;
        header.slot_bytes = slot_bytes;
        header.checksum = state_header_checksum(header);
        bool ok = std::fwrite(&header, 1, sizeof(header), fresh) == sizeof(header);
        for (const auto& stage : *sizes)
            for (const uint64_t bytes : stage)
                if (ok && std::fwrite(&bytes, 8, 1, fresh) != 1) ok = false;
        if (std::fclose(fresh) != 0) ok = false;
        if (!ok) {
            std::error_code ignored;
            std::filesystem::remove(path.string() + ".tmp", ignored);
            error = "cannot write " + path.string();
            return false;
        }
        std::error_code error_code;
        std::filesystem::rename(path.string() + ".tmp", path, error_code);
        if (error_code) {
            error = "cannot publish " + path.string();
            return false;
        }
        out.file = std::fopen(path.string().c_str(), "r+b");
    }
    if (out.file == nullptr) return false;   // absent: the caller keeps the legacy inline form
    if (!read_exact(out.file, &out.header, sizeof(out.header)) || out.header.magic != kStateMagic ||
        out.header.version != kStateVersion || out.header.checksum != state_header_checksum(out.header) ||
        std::memcmp(out.header.identity, identity.bytes.data(), ConversationDiskIdentity::size) != 0 ||
        out.header.stage_count == 0 || out.header.slot_count == 0 ||
        out.header.slot_count > kMaxStateSlots || out.header.cvec != (cvec ? 1 : 0) ||
        (sizes != nullptr && (size_t) out.header.stage_count != sizes->size()))
        return false;
    out.section_sizes.resize(static_cast<size_t>(out.header.stage_count) * 5);
    if (!read_exact(out.file, out.section_sizes.data(), out.section_sizes.size() * 8)) return false;
    uint64_t slot_bytes = 0;
    for (size_t i = 0; i < out.section_sizes.size(); ++i) {
        if (sizes != nullptr && out.section_sizes[i] != (*sizes)[i / 5][i % 5]) return false;
        slot_bytes += out.section_sizes[i];
    }
    if (slot_bytes != out.header.slot_bytes) return false;
    out.slots.assign(out.header.slot_count, SlotHeader{});
    for (size_t i = 0; i < out.slots.size(); ++i) {
        // A slot past the file's end was never written: the file grows one slot at a time, so a
        // fresh (or partially filled) sidecar is valid and its unwritten slots are simply empty.
        if (std::fseek(out.file, static_cast<long>(slot_offset(out.header, i)), SEEK_SET) != 0)
            return false;
        out.slots[i] = SlotHeader{};
        if (!read_exact(out.file, &out.slots[i], sizeof(SlotHeader)) ||
            out.slots[i].magic != kSlotMagic)
            out.slots[i] = SlotHeader{};   // never written, or a torn slot header
    }
    return true;
}

// The slot a new anchor takes: the one that already holds its position, else a free slot, else the
// least recently written slot the retention policy drops.  A PINNED slot is never a victim: when
// none can go the file grows by one slot, so a pin - the caller's explicit retention contract -
// never drops an anchor.  The policy's own bound (`ordinary`) keeps the growth finite: the ordinary
// anchors stay within `ordinary + 1` slots and the file never grows one state per turn.
bool state_slot_for(StateFile& file, uint64_t tokens, size_t ordinary, size_t& index,
                    std::string& error) {
    size_t same = SIZE_MAX, free_slot = SIZE_MAX;
    std::vector<uint64_t> positions;
    positions.reserve(file.slots.size() + 1);
    for (size_t i = 0; i < file.slots.size(); ++i) {
        const SlotHeader& slot = file.slots[i];
        if (slot.magic != kSlotMagic) {
            if (free_slot == SIZE_MAX) free_slot = i;
            continue;
        }
        if (slot.tokens == tokens) same = i;
        positions.push_back(slot.tokens);
    }
    if (same != SIZE_MAX) {
        index = same;
        return true;
    }
    if (free_slot != SIZE_MAX) {
        index = free_slot;
        return true;
    }
    positions.push_back(tokens);
    std::sort(positions.begin(), positions.end());
    // The chain's own terminal boundary, when it has one, keeps the historical anchors: the anchors
    // inside the terminal region describe a reply a rewrite replaces.
    const std::vector<bool> keep = anchor_keep(positions, ordinary, file.header.terminal_from,
                                              file.header.origin_tokens);
    size_t victim = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < file.slots.size(); ++i) {
        const SlotHeader& slot = file.slots[i];
        if (slot.magic != kSlotMagic || (slot.flags & kSlotPinned) != 0) continue;
        if (slot.tokens == tokens) continue;
        const size_t at = static_cast<size_t>(
            std::lower_bound(positions.begin(), positions.end(), slot.tokens) - positions.begin());
        if (at < keep.size() && keep[at]) continue;          // the policy keeps this anchor
        if (slot.generation < oldest) {
            oldest = slot.generation;
            victim = i;
        }
    }
    if (victim != SIZE_MAX) {
        index = victim;
        return true;
    }
    if (file.slots.size() >= kMaxStateSlots) {
        error = "the state file has no room for another pinned anchor";
        return false;
    }
    index = file.slots.size();       // a pin beyond the ordinary slots: the file grows
    return true;
}

static bool state_publish(StateFile& file, const AnchorState& anchor, size_t ordinary,
                          std::string& error) {
    size_t index = 0;
    if (!state_slot_for(file, anchor.tokens, ordinary, index, error)) return false;
    if (index >= file.slots.size()) {
        file.slots.push_back(SlotHeader{});
        file.header.slot_count = static_cast<uint32_t>(file.slots.size());
    }
    const uint64_t offset = slot_offset(file.header, index);
    // The payload first, the slot header (with its checksum) second: a crash leaves the OLD header's
    // checksum over the new bytes, so the slot reads as invalid instead of as a wrong state.
    if (std::fseek(file.file, static_cast<long>(offset + kSlotHeaderBytes), SEEK_SET) != 0) {
        error = "cannot seek the state slot";
        return false;
    }
    uint64_t checksum = conversation_disk_checksum_seed;
    size_t i = 0;
    for (const auto& stage : anchor.stages) {
        for (int e = 0; e < 5; ++e) {
            const std::vector<uint8_t>& bytes = stage[static_cast<size_t>(e)].vec();
            if (i >= file.section_sizes.size() || bytes.size() != file.section_sizes[i]) {
                error = "the state size does not match the state file";
                return false;
            }
            ++i;
            if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), file.file) != bytes.size()) {
                error = "cannot write the state slot";
                return false;
            }
            checksum = conversation_disk_checksum(bytes.data(), bytes.size(), checksum);
        }
    }
    if (i != file.section_sizes.size()) {
        error = "the state stage count does not match the state file";
        return false;
    }
    SlotHeader slot;
    slot.magic = kSlotMagic;
    slot.tokens = anchor.tokens;
    slot.generation = ++file.header.generation;
    slot.payload_checksum = checksum;
    slot.content_checksum = anchor.content_checksum;
    slot.used = anchor.used;
    slot.flags = anchor.pinned ? kSlotPinned : 0;
    if (std::fseek(file.file, static_cast<long>(offset), SEEK_SET) != 0 ||
        std::fwrite(&slot, 1, sizeof(slot), file.file) != sizeof(slot)) {
        error = "cannot write the state slot header";
        return false;
    }
    file.slots[index] = slot;
    file.header.chain_cover = std::max(file.header.chain_cover, anchor.tokens);
    file.header.checksum = state_header_checksum(file.header);
    if (std::fseek(file.file, 0, SEEK_SET) != 0 ||
        std::fwrite(&file.header, 1, sizeof(file.header), file.file) != sizeof(file.header) ||
        std::fflush(file.file) != 0) {
        error = "cannot write the state header";
        return false;
    }
    return true;
}

static bool state_read(const StateFile& file, uint64_t tokens, uint64_t content,
                       const std::vector<std::array<std::vector<uint8_t>*, 5>>& out,
                       std::string& error) {
    size_t index = SIZE_MAX;
    for (size_t i = 0; i < file.slots.size(); ++i)
        if (file.slots[i].magic == kSlotMagic && file.slots[i].tokens == tokens) {
            index = i;
            break;
        }
    if (index == SIZE_MAX) return false;
    const SlotHeader& slot = file.slots[index];
    if (slot.content_checksum != content || out.size() != file.header.stage_count) return false;
    uint64_t at = slot_offset(file.header, index) + kSlotHeaderBytes;
    uint64_t checksum = conversation_disk_checksum_seed;
    size_t i = 0;
    bool ok = true;
    for (const auto& stage : out) {
        for (int e = 0; e < 5 && ok; ++e) {
            const uint64_t bytes = i < file.section_sizes.size() ? file.section_sizes[i] : 0;
            ++i;
            std::vector<uint8_t>* dst = stage[static_cast<size_t>(e)];
            dst->resize(static_cast<size_t>(bytes));
            if (std::fseek(file.file, static_cast<long>(at), SEEK_SET) != 0 ||
                !read_exact(file.file, dst->data(), bytes)) {
                ok = false;
                break;
            }
            checksum = conversation_disk_checksum(dst->data(), static_cast<size_t>(bytes), checksum);
            at += bytes;
        }
    }
    if (ok && checksum == slot.payload_checksum) return true;
    // A torn slot, or one written for other bytes: the position stays a match without a state, so a
    // restore resumes from an earlier anchor instead of trusting a wrong state.
    for (auto& stage : out)
        for (int e = 0; e < 5; ++e) stage[static_cast<size_t>(e)]->clear();
    error = "the stored state slot does not verify";
    return false;
}

bool ConversationDiskStore::state_store(const std::string& name, ConversationDiskRecord& record,
                                        uint64_t end, const std::vector<int32_t>& content_ids,
                                        const std::vector<ConversationImageKey>& content_imgs,
                                        size_t* written_bytes, std::string& error) const {
    if (written_bytes != nullptr) *written_bytes = 0;
    std::vector<std::array<uint64_t, 5>> sizes;
    sizes.reserve(record.stages.size());
    for (const auto& stage : record.stages) {
        const ConversationCheckpoint& live = stage.live;
        sizes.push_back({live.gdn.size(), live.ple.size(), live.tails.size(), live.dead.size(),
                         live.block_pos.size()});
    }
    StateFile states;
    if (!state_open(path_for(name), options_.identity, record.stages.front().cvec, &sizes,
                    anchor_slots(), true, states, error)) {
        return false;
    }
    // Preserve one original continuation once a later input actually consumes
    // its generated end. Dense new input checkpoints must not rotate it away.
    // Content identity proves consumption; a rewritten, unconsumed reply does
    // not become a permanent historical state.
    if (states.header.origin_tokens == 0 && states.header.terminal_from > 0 &&
        states.header.chain_cover < end && states.header.chain_cover <= content_ids.size()) {
        for (const auto& slot : states.slots)
            if (slot.magic == kSlotMagic && slot.tokens == states.header.chain_cover &&
                slot.content_checksum == prefix_checksum(content_ids, content_imgs, (size_t) slot.tokens)) {
                states.header.origin_tokens = slot.tokens;
                break;
            }
    }
    // The chain's terminal boundary travels with its sidecar: a later replacement of the tail needs
    // it to prove that the bytes it drops were the generated reply (0 = unknown, so it branches).
    states.header.terminal_from = (uint64_t) std::max<int64_t>(0, record.stages.front().terminal_from);
    const size_t ordinary = ordinary_slots();
    std::vector<AnchorState> anchors = lift_anchors(record, end, content_ids, content_imgs);
    for (const AnchorState& anchor : anchors)
        if (!state_publish(states, anchor, ordinary, error)) {
            return false;
        }
    if (written_bytes != nullptr)
        *written_bytes = anchors.size() * (kSlotHeaderBytes + (size_t) states.header.slot_bytes) +
                         kStateHeaderBytes;
    return true;
}

void ConversationDiskStore::state_discard(const std::string& name, uint64_t tokens, bool remove_file) const {
    const std::filesystem::path path = state_path_for(name);
    std::error_code ignored;
    if (remove_file) {
        std::filesystem::remove(path, ignored);
        return;
    }
    StateFile states;
    states.file = std::fopen(path.string().c_str(), "r+b");
    if (states.file == nullptr) return;
    if (!read_exact(states.file, &states.header, sizeof(states.header)) ||
        states.header.magic != kStateMagic || states.header.version != kStateVersion ||
        states.header.checksum != state_header_checksum(states.header) ||
        states.header.stage_count == 0 || states.header.slot_count == 0 ||
        states.header.slot_count > kMaxStateSlots)
        return;
    bool changed = false;
    for (uint32_t i = 0; i < states.header.slot_count; ++i) {
        SlotHeader slot;
        if (std::fseek(states.file, static_cast<long>(slot_offset(states.header, i)), SEEK_SET) != 0 ||
            !read_exact(states.file, &slot, sizeof(slot)))
            return;
        if (slot.magic != kSlotMagic || slot.tokens <= tokens) continue;
        const SlotHeader blank;
        if (std::fseek(states.file, static_cast<long>(slot_offset(states.header, i)), SEEK_SET) != 0 ||
            std::fwrite(&blank, 1, sizeof(blank), states.file) != sizeof(blank))
            return;
        changed = true;
    }
    if (states.header.origin_tokens > tokens) {
        states.header.origin_tokens = 0;
        states.header.checksum = state_header_checksum(states.header);
        if (std::fseek(states.file, 0, SEEK_SET) != 0 ||
            std::fwrite(&states.header, 1, sizeof(states.header), states.file) != sizeof(states.header))
            return;
        changed = true;
    }
    if (changed) std::fflush(states.file);
}

// The anchor positions a chain's sidecar currently retains, ascending.  The retention rotates the
// slots, so this - not the index of an earlier write - is the truth about which anchors still hold
// a state on disk.
static std::vector<uint64_t> retained_slots(const StateFile& file, uint64_t cover) {
    std::vector<uint64_t> slots;
    slots.reserve(file.slots.size());
    for (const SlotHeader& slot : file.slots) {
        if (slot.magic != kSlotMagic || slot.tokens == 0 || slot.tokens > cover) continue;
        slots.push_back(slot.tokens);
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    return slots;
}

void ConversationDiskStore::settle_entry_states(Entry& entry, std::vector<bool> inline_states) const {
    StateFile states;
    std::string ignored;
    std::vector<uint64_t> slots;
    if (state_open(path_for(entry.name), options_.identity, entry.cvec, nullptr, 0, false, states, ignored))
        slots = retained_slots(states, entry.cover);
    // A retained anchor that is not a point yet - a previous panel's end, or a pinned anchor below a
    // borrowed base - still is a restore point the caller may select, so the index carries it too.
    for (const uint64_t tokens : slots) {
        if (entry.points.empty() || tokens > (uint64_t) entry.points.front().ids.size()) continue;
        if (find_checkpoint_index(entry.points, (size_t) tokens) != SIZE_MAX) continue;
        ConversationCheckpoint point;
        point.ids.assign(entry.points.front().ids.begin(),
                         entry.points.front().ids.begin() + (std::ptrdiff_t) tokens);
        for (const auto& image : entry.points.front().imgs)
            if ((uint64_t) image.start < tokens) point.imgs.push_back(image);
        entry.points.push_back(std::move(point));
        inline_states.push_back(false);
    }
    inline_states.resize(entry.points.size(), false);
    entry.point_states.assign(entry.points.size(), false);
    for (size_t i = 0; i < entry.points.size(); ++i)
        entry.point_states[i] = inline_states[i] ||
            std::binary_search(slots.begin(), slots.end(), (uint64_t) entry.points[i].ids.size());
    entry.point_inline_states = std::move(inline_states);
}

// The terminal boundary a read reports: the chain's own persisted value, but only when the content
// it returns reaches it.  A shorter prefix cut reports 0 (unknown) instead of a boundary past its
// own ids, so a caller never re-persists a boundary its image does not hold.
static void apply_terminal_from(ConversationDiskRecord& record, uint64_t terminal_from) {
    if (record.stages.empty()) return;
    const uint64_t limit = (uint64_t) record.stages.front().live.ids.size();
    const uint64_t value = terminal_from != 0 && terminal_from <= limit ? terminal_from : 0;
    for (auto& stage : record.stages) stage.terminal_from = (int64_t) value;
}

ConversationDiskStatus ConversationDiskStore::get(const std::string& name, ConversationDiskRecord& record,
                                                  std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    const Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    ConversationDiskStatus status = read_chain(path_for(name), 0, record, error, UINT64_MAX);
    if (status == ConversationDiskStatus::corrupt) {
        // A torn tail (a crash mid-append) truncates back to the last complete panel instead of
        // losing the whole conversation.
        if (truncate_torn_tail(name)) {
            std::string retry_error;
            status = read_chain(path_for(name), 0, record, retry_error, UINT64_MAX);
            if (status == ConversationDiskStatus::ok) {
                error = std::move(retry_error);
                if (Entry* used = find(name)) {
                    used->stamp = ++stamp_;
                    const uint64_t bytes = std::filesystem::file_size(path_for(name));
                    bytes_ += bytes - used->bytes;
                    used->bytes = bytes;
                }
                bound_checkpoint_states(record);
                apply_terminal_from(record, entry->terminal_from);
                return status;
            }
        }
        // Real corruption: drop the chain (and every chain that borrows from it).
        std::error_code remove_error;
        if (Entry* stale = find(name)) {
            if (!stale->base.empty())
                if (Entry* base = find(stale->base)) --base->refs;
            std::filesystem::remove(path_for(name), remove_error);
            bytes_ -= stale->bytes;
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(stale - entries_.data()));
        }
        ++corruptions_;
        return ConversationDiskStatus::corrupt;
    }
    if (status == ConversationDiskStatus::ok) {
        bound_checkpoint_states(record);
        apply_terminal_from(record, entry->terminal_from);
        if (Entry* used = find(name)) used->stamp = ++stamp_;
    }
    return status;
}

ConversationDiskStatus ConversationDiskStore::get_prefix(const std::string& name,
                                                         ConversationDiskRecord& record, uint64_t upto,
                                                         std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    if (upto == 0) {
        error = "the prefix length is zero";
        return ConversationDiskStatus::invalid;
    }
    const Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    const ConversationDiskStatus status = read_chain(path_for(name), 0, record, error, upto);
    if (status != ConversationDiskStatus::ok) return status;
    // The cut often lands on a retained checkpoint inside a panel, not on a panel end. Promote
    // that checkpoint's state to the live state, then bound the rest: the caller asked for an
    // exact restore point and must receive a state at it.
    promote_anchor_state(record, std::min<uint64_t>(upto, entry->cover));
    bound_checkpoint_states(record);
    apply_terminal_from(record, entry->terminal_from);
    if (Entry* used = find(name)) used->stamp = ++stamp_;
    return status;
}

void ConversationDiskStore::promote_anchor_state(ConversationDiskRecord& record, uint64_t upto) {
    for (auto& stage : record.stages) {
        if (stage.live.has_state()) continue;               // a panel ended exactly at the cut
        if ((uint64_t) stage.live.ids.size() != upto) continue;
        for (const auto& checkpoint : stage.checkpoints) {
            if ((uint64_t) checkpoint.ids.size() != upto || !checkpoint.has_state()) continue;
            // A payload copy shares its bytes (ConversationPayload): the promotion costs none.
            stage.live.gdn = checkpoint.gdn;
            stage.live.ple = checkpoint.ple;
            stage.live.tails = checkpoint.tails;
            stage.live.dead = checkpoint.dead;
            stage.live.block_pos = checkpoint.block_pos;
            stage.live.used = checkpoint.used;
            break;
        }
    }
}

// Verifies the LAST panel's payload checksum and truncates the file when it is torn, patching the
// chain header and re-indexing the entry.  Returns false when the tail is intact (or the file does
// not validate at all - a genuine corruption the caller removes).
bool ConversationDiskStore::truncate_torn_tail(const std::string& name) {
    const std::filesystem::path target = path_for(name);
    std::FILE* file = std::fopen(target.string().c_str(), "rb");
    if (file == nullptr) return false;
    ChainHeader header;
    if (!read_exact(file, &header, sizeof(header)) || header.magic != kMagic ||
        header.version != format_version || header.base_kind > kBaseChain || header.panel_count > kMaxPanels ||
        header.panel_count == 0) {
        std::fclose(file);
        return false;
    }
    // Walk to the last panel; verify its payload checksum.
    uint64_t pos = kChainHeaderBytes;
    uint64_t kept_end = header.base_kind != kBaseNone ? header.base_tokens : 0;
    bool ok = true;
    for (uint64_t k = 0; ok && k < header.panel_count; ++k) {
        PanelHeader p{};
        if (std::fseek(file, static_cast<long>(pos), SEEK_SET) != 0 ||
            !read_exact(file, &p, sizeof(p)) || p.magic != kPanelMagic ||
            p.base_tokens > kept_end) {
            ok = false;
            break;
        }
        if (k + 1 == header.panel_count) {
            if (std::fseek(file, static_cast<long>(p.metadata_bytes), SEEK_CUR) != 0) { ok = false; break; }
            uint64_t checksum = conversation_disk_checksum_seed;
            std::vector<uint8_t> scratch(1u << 20);
            uint64_t remaining = p.payload_bytes;
            while (ok && remaining != 0) {
                const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, scratch.size()));
                if (!read_exact(file, scratch.data(), chunk)) { ok = false; break; }
                checksum = conversation_disk_checksum(scratch.data(), chunk, checksum);
                remaining -= chunk;
            }
            if (!ok) break;
            if (checksum == p.payload_checksum) { std::fclose(file); return false; }   // tail intact
            std::fclose(file);
            // Truncate back to the start of the torn panel and patch the chain header.
            const uint64_t panel_at = pos;
            std::error_code ignored;
            std::filesystem::resize_file(target, panel_at, ignored);
            std::FILE* writer = std::fopen(target.string().c_str(), "r+b");
            if (writer == nullptr) return false;
            ChainHeader patched = header;
            patched.panel_count = k;
            patched.total_tokens = kept_end;
            bool patched_ok = std::fwrite(&patched, 1, sizeof(patched), writer) == sizeof(patched);
            std::fclose(writer);
            if (!patched_ok) return false;
            // Re-index the entry from the retained panels.
            Entry rebuilt;
            std::string index_error;
            if (index_file(target, name, rebuilt, index_error) != ConversationDiskStatus::ok) return false;
            if (Entry* used = find(name)) {
                bytes_ += rebuilt.bytes - used->bytes;
                used->bytes = rebuilt.bytes;
                used->cover = rebuilt.cover;
                state_discard(name, kept_end, false);
                used->points = std::move(rebuilt.points);
                used->point_states = std::move(rebuilt.point_states);
                used->point_inline_states = std::move(rebuilt.point_inline_states);
                used->terminal_from = rebuilt.terminal_from;
            }
            return true;
        }
        kept_end = p.end_tokens;
        pos += sizeof(p) + p.metadata_bytes + p.payload_bytes;
    }
    std::fclose(file);
    return false;
}

ConversationDiskStatus ConversationDiskStore::read_chain(const std::filesystem::path& path, int depth,
                                                         ConversationDiskRecord& out, std::string& error,
                                                         uint64_t upto) const {
    return read_chain_impl(path, *this, depth, out, error, upto);
}

ConversationDiskStatus ConversationDiskStore::read_chain_impl(const std::filesystem::path& path,
                                                             const ConversationDiskStore& self, int depth,
                                                             ConversationDiskRecord& out,
                                                             std::string& error, uint64_t upto) {
    if (depth > static_cast<int>(max_base_depth)) {
        error = "corrupt conversation record: chain base depth";
        return ConversationDiskStatus::corrupt;
    }
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        if (errno == ENOENT) return ConversationDiskStatus::miss;
        error = "cannot open " + path.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        std::fclose(file);
        error = std::string("corrupt conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    ChainHeader header;
    std::vector<ParsedPanel> panels;
    if (!read_chain_meta(file, self.options_.identity, self.options_.budget_bytes, header, panels, error,
                         upto))
        return bad(error.empty() ? "header" : error.c_str());
    // The cut: a chain never returns more than it covers, and a prefix read never returns more
    // than the caller asked for.
    const uint64_t limit = std::min<uint64_t>(upto, header.total_tokens);
    const uint64_t base_tokens = header.base_kind != kBaseNone ? header.base_tokens : 0;
    // Resolve the base (depth is bounded; a base's own base is resolved the same way).
    ConversationDiskRecord base;
    if (header.base_kind != kBaseNone) {
        const std::string base_name = base_name_of(header);
        const Entry* base_entry = self.find(base_name);
        if (base_entry == nullptr) return bad("missing base chain");
        const uint64_t base_limit = std::min<uint64_t>(limit, base_tokens);
        const ConversationDiskStatus got = self.read_chain_impl(self.path_for(base_name), self,
                                                               depth + 1, base, error, base_limit);
        if (got != ConversationDiskStatus::ok) return bad("missing base chain");
        if (base.stages.empty() || (uint64_t) base.stages.front().live.ids.size() != base_limit)
            return bad("base coverage");
        // A base kind 2 chain is longer than the borrowed prefix: keep only the borrowed part.
        for (auto& stage : base.stages) {
            if ((uint64_t) stage.live.ids.size() > base_tokens) stage.live.ids.resize((size_t) base_tokens);
            stage.live.imgs.erase(std::remove_if(stage.live.imgs.begin(), stage.live.imgs.end(),
                [&](const ConversationImageKey& image) { return (uint64_t) image.start >= base_tokens; }),
                stage.live.imgs.end());
        }
    }
    // Stage layout must be consistent across the base and every panel.
    const size_t stage_count = panels.front().delta.stages.size();
    if (!base.stages.empty()) {
        if (!stages_compatible(base, panels.front().delta, error)) return bad(error.c_str());
    }
    for (size_t k = 1; k < panels.size(); ++k)
        if (!stages_compatible(panels[0].delta, panels[k].delta, error)) return bad(error.c_str());
    // Build the merged image skeleton from the metadata (ids, imgs, checkpoint identities, shapes).
    out.stages.resize(stage_count);
    std::vector<std::vector<size_t>> cp_start(stage_count);   // per panel, per stage: checkpoint index
    std::vector<std::vector<std::array<size_t, 5>>> c_per(stage_count);   // bytes per unit, per stage/layer
    // Which checkpoint STATES this read keeps, per stage, decided from the metadata alone: a
    // dropped state's bytes are still verified, but they are never allocated.  A legacy panel that
    // carries 149 running states then costs the bound, not 149 payloads.
    std::vector<std::vector<bool>> keep_states(stage_count);
    // The chain's bounded anchor states, when it has a sidecar: they carry both the states and the
    // positions, so a panel of this chain repeats neither.
    StateFile states;
    std::string side_error;
    const bool has_states = state_open(path, self.options_.identity, header.cvec != 0, nullptr, 0,
                                       false, states, side_error);
    std::vector<std::vector<bool>> from_sidecar(stage_count);
    std::vector<uint64_t> checkpoint_limits(panels.size(), limit);
    for (size_t k = panels.size() - 1; k > 0; --k)
        checkpoint_limits[k - 1] = std::min(checkpoint_limits[k], panels[k].base_tokens);
    for (size_t si = 0; si < stage_count; ++si) {
        SavedConversation& st = out.stages[si];
        const SavedConversation* base_st = base.stages.empty() ? nullptr : &base.stages[si];
        const SavedConversation& first = panels.front().delta.stages[si];
        st.geometry = first.geometry;
        st.layer_lo = first.layer_lo;
        st.layer_hi = first.layer_hi;
        st.cvec = first.cvec;
        st.kv.resize(first.kv.size());
        (void) si;
        // The record is the caller's and may already hold a previous read: clear the live state so a
        // cut that ends inside a panel cannot leave the older position's state behind.
        st.live.drop_state();
        st.live.ids.clear();
        st.live.imgs.clear();
        if (base_st != nullptr) {
            st.live.ids = base_st->live.ids;
            st.live.imgs = base_st->live.imgs;
        }
        for (const auto& panel : panels) {
            const SavedConversation& stage = panel.delta.stages[si];
            if ((uint64_t) stage.live.ids.size() != panel.end_tokens - panel.base_tokens)
                return bad("token slice");
            // A panel that starts at or after the cut cannot contribute to [0, limit): skip it.  The
            // panels do not run in order after a rewind, so a later panel may still apply.
            if (panel.base_tokens >= limit) continue;
            if (st.live.ids.size() < panel.base_tokens) return bad("lengths:staids");
            st.live.ids.resize((size_t) panel.base_tokens);
            st.live.imgs.erase(std::remove_if(st.live.imgs.begin(), st.live.imgs.end(),
                [&](const ConversationImageKey& image) { return image.start >= (int64_t) panel.base_tokens; }), st.live.imgs.end());
            const uint64_t take = std::min<uint64_t>(panel.end_tokens, limit) - panel.base_tokens;
            st.live.ids.insert(st.live.ids.end(), stage.live.ids.begin(),
                               stage.live.ids.begin() + (std::ptrdiff_t) take);
            for (const auto& image : stage.live.imgs)
                if ((uint64_t) image.start < limit) st.live.imgs.push_back(image);
        }
        if ((uint64_t) st.live.ids.size() != limit) return bad("lengths:staids");
        st.live.used = panels.back().delta.stages[si].live.used;
        // Checkpoints: the base's (fully loaded arrays), then the panels' (ids/imgs now, arrays
        // streamed below).
        st.checkpoints.clear();
        if (base_st != nullptr)
            for (auto& c : base_st->checkpoints) st.checkpoints.push_back(std::move(c));
        cp_start[si].resize(panels.size());
        for (size_t k = 0; k < panels.size(); ++k) {
            cp_start[si][k] = st.checkpoints.size();
            for (auto& c : panels[k].delta.stages[si].checkpoints) {
                ConversationCheckpoint out_c;
                out_c.ids = std::move(c.ids);
                out_c.imgs = std::move(c.imgs);
                out_c.used = c.used;
                if (out_c.ids.size() > checkpoint_limits[k]) out_c.ids.clear();
                st.checkpoints.push_back(std::move(out_c));
            }
        }
        from_sidecar[si].assign(st.checkpoints.size(), false);
        // The chain's anchors: the sidecar holds their positions and their states, so a panel does
        // not repeat a checkpoint's ids.  A position a legacy panel already carries stays as it is
        // (the state below still attaches to it).  An anchor at or below a BORROWED base is the
        // chain's own - a pin the caller kept across the borrow - so it is offered too; the state
        // binds to its content, so a slot the merged content does not match never verifies.
        if (has_states) {
            for (const SlotHeader& slot : states.slots) {
                if (slot.magic != kSlotMagic || slot.tokens == 0) continue;
                if (slot.tokens > limit) continue;
                bool known = false;
                for (const auto& checkpoint : st.checkpoints)
                    if ((uint64_t) checkpoint.ids.size() == slot.tokens) {
                        known = true;
                        break;
                    }
                if (known) continue;
                ConversationCheckpoint anchor;
                anchor.ids.assign(st.live.ids.begin(), st.live.ids.begin() + (std::ptrdiff_t) slot.tokens);
                for (const auto& image : st.live.imgs)
                    if ((uint64_t) image.start < slot.tokens) anchor.imgs.push_back(image);
                anchor.used = slot.used;
                anchor.pinned = (slot.flags & kSlotPinned) != 0;
                st.checkpoints.push_back(std::move(anchor));
            }
            from_sidecar[si].assign(st.checkpoints.size(), false);
        }
        if (self.options_.max_checkpoint_states != 0) {
            // The caller's own bound (bound_checkpoint_states) runs over the record AFTER the
            // positions above the cut are dropped, so this allocation gate must decide over the same
            // list: an emptied entry keeps its slot for the payload walk but is not a position.
            std::vector<size_t> eligible;
            eligible.reserve(st.checkpoints.size());
            for (size_t j = 0; j < st.checkpoints.size(); ++j)
                if (!st.checkpoints[j].ids.empty()) eligible.push_back(j);
            keep_states[si].assign(st.checkpoints.size(), false);
            for (const size_t j : conversation_checkpoint_selection(st.checkpoints, eligible,
                                                                    self.options_.max_checkpoint_states))
                keep_states[si][j] = true;
            // The caller's own restore point is never a victim, whatever the policy says.
            for (size_t j = 0; j < st.checkpoints.size(); ++j)
                if ((uint64_t) st.checkpoints[j].ids.size() == limit) keep_states[si][j] = true;
        }
        for (size_t m = 0; m < first.kv.size(); ++m) {
            const ConversationKv& shape = first.kv[m];
            ConversationKv& out_kv = st.kv[m];
            // Each panel's tail starts at the byte offset of the PAGE containing its base and
            // runs to the panel's final (page-capped) cells: the overlap page is re-saved whole,
            // so a panel overwrites the previous content's partial page instead of duplicating
            // it.  The merged buffer totals come from the LAST panel's final cell count, or from
            // the cut when the caller asked for fewer tokens than that panel holds.
            const ConversationKv& last_kv = panels.back().delta.stages[si].kv[m];
            const auto last_panel_lens = panel_kv_lens(panels.back().delta, panels.back().lengths);
            if (last_panel_lens.size() <= si || last_panel_lens[si].size() <= m) return bad("lengths:kvlen");
            const std::array<size_t, 5> last_lens = last_panel_lens[si][m];
            const int64_t last_page = std::max<int64_t>(1, last_kv.page_size);
            const bool cut = limit < panels.back().end_tokens;
            out_kv.format = shape.format;
            // A cut re-derives the extent fields: the panel's own values describe its full end,
            // and the engine validates the image against the cut.
            out_kv.cells = cut ? kv_end_units((int64_t) limit, last_page, 0) : last_kv.cells;
            out_kv.heads = shape.heads;
            out_kv.head_dim = shape.head_dim;
            out_kv.page_size = shape.page_size;
            out_kv.pooled_rows = cut
                ? (last_kv.pooled_rows == 0 ? 0 : kv_end_units((int64_t) limit, last_page, 4))
                : last_kv.pooled_rows;
            out_kv.idx_dim = shape.idx_dim;
            // Each panel's tail covers the layout UNITS after its base: page-capped cells for the
            // K/V buffers, pooled rows for the indexer (see kv_units).  The merged buffer totals
            // come from the cut's final unit count.
            std::array<size_t, 5> section_per{};      // bytes per unit of the merged layout
            std::array<size_t, 5> kv_total{};
            for (int e = 0; e < 5; ++e) {
                const int64_t first_unit = last_kv.first_units[(size_t) e];
                const int64_t units = kv_units(first_unit, (int64_t) panels.back().end_tokens, last_page, e);
                if (first_unit < 0 || units <= 0) return bad("lengths:kvcell");
                if (last_lens[(size_t) e] % (uint64_t) units != 0) return bad("lengths:kvmod");
                section_per[(size_t) e] = last_lens[(size_t) e] / (uint64_t) units;
                kv_total[(size_t) e] =
                    (size_t) kv_end_units((int64_t) limit, last_page, e) *
                    section_per[(size_t) e];
            }
            // Validate every panel's tail lengths against the same per-unit size (the stage layout
            // is already validated identical across panels).
            for (auto& panel : panels) {
                const ConversationKv& kv_panel = panel.delta.stages[si].kv[m];
                const auto panel_lens = panel_kv_lens(panel.delta, panel.lengths);
                if (panel_lens.size() <= si || panel_lens[si].size() <= m) return bad("lengths:kvlen");
                const std::array<size_t, 5> lens = panel_lens[si][m];
                for (int e = 0; e < 5; ++e) {
                    const int64_t first = kv_panel.first_units[(size_t) e];
                    const int64_t units = kv_units(first, (int64_t) panel.end_tokens, last_page, e);
                    if (first < 0 || units <= 0 || lens[(size_t) e] != (uint64_t) units * section_per[(size_t) e])
                        return bad("lengths:kvlen");
                }
            }
            c_per[si].resize(first.kv.size());
            c_per[si][m] = section_per;
            const ConversationKv* base_kv = base_st != nullptr ? &base_st->kv[m] : nullptr;
            out_kv.k.resize(kv_total[0]);
            out_kv.v.resize(kv_total[1]);
            out_kv.k_scale.resize(kv_total[2]);
            out_kv.v_scale.resize(kv_total[3]);
            out_kv.pooled.resize(kv_total[4]);
            (void) m;
            // Copy the base's bytes into the merged buffers.
            if (base_kv != nullptr) {
                auto copy_from = [&](ConversationBuffer& dst, const ConversationBuffer& src) {
                    const size_t count = std::min(src.size(), dst.size());
                    if (count == 0) return true;
                    return dst.visit(0, count, [&](uint8_t* p, size_t chunk, size_t at) {
                        return src.read(p, at, chunk);
                    });
                };
                if (!copy_from(out_kv.k, base_kv->k) || !copy_from(out_kv.v, base_kv->v) ||
                    !copy_from(out_kv.k_scale, base_kv->k_scale) || !copy_from(out_kv.v_scale, base_kv->v_scale) ||
                    !copy_from(out_kv.pooled, base_kv->pooled))
                    return bad("base copy");
            }
        }
    }
    // The anchors' states, and the cut panel's live state when that panel carries no inline state,
    // come from the sidecar.  A slot that does not verify leaves the position without a state, so a
    // restore resumes from an earlier anchor instead of trusting a wrong state.
    bool live_from_sidecar = false;
    if (has_states) {
        const bool inline_live = !panels.back().lengths.empty() && panels.back().lengths[0] != 0;
        if (!inline_live && panels.back().end_tokens == limit && !out.stages.empty()) {
            std::vector<std::array<std::vector<uint8_t>*, 5>> destinations;
            destinations.reserve(stage_count);
            for (auto& stage : out.stages) destinations.push_back(live_arrays(stage.live));
            std::string ignored;
            live_from_sidecar = state_read(states, limit,
                                           prefix_checksum(out.stages.front().live.ids,
                                                           out.stages.front().live.imgs),
                                           destinations, ignored);
        }
        for (size_t si = 0; si < stage_count; ++si) {
            for (size_t j = 0; j < out.stages[si].checkpoints.size(); ++j) {
                ConversationCheckpoint& anchor = out.stages[si].checkpoints[j];
                if (anchor.ids.empty() || anchor.has_state()) continue;
                // The load's state bound already dropped this position: the slot's bytes are not
                // read at all (the position stays a match, exactly like a legacy inline one).
                if (!keep_states[si].empty() && !keep_states[si][j]) continue;
                std::vector<std::array<std::vector<uint8_t>*, 5>> destinations;
                std::vector<size_t> indices;
                bool aligned = true;
                for (size_t other = 0; other < stage_count && aligned; ++other) {
                    const size_t at = find_checkpoint_index(out.stages[other].checkpoints,
                                                           anchor.ids.size());
                    if (at == SIZE_MAX) {
                        aligned = false;
                        break;
                    }
                    destinations.push_back(live_arrays(out.stages[other].checkpoints[at]));
                    indices.push_back(at);
                }
                if (!aligned) continue;
                std::string ignored;
                if (!state_read(states, (uint64_t) anchor.ids.size(),
                                prefix_checksum(anchor.ids, anchor.imgs), destinations, ignored))
                    continue;
                for (size_t other = 0; other < stage_count; ++other)
                    from_sidecar[other][indices[other]] = true;
            }
        }
    }
    // Stream every panel's payload in order.  Each panel's payload checksum is verified against its
    // own header; the file position was left at the end of the chain by read_chain_meta, so seek to
    // each payload.
    std::vector<uint8_t> scratch;
    for (size_t k = 0; k < panels.size(); ++k) {
        const ParsedPanel& panel = panels[k];
        // Same rule as the ids merge: a panel that starts at or after the cut contributes nothing to
        // [0, limit), so its payload is neither read nor verified.
        if (panel.base_tokens >= limit) continue;
        if (std::fseek(file, static_cast<long>(panel.payload_at), SEEK_SET) != 0)
            return bad("payload position");
        const bool last = k + 1 == panels.size();
        // The running state belongs to the panel's END.  It is the merged image's live state only
        // when that end is exactly the cut: a state never claims a position it was not captured at.
        const bool live_state = last && panel.end_tokens == limit;
        uint64_t checksum = conversation_disk_checksum_seed;
        auto read_checksummed = [&](void* dst, uint64_t bytes) -> bool {
            if (!read_exact(file, dst, bytes)) return false;
            checksum = conversation_disk_checksum(static_cast<const uint8_t*>(dst),
                                                  static_cast<size_t>(bytes), checksum);
            return true;
        };
        size_t li = 0;
        auto length = [&]() -> uint64_t {
            return li < panel.lengths.size() ? panel.lengths[li++] : 0;
        };
        for (size_t si = 0; si < stage_count; ++si) {
            SavedConversation& st = out.stages[si];
            // Stage live run-state: only the cut panel's lands in the merged image.
            std::array<std::vector<uint8_t>*, 5> live{};
            if (live_state && !live_from_sidecar) live = live_arrays(st.live);
            for (int e = 0; e < 5; ++e) {
                const uint64_t bytes = length();
                if (live_state && !live_from_sidecar) {
                    live[(size_t) e]->resize(static_cast<size_t>(bytes));
                    if (!read_checksummed(live[(size_t) e]->data(), bytes)) return bad("payload read");
                } else {
                    scratch.resize(static_cast<size_t>(bytes));
                    if (!read_checksummed(scratch.data(), bytes)) return bad("payload read");
                }
            }
            // Checkpoints: the panels' land in order after the base's.  A checkpoint whose ids were
            // dropped (it is at or above the cut, or a later panel rewrote its position), and one
            // the state bound does not retain, keeps no state: its bytes are verified and discarded.
            for (size_t j = 0; j < panel.delta.stages[si].checkpoints.size(); ++j) {
                const size_t at_index = cp_start[si][k] + j;
                ConversationCheckpoint& out_c = st.checkpoints[at_index];
                const bool keep = !out_c.ids.empty() &&
                                  (keep_states[si].empty() || keep_states[si][at_index]) &&
                                  !from_sidecar[si][at_index];
                std::array<std::vector<uint8_t>*, 5> dst{};
                if (keep) dst = live_arrays(out_c);
                for (int e = 0; e < 5; ++e) {
                    const uint64_t bytes = length();
                    if (keep) {
                        dst[(size_t) e]->resize(static_cast<size_t>(bytes));
                        if (!read_checksummed(dst[(size_t) e]->data(), bytes)) return bad("payload read");
                    } else {
                        scratch.resize(static_cast<size_t>(bytes));
                        if (!read_checksummed(scratch.data(), bytes)) return bad("payload read");
                    }
                }
            }
            // K/V tails into the pre-sized buffers at the page-aligned OFFSET of each panel's
            // base (tails overlap the previous content's partial page by design - they rewrite it).
            for (size_t m = 0; m < st.kv.size(); ++m) {
                ConversationKv& out_kv = st.kv[m];
                std::array<ConversationBuffer*, 5> dst = kv_arrays(out_kv);
                const int64_t page = std::max<int64_t>(1, out_kv.page_size);
                const ConversationKv& kv_panel = panel.delta.stages[si].kv[m];
                for (int e = 0; e < 5; ++e) {
                    const uint64_t bytes = length();
                    const int64_t first = kv_panel.first_units[(size_t) e];
                    const int64_t units = kv_units(first, (int64_t) panel.end_tokens, page, e);
                    if (first < 0 || units <= 0 || bytes != (uint64_t) units * c_per[si][m][(size_t) e])
                        return bad("lengths");
                    const size_t at = (size_t) first * c_per[si][m][(size_t) e];
                    const size_t available = at < dst[(size_t) e]->size() ? dst[(size_t) e]->size() - at : 0;
                    const size_t retained = std::min<size_t>((size_t) bytes, available);
                    if (retained && !dst[(size_t) e]->visit(at, retained,
                            [&](uint8_t* p, size_t chunk, size_t) { return read_checksummed(p, chunk); }))
                        return bad("payload read");
                    uint64_t remaining = bytes - retained;
                    scratch.resize(1u << 20);
                    while (remaining) {
                        const size_t chunk = (size_t) std::min<uint64_t>(remaining, scratch.size());
                        if (!read_checksummed(scratch.data(), chunk)) return bad("payload read");
                        remaining -= chunk;
                    }
                }
            }
        }
        if (li != panel.lengths.size()) return bad("section sizes");
        if (checksum != panel.payload_checksum) return bad("payload checksum");
    }
    for (auto& stage : out.stages)
        stage.checkpoints.erase(std::remove_if(stage.checkpoints.begin(), stage.checkpoints.end(),
            [&](const ConversationCheckpoint& checkpoint) {
                return checkpoint.ids.empty() || checkpoint.ids.size() > stage.live.ids.size() ||
                       !std::equal(checkpoint.ids.begin(), checkpoint.ids.end(), stage.live.ids.begin());
            }), stage.checkpoints.end());
    std::fclose(file);
    return ConversationDiskStatus::ok;
}
} // namespace strata::core
