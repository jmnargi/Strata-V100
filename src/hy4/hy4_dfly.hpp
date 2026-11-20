// src/hy4/hy4_dfly.hpp - Hy4-flash's DFly2 drafter: its own GGUF (general.architecture hy_v4_flash_dfly, read directly,
// its Q8_0 projections and F32 vectors used as stored), the target rows it reads (each decode window's last-layer hidden
// state after the output norm, and each sparse layer's indexer picks), and the block pass over the target's own KV cache.
//
// A block is the anchor token (the last pick, whose position the target has not run) and n mask tokens. The anchor's
// embedding is fused with the target's hidden state at the row before it (gated fusion); five dense MLA layers run over
// the block's rows, each attending the rows its target layer selected for the source row (index share) in that layer's
// latent cache (KV share) and the block's own rows before it; each mask row then passes the sequential hidden correction
// with the embedding of the token drafted before it, and the target's output head gives the draft. The drafter keeps no
// cache and no indexer: a rejected block leaves nothing behind. Drafts are greedy; the target's own forward decides every
// row, so a draft only changes the speed.
//
// The forward (the inference reference is vLLM's hy_v4 DFly drafter; see Basalt src/models/hy4flash/dfly.hpp):
//   1. Gated fusion, row 0 only: x0 = e(a) + gate * eh_proj(enorm(e(a)) :: hnorm(h_s)), h_s the tap's hidden row after
//      the output norm. Rows 1..n: e(mask).
//   2. Five dense layers, one residual stream, pre-norm: y = input_ln(x); x += post_proj_norm(o_proj(attention(y)));
//      x += mlp(post_attention_ln(x)), a plain SwiGLU. z = final_norm(x).
//   3. Attention of layer i: the target's absorbed MLA with the drafter's own weights. q = q_b(q_a_ln(q_a(y))), heads of
//      192 nope + 64 rope; latent kv_a_ln(kv_a(y)[:512]), rope part kv_a(y)[512:]; interleaved-pair rope at the target's
//      base, position pos + r; kv_b rows per head are 192 key rows then 256 value rows, used absorbed; scale 1/sqrt(256);
//      the sink adds exp(sink) to the softmax denominator; the output is gated by sigmoid(linear_gate(y)), then o_proj.
//      The keys and values are target layer kv_layers[i]'s cache rows; the rows are the cells its indexer selected for
//      row s (every position before the anchor while its indexer keeps them all), the same for every block row; block row
//      r also attends the block's rows 0..r, whose K/V come from the drafter's own kv_a and live in scratch (rounded to
//      the target's cache form first when it is int8). Nothing is written into the target's cache.
//   4. Hidden correction, in order, draft j = 0..n-1 from row j + 1 and the token before it (a, then draft j - 1):
//      c = hidden_norm(z[j+1]) :: embed_norm(e(prev)), h' = z[j+1] + down(silu(gate(c)) * up(c)).
//   5. Head: the target's output head over h', argmax. Confidence: sigmoid(confidence_head(h')).
//
// The target supplies what the drafter does not own: an embedding row for a token (host), the output head over a device
// row (its own projection), each draft layer's hidden row and indexer cells through the tap, and each target layer's latent
// cache rows as a DflyCache (device, format). The drafter gathers the selected rows itself, coalescing consecutive cells;
// a cache on another device is copied with cudaMemcpyPeerAsync. The target's cache writes for the last window must be
// quiescent when draft_begin runs.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace hy4 {

constexpr int kDflyBlock = 6;      // rows of a full block: the anchor and five masks
constexpr int kDflyDrafts = kDflyBlock - 1;
constexpr int kDflyTapRows = 16;   // rows of a window the tap keeps
constexpr int kDflyMaxLayers = 8;
constexpr int kDflyNope = 192, kDflyRank = 512, kDflyV = 256, kDflyRope = 64;
constexpr int kDflyRowWidth = kDflyRank + kDflyRope;   // 576
constexpr int kDflyKvbHead = kDflyNope + kDflyV;       // 448 rows of one head in kv_b_proj

/// The drafter's shape, from the GGUF's hy_v4_flash_dfly.* keys and its tensors.
struct DflyDims {
    int layers = 0, E = 0, q_lora = 0, heads = 0, ffn = 0, hc_ffn = 0;
    int vocab = 0, mask_id = 0;
    int topk = 2048;                  // the target's selection width (target_index_topk)
    float rms_eps = 1e-5f, rope_base = 1e4f;
    std::vector<int> kv_layers;       // the target layer whose cache each draft layer reads
    std::vector<int> idx_layers;      // the target layer whose selection each draft layer reads
};

/// What the drafter must agree with in the target (from hy4's Geo; flags per target layer).
struct DflyTarget {
    int n_embd = 0, n_vocab = 0, topk = 0;
    float rms_eps = 1e-5f, rope_base = 1e4f;
    std::vector<int> heads;                // per target layer
    std::vector<uint8_t> sparse_indexed;   // per target layer: indexed and not sliding-window
    bool kv_q8 = false;                    // the target's latent cache rows are kv_store's 8-bit form
};

struct DflyInfo {
    DflyDims dims;
    uint64_t weights = 0;   // the drafter's tensors on the card
    uint64_t scratch = 0;   // the tap and the block pass's buffers
};

/// One target layer's latent cache. `rows` is row 0 of the layer's cache: F16 rows of 1152 bytes (576 values), or,
/// with q8, kv_store's rows of kv_q8_row_bytes(512, 64) = 672 bytes. `ring` 0: a row a position (the indexed layers).
struct DflyCache {
    /// Optional: copies n of this layer's raw cache rows into dst (head GPU) on `s`; `cells` are on the head GPU (the tap's
    /// copy) and null means positions 0 .. n-1. Without it the drafter copies same-device rows with a kernel, and rows of a
    /// cache on another device by coalescing consecutive cells (a device-to-host read of the cells, then peer copies).
    using Gather = std::function<bool(const int32_t* cells, int n, void* dst, cudaStream_t s)>;
    const void* rows = nullptr;
    int64_t row_bytes = 0;
    int ring = 0;
    int device = -1;
    bool q8 = false;
    Gather gather;
};

/// The drafter: its weights, its tap and its block pass, on one device (the target's head GPU).
class Dfly {
public:
    /// A token's embedding row (E values, host) as the target would put at that position.
    using Embed = std::function<bool(int32_t token, float* row)>;
    /// The target's output head over a device row of E values (its own matrix and quantisation), into device logits.
    using Head = std::function<bool(const float* h, float* logits, cudaStream_t s)>;

    Dfly();
    ~Dfly();
    Dfly(const Dfly&) = delete;
    Dfly& operator=(const Dfly&) = delete;

    /// Reads and checks the drafter GGUF's header and tensors without allocating: the dims and what it would take
    /// (weights, and the tap + block pass's scratch for sel_cap cells). The tap is a fixed 16-position ring, so `ctx`
    /// does not change the budget; it is validated against the drafter's position guard.
    static bool probe(const std::string& gguf, const DflyTarget& target, int sel_cap, int64_t ctx, DflyInfo& info,
                      std::string& err);

    /// Loads the drafter's tensors onto `device` (the head GPU; the current device is restored on return) and allocates
    /// the block pass's scratch. `sel_cap` is the most cells a target selection can hold (ix_top_k + ix_kpool); `ctx` the
    /// context in positions. `head` and `embed` may be empty until the first draft (they are then required).
    bool load(const std::string& gguf, const DflyTarget& target, int sel_cap, int64_t ctx, int device, Head head, Embed embed,
              std::string& err);
    bool on() const;
    const DflyDims& dims() const;
    uint64_t weights_bytes() const;
    uint64_t scratch_bytes() const;

    /// ---- the tap: what the target leaves of its last decode window ----
    /// Rows are kept by absolute position across windows: draft_begin(pos) reads exactly the row at pos - 1, which may be
    /// from an earlier window (a verifier's accepted prefix ends inside it). begin_window(pos0) drops the rows at and
    /// after pos0 - the new window recomputes those (a replay overwrites its slot) - and keeps the ones before it.
    void begin_window(int64_t pos0);
    void invalidate();                 // no row is valid (a prompt chunk, another path)
    void run(int row0) { row0_ = row0; }   // the next run of rows starts at this row of the window (-1: none is kept)
    /// The run's rows `first` .. `first` + n - 1 have the hidden states fin[0..n) (device rows of E floats, head GPU).
    void hidden(int first, int n, const float* fin, cudaStream_t s);
    /// Target layer `layer` selected `n` cells for run row `t`; n < 0: every position up to the row (count = pos).
    /// `cells` are on the head GPU. (Rows are absolute positions: begin_window/run set the window's start.)
    void selection(int layer, int t, const int32_t* cells, int n, cudaStream_t s);
    /// The same by absolute position (a window's row t is pos0 + t); `cells_device` is where `cells` lives.
    void hidden_at(int64_t pos, const float* fin, cudaStream_t s);
    void selection_at(int layer, int64_t pos, const int32_t* cells, int n, int cells_device, cudaStream_t s);
    /// The position whose row draft_begin(pos) uses (pos - 1), or -1 when the tap did not keep that row whole.
    int source_row(int64_t pos) const;

    /// ---- the block pass ----
    /// Enqueues the block of `n` drafts (1..5) after `anchor` at `pos`: `caches` indexed by target layer id holds each
    /// draft layer's cache. True with nothing pending when the tap lacks the source row (a prompt's end, another path).
    bool draft_begin(int32_t anchor, int64_t pos, int n, const DflyCache* caches, cudaStream_t s, int64_t rope_pos,
                     std::string& err);
    /// Runs the heads over the block begin() made, in order, each with the token before it: up to max_n drafts and their
    /// acceptance estimates.
    bool draft_end(int32_t* ids, float* conf, int max_n, int& n, cudaStream_t s, std::string& err);
    bool has_pending() const { return pending_ > 0; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    int row0_ = -1;
    int pending_ = 0, made_ = 0;
    int32_t anchor_ = 0;
};

}  // namespace hy4
