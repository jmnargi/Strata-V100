// src/kernels/batch_rows_parity.cpp - the batch window's one-launch-for-all-rows kernels (#2234, STRATA_BATCH_ATTN_ROWS /
// STRATA_BATCH_GDN_GROUPS) against the per-row launches they replace, bitwise (GPU, synthetic, no model).
//
//   1. qsa_block_scores_rows   == qsa_block_scores, one call per row, each row with its OWN indexer keys;
//   2. qsa_decode_attn_batch_rows == qsa_decode_attn_batch, one call per row, each row over its OWN K/V pools
//      (int8 codes + scales, and fp16);
//   3. STRATA_QSA_SEGMENTS: a run of 1/2/4/8 contiguous rows of ONE slot (one indexer key set, one K/V pool, the cells
//      at successive positions) - qsa_block_scores / qsa_block_topk / qsa_decode_attn_batch with nq = the run,
//      CAPTURED into a graph and instantiated once, the exec replayed A/B/A with changed device step contents (the batch
//      window's replay) against the eager per-row loop - and native_qsa_indexer_append_steps over the run (the partial
//      tail at 2046 and the block the run completes) == one call per row;
//   4. gdn_step_norm_groups    == gdn_step_norm_multi, one call per slot group, in the window (all outputs, the state
//      untouched) and in the 1-row commit (the state written after `keep` tokens).
// Every case compares with memcmp, not a tolerance. Usage: batch_rows_parity
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

int bad = 0;
void report(const char* what, bool same) {
    std::printf("  %-64s %s\n", what, same ? "bitwise" : "*** DIFFERS ***");
    if (!same) ++bad;
}

// ---- 1. the block scores ----
void scores_case(int rows) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    std::mt19937 rng(11 + rows);
    std::normal_distribution<float> nd(0.f, 1.f);
    k::QsaScoreRows sr;
    sr.n = rows;
    std::vector<const float*> pooled(rows), dead(rows);
    std::vector<float> q((size_t) rows * s.idx_n_head * s.idx_dim);
    for (auto& x : q) x = nd(rng);
    std::vector<int32_t> steps((size_t) rows * k::kStepCount);
    for (int r = 0; r < rows; ++r) {
        std::vector<float> p((size_t) max_blocks * s.idx_dim), d((size_t) s.idx_dim);
        for (auto& x : p) x = nd(rng);
        for (auto& x : d) x = nd(rng);
        pooled[r] = up(p);
        dead[r] = up(d);
        sr.pooled[r] = pooled[r];
        sr.dead[r] = dead[r];
        const int64_t pos = 700 + 997 * r + (r % 3) * 5;   // different lengths, and an incomplete block at the tail
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) pos;
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) (pos + 1);
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *d_rows = nullptr, *d_ref = nullptr;
    const size_t n = (size_t) rows * max_blocks;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_block_scores_rows(sr, d_q, d_steps, rows, max_blocks, s, d_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_scores(pooled[r], dead[r], d_q + (size_t) r * s.idx_n_head * s.idx_dim, d_steps + (size_t) r * k::kStepCount, 1,
                            max_blocks, s, d_ref + (size_t) r * max_blocks, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_block_scores_rows, %d rows (own keys each)", rows);
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
    // the window selects with one top-k launch over all rows (row-independent) where the per-row path ran one per row
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    int32_t *i_rows = nullptr, *i_ref = nullptr;
    ck(cudaMalloc(&i_rows, (size_t) rows * cap * 4), "malloc");
    ck(cudaMalloc(&i_ref, (size_t) rows * cap * 4), "malloc");
    ck(cudaMemset(i_rows, 0, (size_t) rows * cap * 4), "memset");
    ck(cudaMemset(i_ref, 0, (size_t) rows * cap * 4), "memset");
    k::qsa_block_topk(d_ref, d_steps, rows, max_blocks, cap, s, i_rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_block_topk(d_ref + (size_t) r * max_blocks, d_steps + (size_t) r * k::kStepCount, 1, max_blocks, cap, s,
                          i_ref + (size_t) r * cap, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<int32_t> ta = down(i_rows, (size_t) rows * cap), tb = down(i_ref, (size_t) rows * cap);
    bool tsame = true;
    for (int r = 0; r < rows; ++r) {
        const int64_t w = steps[(size_t) r * k::kStepCount + k::kStepWidth];
        tsame = tsame && std::memcmp(ta.data() + (size_t) r * cap, tb.data() + (size_t) r * cap, (size_t) w * 4) == 0;
    }
    std::snprintf(name, sizeof name, "qsa_block_topk over %d rows == one call per row", rows);
    report(name, tsame);
}

// ---- 2. the decode attention ----
void attn_case(int rows, bool int8) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t ctx_max = 5000, pages = (ctx_max + PS - 1) / PS, prow = pages * NKV * PS;
    const int64_t cap = k::qsa_selection_width(ctx_max, s);
    std::mt19937 rng(23 + rows + int8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    k::QsaAttnPoolsRows pr;
    pr.n = rows;
    std::vector<float> q((size_t) rows * NH * HD);
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> ids((size_t) rows * cap, 0), steps((size_t) rows * k::kStepCount, 0);
    for (int r = 0; r < rows; ++r) {
        k::QsaAttnPools& pl = pr.p[r];
        if (int8) {
            std::vector<int8_t> kq((size_t) prow * HD), vq((size_t) prow * HD);
            std::vector<uint16_t> ks((size_t) prow * (HD / 64)), vs(ks.size());
            for (auto& x : kq) x = (int8_t) code(rng);
            for (auto& x : vq) x = (int8_t) code(rng);
            for (auto& x : ks) x = f2h(sc(rng));
            for (auto& x : vs) x = f2h(sc(rng));
            pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs);
        } else {
            std::vector<uint16_t> kh((size_t) prow * HD), vh(kh.size());
            for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
            for (auto& x : vh) x = f2h(nd(rng));
            pl.k_pool = up(kh); pl.v_pool = up(vh);
        }
        std::vector<int32_t> table(pages);
        for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
        std::shuffle(table.begin(), table.end(), rng);   // each row's pool has its own page order
        pl.page_table = up(table);
        const int64_t nkv = 900 + 641 * r + (r % 2) * 3;
        const int64_t w = k::qsa_selection_width(nkv, s);
        std::vector<int32_t> all((size_t) nkv);
        for (int64_t c = 0; c < nkv; ++c) all[c] = (int32_t) c;
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + w);
        std::copy(all.begin(), all.begin() + w, ids.begin() + (size_t) r * cap);
        steps[(size_t) r * k::kStepCount + k::kStepPos] = (int32_t) (nkv - 1);
        steps[(size_t) r * k::kStepCount + k::kStepNKv] = (int32_t) nkv;
        steps[(size_t) r * k::kStepCount + k::kStepNBid] = (int32_t) (nkv / s.idx_block);
        steps[(size_t) r * k::kStepCount + k::kStepWidth] = (int32_t) w;
    }
    const float* d_q = up(q);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const int64_t scr = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float *scratch = nullptr, *d_rows = nullptr, *d_ref = nullptr;
    ck(cudaMalloc(&scratch, (size_t) rows * scr * 4), "malloc");
    const size_t n = (size_t) rows * NH * HD;
    ck(cudaMalloc(&d_rows, n * 4), "malloc");
    ck(cudaMalloc(&d_ref, n * 4), "malloc");
    ck(cudaMemset(d_rows, 0, n * 4), "memset");
    ck(cudaMemset(d_ref, 0, n * 4), "memset");
    k::qsa_decode_attn_batch_rows(d_q, pr, d_ids, d_steps, cap, s, scratch, d_rows, rows, nullptr);
    for (int r = 0; r < rows; ++r)
        k::qsa_decode_attn_batch(d_q + (size_t) r * NH * HD, pr.p[r], d_ids + (size_t) r * cap, d_steps + (size_t) r * k::kStepCount, cap,
                                 s, scratch + (size_t) r * scr, d_ref + (size_t) r * NH * HD, 1, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const std::vector<float> a = down(d_rows, n), b = down(d_ref, n);
    char name[96];
    std::snprintf(name, sizeof name, "qsa_decode_attn_batch_rows, %d rows, %s pools (own pools each)", rows, int8 ? "int8" : "fp16");
    report(name, std::memcmp(a.data(), b.data(), n * 4) == 0);
}

// ---- 3. one slot's rows in one call (STRATA_QSA_SEGMENTS) ----
// A batch window's run of one slot's proposal rows: the run reads ONE indexer key set and ONE K/V pool and its cells
// sit at successive positions (2046.. : 2046 leaves the partial tail, 2047 completes its block inside the run).  The
// grouped calls the switch makes - qsa_block_scores / qsa_block_topk / qsa_decode_attn_batch with nq = the run, and
// the run's native_qsa_indexer_append_steps - against the per-row loop verify.cpp runs without the switch.  One
// capture, one instantiation, then the exec replayed over two step sets (A, B, A) with the steps changed in place.
void segments_case(int rows, bool int8) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = 3000;
    const int64_t ID = s.idx_dim, NH = s.n_head, HD = s.head_dim, NKV = s.n_head_kv, PS = s.page_size;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t sf = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    const int64_t bases[2] = {2046, 2049};   // a run starting on a partial tail, then one starting past the block edge
    const int64_t cells = bases[1] + rows, pages = (cells + PS - 1) / PS, prow = pages * NKV * PS;
    std::mt19937 rng(41 + rows + int8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scl(0.005f, 0.03f);
    // the run's ONE indexer key set, K/V pool and page table
    std::vector<float> hp((size_t) max_blocks * ID), hd((size_t) ID), q((size_t) rows * s.idx_n_head * ID);
    for (auto& x : hp) x = nd(rng);
    for (auto& x : hd) x = nd(rng);
    for (auto& x : q) x = nd(rng) * 2.0f;
    const float* pooled = up(hp);
    const float* dead = up(hd);
    const float* d_q = up(q);
    k::QsaAttnPools pl;
    if (int8) {
        std::vector<int8_t> kq((size_t) prow * HD), vq((size_t) prow * HD);
        std::vector<uint16_t> ks((size_t) prow * (HD / 64)), vs(ks.size());
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(scl(rng));
        for (auto& x : vs) x = f2h(scl(rng));
        pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs);
    } else {
        std::vector<uint16_t> kh((size_t) prow * HD), vh(kh.size());
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
        pl.k_pool = up(kh); pl.v_pool = up(vh);
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    pl.page_table = up(table);
    std::vector<float> qc((size_t) rows * NH * HD);
    for (auto& x : qc) x = nd(rng) * 2.0f;
    const float* d_qc = up(qc);
    std::vector<int32_t> steps((size_t) rows * k::kStepCount);
    int32_t* d_steps = up(steps);
    const size_t sn = (size_t) rows * max_blocks, in = (size_t) rows * cap, an = (size_t) rows * NH * HD;
    float *sc_ref = nullptr, *sc_got = nullptr, *scr_ref = nullptr, *scr_got = nullptr;
    int32_t *id_ref = nullptr, *id_got = nullptr;
    float *at_ref = nullptr, *at_got = nullptr;
    ck(cudaMalloc(&sc_ref, sn * 4), "malloc");
    ck(cudaMalloc(&sc_got, sn * 4), "malloc");
    ck(cudaMalloc(&scr_ref, (size_t) rows * sf * 4), "malloc");
    ck(cudaMalloc(&scr_got, (size_t) rows * sf * 4), "malloc");
    ck(cudaMalloc(&id_ref, in * 4), "malloc");
    ck(cudaMalloc(&id_got, in * 4), "malloc");
    ck(cudaMalloc(&at_ref, an * 4), "malloc");
    ck(cudaMalloc(&at_got, an * 4), "malloc");
    // The engine's batch window is a captured graph whose step records change every window.  Capture the grouped
    // calls once, instantiate once, and replay the exec over two step sets - A, B, then A again on the same exec - with
    // one call per row (the loop verify.cpp runs without the switch) as the reference for each replay.  A kernel or a
    // grid that took n_bid/n_kv/width from host memory at capture time instead of reading the step record on every
    // launch fails the B (and the second A) replay.
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreate(&cs), "stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
    k::qsa_block_scores(pooled, dead, d_q, d_steps, rows, max_blocks, s, sc_got, cs);
    k::qsa_block_topk(sc_got, d_steps, rows, max_blocks, cap, s, id_got, cs);
    k::qsa_decode_attn_batch(d_qc, pl, id_got, d_steps, cap, s, scr_got, at_got, rows, cs);
    ck(cudaStreamEndCapture(cs, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    bool same = true;
    char name[128];
    for (int replay = 0; replay < 3; ++replay) {
        const int64_t base = bases[replay == 1 ? 1 : 0];   // A, B, A: the device steps change between replays
        for (int r = 0; r < rows; ++r) k::qsa_step_fill(steps.data() + (size_t) r * k::kStepCount, base + r, s);
        ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
        // the reference: one call per row, in the same stream order
        ck(cudaMemsetAsync(sc_ref, 0, sn * 4, cs), "memset");
        ck(cudaMemsetAsync(id_ref, 0, in * 4, cs), "memset");
        ck(cudaMemsetAsync(at_ref, 0, an * 4, cs), "memset");
        for (int r = 0; r < rows; ++r) {
            k::qsa_block_scores(pooled, dead, d_q + (size_t) r * s.idx_n_head * ID,
                                d_steps + (size_t) r * k::kStepCount, 1, max_blocks, s,
                                sc_ref + (size_t) r * max_blocks, cs);
            k::qsa_block_topk(sc_ref + (size_t) r * max_blocks, d_steps + (size_t) r * k::kStepCount, 1, max_blocks,
                              cap, s, id_ref + (size_t) r * cap, cs);
            k::qsa_decode_attn_batch(d_qc + (size_t) r * NH * HD, pl, id_ref + (size_t) r * cap,
                                     d_steps + (size_t) r * k::kStepCount, cap, s, scr_ref + (size_t) r * sf,
                                     at_ref + (size_t) r * NH * HD, 1, cs);
        }
        // the captured run, replayed into zeroed outputs: a node that never ran stays zero and differs
        ck(cudaMemsetAsync(sc_got, 0, sn * 4, cs), "memset");
        ck(cudaMemsetAsync(id_got, 0, in * 4, cs), "memset");
        ck(cudaMemsetAsync(at_got, 0, an * 4, cs), "memset");
        ck(cudaGraphLaunch(exec, cs), "graph launch");
        ck(cudaStreamSynchronize(cs), "sync");
        const std::vector<float> sr = down(sc_ref, sn);
        const std::vector<int32_t> ir = down(id_ref, in);
        const std::vector<float> ar = down(at_ref, an);
        const std::vector<float> sg = down(sc_got, sn);
        const std::vector<int32_t> ig = down(id_got, in);
        const std::vector<float> ag = down(at_got, an);
        for (int r = 0; r < rows; ++r) {
            const size_t nb = (size_t) steps[(size_t) r * k::kStepCount + k::kStepNBid] + 1;   // through n_bid
            const size_t w = (size_t) steps[(size_t) r * k::kStepCount + k::kStepWidth];
            same = same && std::memcmp(&sr[(size_t) r * max_blocks], &sg[(size_t) r * max_blocks], nb * 4) == 0;
            same = same && std::memcmp(&ir[(size_t) r * cap], &ig[(size_t) r * cap], w * 4) == 0;
            same = same &&
                   std::memcmp(&ar[(size_t) r * NH * HD], &ag[(size_t) r * NH * HD], (size_t) (NH * HD) * 4) == 0;
        }
    }
    std::snprintf(name, sizeof name,
                  "one slot's %d rows: captured scores/topk/attention, 3 replays over 2 step sets (%s)", rows,
                  int8 ? "int8" : "fp16");
    report(name, same);
    // the append run starts at the partial tail (2046), not at the last step set's base
    for (int r = 0; r < rows; ++r) k::qsa_step_fill(steps.data() + (size_t) r * k::kStepCount, bases[0] + r, s);
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    // the run's indexer appends - one _steps call (the one-row run's single append) against one per row; the
    // buffers must be left as the sequential calls leave them, partial tail and completed block included.  The
    // appends are eager: they are stateful, so a replay would append the cells twice.
    const int64_t tail_n = (s.idx_block - 1) * ID, max_cells = (max_blocks - 1) * s.idx_block;
    std::vector<float> raw((size_t) rows * ID), gamma((size_t) ID);
    for (auto& x : raw) x = nd(rng);
    for (auto& x : gamma) x = nd(rng);
    const float* d_raw = up(raw);
    const float* d_gamma = up(gamma);
    struct IdxSide { float* tail; float* dead; float* pooled; int32_t* bpos; };
    std::vector<float> idx_dead((size_t) ID), idx_pooled((size_t) max_blocks * ID);
    for (auto& v : idx_dead) v = nd(rng);
    for (auto& v : idx_pooled) v = nd(rng);
    auto idx_side = [&]() {   // both sides start from the same tail/dead/pooled: only the appends may differ
        IdxSide x;
        std::vector<float> t((size_t) tail_n, 0.f);
        x.tail = up(t);
        x.dead = up(idx_dead);
        x.pooled = up(idx_pooled);
        x.bpos = up(std::vector<int32_t>{0});
        return x;
    };
    const IdxSide seq = idx_side();
    const IdxSide grp = idx_side();
    const strata::kernels::RopeScaling rst;
    for (int r = 0; r < rows; ++r) {
        const k::QsaIndexerBuffers b{seq.tail, seq.dead, seq.pooled, seq.bpos};
        k::native_qsa_indexer_append(d_raw + (size_t) r * ID, d_steps + (size_t) r * k::kStepCount + k::kStepPos, 0,
                                     d_gamma, k::qsa_rms_eps(), b, s, max_cells, rst, cs);
    }
    const k::QsaIndexerBuffers gb{grp.tail, grp.dead, grp.pooled, grp.bpos};
    if (rows == 1)   // a one-row run keeps the per-row call, as verify.cpp does under the switch
        k::native_qsa_indexer_append(d_raw, d_steps + k::kStepPos, 0, d_gamma, k::qsa_rms_eps(), gb, s, max_cells,
                                     rst, cs);
    else
        k::native_qsa_indexer_append_steps(d_raw, d_steps + k::kStepPos, (int) k::kStepCount, rows, 0, d_gamma,
                                           k::qsa_rms_eps(), gb, s, max_cells, rst, cs);
    ck(cudaStreamSynchronize(cs), "sync");
    const std::vector<float> sp = down(seq.pooled, (size_t) max_blocks * ID), gp = down(grp.pooled, (size_t) max_blocks * ID);
    const std::vector<float> sd = down(seq.dead, (size_t) ID), gd = down(grp.dead, (size_t) ID);
    const std::vector<float> st = down(seq.tail, (size_t) tail_n), gt = down(grp.tail, (size_t) tail_n);
    const std::vector<int32_t> sb = down(seq.bpos, 1), gbb = down(grp.bpos, 1);
    const bool asame = std::memcmp(sp.data(), gp.data(), sp.size() * 4) == 0 &&
                       std::memcmp(sd.data(), gd.data(), sd.size() * 4) == 0 &&
                       std::memcmp(st.data(), gt.data(), st.size() * 4) == 0 && sb[0] == gbb[0];
    std::snprintf(name, sizeof name, "one slot's %d rows: native_qsa_indexer_append%s (partial tail)", rows,
                  rows == 1 ? "" : "_steps");
    report(name, asame);
    ck(cudaGraphExecDestroy(exec), "exec destroy");
    ck(cudaGraphDestroy(graph), "graph destroy");
    ck(cudaStreamDestroy(cs), "stream destroy");
}

// ---- 4. the GDN recurrence ----
void gdn_case(const std::vector<int>& ntok, bool commit) {
    const int S = 128, h_k = 16, h_v = 48;   // the artifact's geometry
    const int qk = S * h_k, C = 2 * qk + S * h_v, vd = S * h_v;
    const float eps = 1e-6f;
    int T = 0;
    for (int n : ntok) T += n;
    const int G = (int) ntok.size();
    std::mt19937 rng(31 + T * 7 + G + commit);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) T * C), gate((size_t) T * h_v), beta((size_t) T * h_v), z((size_t) T * vd), gamma(S);
    for (auto& v : h) v = nd(rng) * 0.1f;
    for (auto& v : gate) v = -(0.5f + 3.0f * (float) (rng() % 100) / 100.0f);
    for (auto& v : beta) v = (float) (rng() % 100) / 100.0f;
    for (auto& v : z) v = nd(rng);
    for (auto& v : gamma) v = 1.0f + 0.1f * nd(rng);
    const size_t st_n = (size_t) S * h_v * S;
    std::vector<std::vector<float>> st0(G);
    std::vector<float*> st_g(G), st_r(G);
    std::vector<int32_t*> keep(G);
    k::GdnGroups gg;
    gg.n = G;
    int first = 0;
    for (int i = 0; i < G; ++i) {
        st0[i].resize(st_n);
        for (auto& v : st0[i]) v = nd(rng) * 0.1f;
        st_g[i] = up(st0[i]);
        st_r[i] = up(st0[i]);
        const int32_t kp = commit ? ntok[i] : 0;   // a commit group keeps all of its rows here
        keep[i] = up(std::vector<int32_t>{kp});
        gg.state[i] = st_g[i];
        gg.keep[i] = commit ? keep[i] : nullptr;
        gg.first[i] = first;
        gg.ntok[i] = ntok[i];
        first += ntok[i];
    }
    const float* d_h = up(h);
    const float* d_gate = up(gate);
    const float* d_beta = up(beta);
    const float* d_z = up(z);
    const float* d_gm = up(gamma);
    float *d_y = nullptr, *d_yr = nullptr;
    ck(cudaMalloc(&d_y, (size_t) T * vd * 4), "malloc");
    ck(cudaMalloc(&d_yr, (size_t) T * vd * 4), "malloc");
    ck(cudaMemset(d_y, 0, (size_t) T * vd * 4), "memset");
    ck(cudaMemset(d_yr, 0, (size_t) T * vd * 4), "memset");
    k::gdn_step_norm_groups(gg, d_h, C, d_gate, d_beta, d_z, d_gm, eps, d_y, h_k, h_v, commit, nullptr);
    for (int i = 0; i < G; ++i) {
        const int f = gg.first[i];
        // the engine's per-slot call: a window group runs its rows with all outputs; a commit group (1 row) keeps them
        k::gdn_step_norm_multi(st_r[i], d_h + (size_t) f * C, C, d_gate + (size_t) f * h_v, d_beta + (size_t) f * h_v,
                               d_z + (size_t) f * vd, d_gm, eps, d_yr + (size_t) f * vd, h_k, h_v, ntok[i],
                               commit ? keep[i] : nullptr, nullptr, 0);
    }
    ck(cudaDeviceSynchronize(), "sync");
    bool same = true;
    for (int i = 0; i < G; ++i) {
        const std::vector<float> a = down(st_g[i], st_n), b = down(st_r[i], st_n);
        same = same && std::memcmp(a.data(), b.data(), st_n * 4) == 0;
        if (!commit) same = same && std::memcmp(a.data(), st0[i].data(), st_n * 4) == 0;   // a window leaves the state alone
    }
    if (!commit) {
        const std::vector<float> a = down(d_y, (size_t) T * vd), b = down(d_yr, (size_t) T * vd);
        same = same && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    }
    char name[96], rows[64] = "";
    for (int i = 0; i < G && i < 8; ++i) std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows), "%s%d", i ? "," : "", ntok[i]);
    std::snprintf(name, sizeof name, "gdn_step_norm_groups %s, groups of %s rows", commit ? "commit" : "window", rows);
    report(name, same);
}
}  // namespace

int main() {
    std::printf("batch rows parity (#2234 kernels against the per-row launches)\n");
    for (int r : {1, 2, 4, 8}) scores_case(r);
    for (int r : {1, 3, 8}) {
        attn_case(r, true);
        attn_case(r, false);
    }
    for (int r : {1, 2, 4, 8}) {
        segments_case(r, true);
        segments_case(r, false);
    }
    gdn_case({1}, false);
    gdn_case({1, 1, 1, 1}, false);
    gdn_case({2, 2, 2, 2}, false);
    gdn_case({1, 3, 2, 1, 1}, false);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, false);
    gdn_case({1}, true);
    gdn_case({1, 1, 1, 1}, true);
    gdn_case({1, 1, 1, 1, 1, 1, 1, 1}, true);
    std::printf("%s\n", bad ? "batch_rows_parity: FAILED" : "batch_rows_parity: ok");
    return bad ? 1 : 0;
}
