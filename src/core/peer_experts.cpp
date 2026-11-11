// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;   // entries per layer: MAXT tokens x top-10

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {0, device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: device 0 -> peer, 1: peer -> device 0
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        {
            On on(dev[to]);
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { On on(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    ranked_.clear();
    slot_layer_.clear();
    pending_.clear();
    if (device_ < 0) return;
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err, int64_t segment_mib) {
    close();
    if (segment_mib < 0) {
        err = "peer experts: the segment size must not be negative";
        return false;
    }
    // Segmented backing for VRAM handoff: set BEFORE the open, and the segmented path wins over the global
    // ExpertCache::set_vmm that the primary's elastic K/V turns on.  0 keeps the existing backing choice.
    cache_.set_segment_bytes(segment_mib * 1024 * 1024);
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(0); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(0); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    if (segment_mib > 0) {
        // the handoff's bookkeeping, only when it was asked for: the rank order to restore from and the layer each
        // slot was laid out for (a slot NEVER changes layer, so its blob size never changes either)
        ranked_ = ranked;
        slot_layer_.assign(pick.size(), -1);
    }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        src.note_async_read(b, refill_);   // #1237
        if (!slot_layer_.empty() && slot >= 0 && (size_t) slot < slot_layer_.size()) slot_layer_[(size_t) slot] = pr.first;
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

void PeerExperts::recount_resident() {
    int64_t n = 0;
    for (const int32_t s : res_)
        if (s >= 0) ++n;
    resident_ = n;
}

bool PeerExperts::restore_free_slots(const int32_t* primary_res, std::string& err) {
    const int64_t live = cache_.slots();
    if (live <= 0 || ranked_.empty()) return true;
    if ((int64_t) slot_layer_.size() < live) {
        err = "peer experts: the slot-to-layer table does not cover the peer cache";
        return false;
    }
    On on(device_);   // refill_ and the cache's slots are this device's: queue and sync there, not on the caller's
    // a slot is free when no live expert claims it; every claim must also point inside the backing that is left
    std::vector<uint8_t> used((size_t) live, 0);
    for (int32_t& s : res_) {
        if (s >= live) s = kNotResident;
        else if (s >= 0) used[(size_t) s] = 1;
    }
    // the startup rank, per layer, so a slot only ever gets an expert whose blob is exactly its layer's size
    std::vector<std::vector<int32_t>> cand((size_t) n_layers_);
    for (const auto& pr : ranked_)
        if (pr.first >= 0 && pr.first < n_layers_ && pr.second >= 0 && pr.second < n_expert_)
            cand[(size_t) pr.first].push_back(pr.second);
    std::vector<size_t> next((size_t) n_layers_, 0);
    std::vector<uint8_t> picked(res_.size(), 0);
    std::vector<std::pair<int32_t, int32_t>> fills;   // (residency index, slot): invisible until the copies land
    const auto& lay = strata::kernels::cpu::expert_layout();
    bool ok = true, stop = false;
    for (int64_t s = 0; s < live && !stop; ++s) {
        if (used[(size_t) s]) continue;
        const int32_t layer = slot_layer_[(size_t) s];
        if (layer < 0 || layer >= n_layers_) {
            err = "peer experts: slot " + std::to_string(s) + " has no layer";
            ok = false;
            stop = true;
            break;
        }
        auto& c = cand[(size_t) layer];
        size_t& at = next[(size_t) layer];
        const uint64_t bytes = lay.blob_bytes(layer);
        if ((uint64_t) cache_.slot_room((int32_t) s) < bytes) {
            err = "peer experts: slot " + std::to_string(s) + " is too small for layer " + std::to_string(layer);
            ok = false;
            stop = true;
            break;
        }
        bool filled = false;
        std::string skipped;   // a ranked expert the source cannot give us; the next one may still fit
        while (at < c.size()) {
            const int32_t e = c[at];
            const size_t idx = (size_t) ((int64_t) layer * n_expert_ + e);
            if (res_[idx] >= 0 || picked[idx] || (primary_res != nullptr && primary_res[idx] >= 0)) {
                ++at;
                continue;
            }
            const uint8_t* b = src_ != nullptr ? src_->blob(layer, e) : nullptr;
            if (b == nullptr) {
                skipped = "peer experts: no blob for layer " + std::to_string(layer) + ", expert " + std::to_string(e) +
                          " to restore slot " + std::to_string(s);
                ++at;
                continue;
            }
            if (!cache_.fill_slot((int32_t) s, b, refill_, err, (int64_t) bytes)) {
                err = "peer experts: restore fill failed: " + err;
                ok = false;
                stop = true;   // the stream may be in error: queue nothing more
                break;
            }
            src_->note_async_read(b, refill_);
            picked[idx] = 1;
            fills.emplace_back((int32_t) idx, (int32_t) s);
            used[(size_t) s] = 1;
            ++at;
            filled = true;
            break;
        }
        if (!filled && !skipped.empty() && !stop) {
            err = skipped;   // honest: the slot stays empty and this restore is partial
            ok = false;
        }
    }
    if (fills.empty()) return ok;   // nothing queued: the refill stream was already drained by the resize
    // residency is published only after every queued copy has completed
    std::string serr;
    if (!ck(cudaStreamSynchronize(refill_), "restore fill", serr)) {
        if (!err.empty()) err += "; ";
        err += serr;
        ok = false;
    } else {
        for (const auto& [idx, slot] : fills) res_[(size_t) idx] = slot;
    }
    return ok;
}

bool PeerExperts::resize_reserve(int64_t reserve_mib, const int32_t* primary_res, std::string& err) {
    if (device_ < 0) {
        err = "peer experts: no peer tier is open";
        return false;
    }
    if (reserve_mib < 0) {
        err = "peer experts: the reserve must not be negative";
        return false;
    }
    if (!cache_.segmented()) {
        err = "peer experts: the peer cache is not segmented (open it with a segment size for VRAM handoff)";
        return false;
    }
    if ((int64_t) slot_layer_.size() != cache_.full_slots()) {
        err = "peer experts: the slot-to-layer table does not cover the peer cache";
        return false;
    }
    // Quiesce BOTH streams before any unmap/reuse: the verification window's peer work and the adaptive tier's
    // queued refills.  The refill copies have landed by here, so the pending table can be published below.
    {
        On on(device_);
        if (!ck(cudaStreamSynchronize(stream_), "resize: peer work", err) ||
            !ck(cudaStreamSynchronize(refill_), "resize: pending refill", err))
            return false;
    }
    const int64_t live_before = cache_.slots();
    for (const auto& [idx, slot] : pending_)
        if (idx >= 0 && (size_t) idx < res_.size() && slot >= 0 && slot < live_before && res_[(size_t) idx] < 0)
            res_[(size_t) idx] = slot;
    pending_.clear();
    for (int32_t& s : res_)
        if (s >= live_before) s = kNotResident;   // defensive: nothing may point past the backing

    if (reserve_mib == 0) {   // idle restore: map the arena back towards its startup size, then refill the holes
        const int64_t want = cache_.full_bytes();
        bool ok = true;
        if (cache_.mapped_bytes() < want) {
            std::string gerr;
            bool grown = false;
            {
                On on(device_);   // the arena's context (and the device cudaDeviceSynchronize inside grow)
                // The arena's last segment is rounded up past full_bytes(); `grow` only maps a segment whose
                // whole size fits in the target, so ask past that rounding - it still stops at the arena.
                grown = cache_.grow(want + cache_.segment_bytes(), gerr);
            }
            if (grown && cache_.slots() < cache_.full_slots()) {
                gerr = "the driver backed only " + std::to_string(cache_.slots()) + " of " +
                       std::to_string(cache_.full_slots()) + " slots";
                grown = false;
            }
            if (!grown) {
                err = "peer experts: the peer cache could not be mapped back in full: " +
                      (gerr.empty() ? std::string("the driver has no VRAM for another segment") : gerr);
                ok = false;
            }
        }
        const int64_t live = cache_.slots();
        for (int32_t& s : res_)
            if (s >= live) s = kNotResident;
        std::string rerr;
        if (!restore_free_slots(primary_res, rerr)) {
            if (!ok) err += "; then " + rerr;
            else err = rerr;
            ok = false;
        }
        recount_resident();
        return ok;
    }

    // positive: leave at least `reserve_mib` MiB of physical VRAM free on the peer card
    size_t free_b = 0, total_b = 0;
    {
        On on(device_);
        if (!ck(cudaMemGetInfo(&free_b, &total_b), "resize: cudaMemGetInfo", err)) return false;
    }
    const int64_t reserve = reserve_mib * 1024 * 1024;
    if ((int64_t) free_b >= reserve) {
        recount_resident();
        return true;
    }
    const int64_t need = reserve - (int64_t) free_b;
    const int64_t mapped = cache_.mapped_bytes();
    const int64_t keep = cache_.keep_bytes_for_free(need);
    if (keep == 0 && need > mapped) {
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "peer experts: cannot leave %lld MiB free on GPU %d: the peer cache backs only %.1f MiB and "
                      "%.1f MiB is already free", (long long) reserve_mib, device_,
                      (double) mapped / 1048576.0, (double) free_b / 1048576.0);
        err = buf;
        recount_resident();
        return false;
    }
    const int64_t new_live = cache_.slots_within(keep);
    for (int32_t& s : res_)
        if (s >= new_live) s = kNotResident;   // these slots are about to be unmapped: their experts become CPU rows
    bool shrunk = false;
    {
        On on(device_);   // the arena's context (and the device cudaDeviceSynchronize inside shrink)
        shrunk = cache_.shrink(keep, err);
    }
    if (!shrunk) {
        if (err.empty()) err = "peer experts: the peer cache could not release VRAM";
        const int64_t live = cache_.slots();
        for (int32_t& s : res_)
            if (s >= live) s = kNotResident;
        recount_resident();
        return false;
    }
    for (int32_t& s : res_)   // the backing shrank exactly as planned; clamp again so no path can dangle
        if (s >= cache_.slots()) s = kNotResident;
    recount_resident();
    return true;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    static const bool direct_env = [] { const char* v = std::getenv("STRATA_PEER_DIRECT"); return v == nullptr || std::atoi(v) != 0; }();
    const bool direct = direct_env && out != nullptr;
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0 || slot >= cache_.slots()) {
            err = "peer experts: an entry was planned for the peer but is not resident";
            return false;
        }
        m.ptr[groups] = (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
        !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
        return false;
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& f = lay.fmt[(size_t) layer];
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= cache_.slots()) r1[e] = kNotResident;   // a resize released the slot: no dangling victim
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    On on(device_);
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || slot >= cache_.slots() || b == nullptr) continue;
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    return ck(cudaEventRecord(refill_ev_, refill_), "refill event", err);
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_)
        if (slot >= 0 && slot < cache_.slots()) res_[(size_t) i] = slot;
    pending_.clear();
}

}  // namespace strata::core
