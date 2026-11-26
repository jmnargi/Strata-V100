// src/hy4/hy4_dfly.cu - Hy4-flash's DFly2 drafter (hy4_dfly.hpp): the drafter GGUF, the tap, and the block pass.
//
// The GGUF is read directly (general.architecture hy_v4_flash_dfly): its projections are Q8_0 and its vectors F32, used as
// stored. `mlp.down_proj` in this artifact carries the checkpoint's [E][ffn] order (its ggml dims record the transpose but
// the bytes are the checkpoint's row-major [out][in]); every other matrix matches the checkpoint's [out][in] in ggml's
// reverse order. The block pass and the tap follow Basalt's (src/models/hy4flash/dfly.cpp), with the target's cache form:
// F16 rows, or kv_store's int8 rows when the target runs --kv q8, so the drafter rounds its own rows exactly as the target
// would and never writes into the target's cache.
#include "hy4_dfly.hpp"

#include "hy4_kernels.cuh"
#include "../ds4/ds4_iq.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// (ggml-quants, linked into the engine: the split kv_b's transposed half is dequantized and requantized row by row)
extern "C" void dequantize_row_q8_0(const void* x, float* y, int64_t k);
extern "C" void quantize_row_q8_0_ref(const float* x, void* y, int64_t k);

namespace hy4 {

namespace iq = strata::ds4::gguf::iq;

namespace {

constexpr int kQ8_0 = 8, kF32 = 0;

struct TensorRef {
    uint64_t off = 0;
    int type = -1;
    std::vector<uint64_t> ne;   // ggml order: ne0 innermost
};

struct Gguf {
    uint64_t data_start = 0, file_bytes = 0;
    std::map<std::string, TensorRef> tensors;
    std::map<std::string, int64_t> ints;
    std::map<std::string, double> floats;
    std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<int64_t>> arrays;
};

bool fail(std::string& err, const std::string& m) {
    err = m;
    return false;
}

// ---- GGUF v3 ------------------------------------------------------------------------------------------------------------
struct Reader {
    std::ifstream f;
    uint64_t left = 0;
    bool ok = true;

    bool bytes(void* p, size_t n) {
        if (!ok || left < n) return ok = false;
        f.read((char*) p, (std::streamsize) n);
        if (!f) return ok = false;
        left -= n;
        return true;
    }
    template <class T>
    bool val(T& x) {
        return bytes(&x, sizeof(T));
    }
    bool str(std::string& s) {
        uint64_t n = 0;
        if (!val(n) || n > (1ull << 31)) return ok = false;
        s.resize((size_t) n);
        return n == 0 || bytes(s.data(), (size_t) n);
    }
    bool value(uint32_t type, int64_t& i, double& d, std::string& s, std::vector<int64_t>& a, bool keep) {
        switch (type) {
            case 0: { uint8_t x; if (!val(x)) return false; i = x; return true; }
            case 1: { int8_t x; if (!val(x)) return false; i = x; return true; }
            case 2: { uint16_t x; if (!val(x)) return false; i = x; return true; }
            case 3: { int16_t x; if (!val(x)) return false; i = x; return true; }
            case 4: { uint32_t x; if (!val(x)) return false; i = x; return true; }
            case 5: { int32_t x; if (!val(x)) return false; i = x; return true; }
            case 6: { float x; if (!val(x)) return false; d = x; i = (int64_t) x; return true; }
            case 7: { uint8_t x; if (!val(x)) return false; i = x; return true; }
            case 8: return str(s);
            case 9: {
                uint32_t et = 0;
                uint64_t n = 0;
                if (!val(et) || !val(n) || n > (1ull << 31)) return ok = false;
                const bool num = et <= 7 || et == 10 || et == 11 || et == 12;
                if (keep && num) a.reserve((size_t) n);
                for (uint64_t k = 0; k < n; ++k) {
                    int64_t x = 0;
                    double dd = 0;
                    std::string ss;
                    std::vector<int64_t> aa;
                    if (!value(et, x, dd, ss, aa, false)) return false;
                    if (keep && num) a.push_back(et == 6 || et == 12 ? (int64_t) dd : x);
                }
                return true;
            }
            case 10: { uint64_t x; if (!val(x)) return false; i = (int64_t) x; return true; }
            case 11: { int64_t x; if (!val(x)) return false; i = x; return true; }
            case 12: { double x; if (!val(x)) return false; d = x; i = (int64_t) x; return true; }
            default: return ok = false;
        }
    }
};

bool read_gguf(const std::string& path, Gguf& g, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(err, path + ": cannot open");
    f.seekg(0, std::ios::end);
    g.file_bytes = (uint64_t) f.tellg();
    f.seekg(0, std::ios::beg);
    Reader r{std::move(f), g.file_bytes, true};
    uint32_t magic = 0, ver = 0;
    uint64_t nt = 0, nkv = 0;
    if (!r.val(magic) || !r.val(ver) || !r.val(nt) || !r.val(nkv) || magic != 0x46554747u)
        return fail(err, path + ": not a GGUF file");
    if (ver != 3) return fail(err, path + ": GGUF v" + std::to_string(ver) + ", v3 expected");
    if (nt > (1u << 20) || nkv > (1u << 20)) return fail(err, path + ": GGUF counts out of range");
    for (uint64_t k = 0; k < nkv; ++k) {
        std::string key;
        uint32_t type = 0;
        if (!r.str(key) || !r.val(type)) return fail(err, path + ": truncated metadata");
        int64_t i = 0;
        double d = 0;
        std::string s;
        std::vector<int64_t> a;
        if (!r.value(type, i, d, s, a, true)) return fail(err, path + ": truncated metadata " + key);
        if (type == 8) g.strs[key] = std::move(s);
        else if (type == 9) g.arrays[key] = std::move(a);
        else if (type == 6 || type == 12) g.floats[key] = d;
        else g.ints[key] = i;
    }
    uint64_t align = 32;
    if (auto it = g.ints.find("general.alignment"); it != g.ints.end() && it->second > 0) align = (uint64_t) it->second;
    for (uint64_t k = 0; k < nt; ++k) {
        std::string name;
        uint32_t nd = 0;
        if (!r.str(name) || !r.val(nd) || nd > 4) return fail(err, path + ": truncated tensor table");
        TensorRef t;
        t.ne.resize(nd);
        for (uint32_t j = 0; j < nd; ++j)
            if (!r.val(t.ne[j])) return fail(err, path + ": truncated tensor table");
        uint32_t type = 0;
        if (!r.val(type) || !r.val(t.off)) return fail(err, path + ": truncated tensor table");
        t.type = (int) type;
        g.tensors[name] = std::move(t);
    }
    const uint64_t pos = (uint64_t) r.f.tellg();
    g.data_start = (pos + align - 1) / align * align;
    if (g.data_start > g.file_bytes) return fail(err, path + ": truncated data section");
    return true;
}

// ---- the drafter's tensors ----------------------------------------------------------------------------------------------
enum Kind { F32V, Q8M };

struct Spec {
    std::string name;
    std::string file;   // the file's tensor (the canonical name unless the drafter is AngelSlim's llama.cpp build)
    std::string alt;    // the second tensor of a split kv_b (AngelSlim stores attn_k_b and attn_v_b apart)
    uint64_t a = 0, b = 0;   // Q8M: a = k (inputs), b = n (outputs); F32V: a = values
    Kind kind = F32V;
    bool split_kv = false;
    uint64_t elems() const { return kind == Q8M ? a * b : a; }
    uint64_t bytes() const { return kind == Q8M ? elems() / 32 * 34 : elems() * 4; }
};

std::string lp(const std::string& p, const char* n) { return "model.dfly_layers." + p + n; }

uint64_t elems_of(const TensorRef& t) {
    uint64_t n = 1;
    for (uint64_t d : t.ne) n *= d;
    return n;
}

Spec q8m(const std::string& name, uint64_t in, uint64_t out) {
    Spec s;
    s.name = name;
    s.kind = Q8M;
    s.a = in;
    s.b = out;
    return s;
}
Spec f32v(const std::string& name, uint64_t n) {
    Spec s;
    s.name = name;
    s.kind = F32V;
    s.a = n;
    return s;
}

std::vector<Spec> specs(const DflyDims& d) {
    const uint64_t E = (uint64_t) d.E, H = (uint64_t) d.heads, F = (uint64_t) d.ffn, Q = (uint64_t) d.q_lora,
                   C = (uint64_t) d.hc_ffn;
    std::vector<Spec> v;
    for (int i = 0; i < d.layers; ++i) {
        const std::string p = std::to_string(i) + ".";
        v.push_back(f32v(lp(p, "input_layernorm.weight"), E));
        v.push_back(f32v(lp(p, "post_attention_layernorm.weight"), E));
        v.push_back(q8m(lp(p, "mlp.gate_proj.weight"), E, F));
        v.push_back(q8m(lp(p, "mlp.up_proj.weight"), E, F));
        v.push_back(q8m(lp(p, "mlp.down_proj.weight"), F, E));
        v.push_back(q8m(lp(p, "self_attn.q_a_proj.weight"), E, Q));
        v.push_back(f32v(lp(p, "self_attn.q_a_layernorm.weight"), Q));
        v.push_back(q8m(lp(p, "self_attn.q_b_proj.weight"), Q, H * 256));
        v.push_back(q8m(lp(p, "self_attn.kv_a_proj_with_mqa.weight"), E, kDflyRowWidth));
        v.push_back(f32v(lp(p, "self_attn.kv_a_layernorm.weight"), kDflyRank));
        v.push_back(q8m(lp(p, "self_attn.kv_b_proj.weight"), kDflyRank, H * kDflyKvbHead));
        v.push_back(q8m(lp(p, "self_attn.linear_gate.weight"), E, H * kDflyV));
        v.push_back(q8m(lp(p, "self_attn.o_proj.weight"), H * kDflyV, E));
        v.push_back(f32v(lp(p, "self_attn.post_proj_norm.weight"), E));
        v.push_back(f32v(lp(p, "self_attn.learnable_sink_param"), H));
    }
    v.push_back(f32v("model.dfly_layers.final_norm.weight", E));
    v.push_back(q8m("model.dfly_layers.mtp_fusion.eh_proj.weight", 2 * E, E));
    v.push_back(f32v("model.dfly_layers.mtp_fusion.enorm.weight", E));
    v.push_back(f32v("model.dfly_layers.mtp_fusion.hnorm.weight", E));
    v.push_back(f32v("model.dfly_layers.mtp_fusion.gate", E));
    v.push_back(q8m("model.dfly_layers.hidden_correction.gate_proj.weight", 2 * E, C));
    v.push_back(q8m("model.dfly_layers.hidden_correction.up_proj.weight", 2 * E, C));
    v.push_back(q8m("model.dfly_layers.hidden_correction.down_proj.weight", C, E));
    v.push_back(f32v("model.dfly_layers.hidden_correction.hidden_norm.weight", E));
    v.push_back(f32v("model.dfly_layers.hidden_correction.embed_norm.weight", E));
    v.push_back(f32v("model.dfly_layers.confidence_head.proj.weight", E));
    v.push_back(f32v("model.dfly_layers.confidence_head.proj.bias", 1));
    return v;
}

uint64_t drb(bool q8) { return q8 ? k::kv_q8_row_bytes(kDflyRank, kDflyRope) : (uint64_t) kDflyRowWidth * 2; }

// AngelSlim's own drafter GGUF (general.architecture hy_v4_next_dfly, llama.cpp's tensor names) carries the same weights
// as the HF-named build this file was written for, under other names, with kv_b_proj split into a transposed attn_k_b and
// an attn_v_b. `alias_next` maps a canonical name to the file's (`alt` the split's second tensor); false when it has none.
bool alias_next(const std::string& canon, std::string& file, std::string& alt, bool& split) {
    static const std::pair<const char*, const char*> top[] = {
        {"final_norm.weight", "output_norm.weight"},
        {"mtp_fusion.eh_proj.weight", "dfly_fusion_eh.weight"},
        {"mtp_fusion.enorm.weight", "dfly_fusion_enorm.weight"},
        {"mtp_fusion.hnorm.weight", "dfly_fusion_hnorm.weight"},
        {"mtp_fusion.gate", "dfly_fusion_gate.weight"},
        {"hidden_correction.gate_proj.weight", "dfly_correction_gate.weight"},
        {"hidden_correction.up_proj.weight", "dfly_correction_up.weight"},
        {"hidden_correction.down_proj.weight", "dfly_correction_down.weight"},
        {"hidden_correction.hidden_norm.weight", "dfly_correction_hnorm.weight"},
        {"hidden_correction.embed_norm.weight", "dfly_correction_enorm.weight"},
        {"confidence_head.proj.weight", "dfly_confidence.weight"},
        {"confidence_head.proj.bias", "dfly_confidence.bias"},
    };
    static const std::pair<const char*, const char*> lay[] = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
        {"self_attn.q_a_proj.weight", "attn_q_a.weight"},
        {"self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight"},
        {"self_attn.q_b_proj.weight", "attn_q_b.weight"},
        {"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
        {"self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight"},
        {"self_attn.kv_b_proj.weight", "attn_k_b.weight"},
        {"self_attn.linear_gate.weight", "attn_gate.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.post_proj_norm.weight", "post_attention_norm.weight"},
        {"self_attn.learnable_sink_param", "attn_sinks.weight"},
    };
    split = false;
    const std::string pre = "model.dfly_layers.";
    if (canon.compare(0, pre.size(), pre) != 0) return false;
    const std::string rest = canon.substr(pre.size());
    for (const auto& e : top)
        if (rest == e.first) {
            file = e.second;
            return true;
        }
    const size_t dot = rest.find('.');
    if (dot == std::string::npos || dot == 0) return false;
    const std::string idx = rest.substr(0, dot);
    for (char ch : idx)
        if (ch < '0' || ch > '9') return false;
    const std::string part = rest.substr(dot + 1);
    for (const auto& e : lay)
        if (part == e.first) {
            file = "blk." + idx + "." + e.second;
            if (part == "self_attn.kv_b_proj.weight") {
                split = true;
                alt = "blk." + idx + ".attn_v_b.weight";
            }
            return true;
        }
    return false;
}

uint64_t scratch_bytes_for(const DflyDims& d, bool kv_q8, int sel_cap) {
    const uint64_t R = (uint64_t) d.block, E = (uint64_t) d.E, H = (uint64_t) d.heads, F = (uint64_t) d.ffn,
                   C = (uint64_t) d.hc_ffn, Q = (uint64_t) d.q_lora;
    const uint64_t rb = drb(kv_q8);
    const uint64_t maxk = std::max({2 * E, Q, F, C, (uint64_t) H * kDflyV});
    uint64_t s = 0;
    s += (uint64_t) kDflyTapRows * E * 4;                                   // the tap's hidden rows
    s += (uint64_t) d.layers * kDflyTapRows * (uint64_t) sel_cap * 4;       // ... its selections
    s += 6 * R * E * 4;                                                     // X, xn, o, o2, mlp, z
    s += R * (2 * Q + (uint64_t) H * 256 + kDflyRowWidth + (uint64_t) H * kDflyRank + (uint64_t) H * kDflyRowWidth +
              (uint64_t) H * kDflyRank + 2 * (uint64_t) H * kDflyV) * 4;
    s += 3 * R * F * 4;                                                     // g, u, h
    s += (2 * E + E + E + E + 3 * C + E + E + (uint64_t) d.vocab + 1 + 2) * 4;
    s += ((uint64_t) sel_cap + R) * rb;                                     // ctx
    s += R * ((uint64_t) sel_cap + R) * 4 + 2 * R * 4;                      // keys, nk, pos
    s += R * maxk + R * maxk / 32 * 4;                                      // the projections' int8 inputs
    s += k::attn_ws_floats((int) R, (int) H, kDflyRank, (int) (sel_cap + R)) * 4;
    return s;
}

bool parse(const std::string& path, const DflyTarget& target, DflyDims& d, Gguf& g, std::vector<Spec>& sp, std::string& err) {
    if (!read_gguf(path, g, err)) return false;
    const std::string arch = g.strs.count("general.architecture") ? g.strs["general.architecture"] : "";
    // hy_v4_flash_dfly: the HF-named build this file was written for (Maya-S's drafter). hy_v4_next_dfly: AngelSlim's own
    // llama.cpp build (hy4-flash_dfly_Q8_0.gguf), whose tensors alias_next maps.
    const bool next = arch == "hy_v4_next_dfly";
    if (!next && arch != "hy_v4_flash_dfly")
        return fail(err, path + ": architecture " + arch + ", not hy_v4_flash_dfly or hy_v4_next_dfly");
    const std::string ap = arch + ".";
    auto gi = [&](const char* k, int64_t dflt) {
        auto it = g.ints.find(ap + std::string(k));
        return it == g.ints.end() ? dflt : it->second;
    };
    auto gs = [&](const char* k) -> std::string {
        auto it = g.strs.find(ap + std::string(k));
        return it == g.strs.end() ? std::string() : it->second;
    };
    auto ga = [&](const char* k) -> std::vector<int64_t> {
        auto it = g.arrays.find(ap + std::string(k));
        return it == g.arrays.end() ? std::vector<int64_t>() : it->second;
    };
    if (gs("mtp_fusion") != "slot0") return fail(err, path + ": dfly mtp_fusion is not slot0");
    if (gi("hidden_correction", 1) != 1) return fail(err, path + ": dfly without hidden correction is not run");
    d.layers = (int) gi("block_count", 0);
    {
        auto it = g.ints.find(next ? "tokenizer.ggml.mask_token_id" : ap + "mask_token_id");
        d.mask_id = it == g.ints.end() ? -1 : (int) it->second;
    }
    d.topk = (int) gi(next ? "target_index_top_k" : "target_index_topk", 2048);
    d.block = (int) gi("block_size", 6);
    if (d.block < 2 || d.block > kDflyMaxBlock)
        return fail(err, path + ": block_size " + std::to_string(d.block) + ", not 2.." + std::to_string(kDflyMaxBlock));
    for (int64_t x : ga("target_kv_layers")) d.kv_layers.push_back((int) x);
    for (int64_t x : ga("target_index_layers")) d.idx_layers.push_back((int) x);
    if (d.layers < 1 || d.layers > kDflyMaxLayers || (int) d.kv_layers.size() != d.layers ||
        (int) d.idx_layers.size() != d.layers || d.mask_id < 0)
        return fail(err, path + ": block_count, target_kv_layers, target_index_layers or mask_token_id are missing or disagree");
    auto fname = [&](const std::string& n) -> std::string {
        if (!next) return n;
        std::string f, a;
        bool split = false;
        return alias_next(n, f, a, split) ? f : std::string();
    };
    auto mj = [&](const std::string& n, TensorRef& t) -> bool {
        const std::string f = fname(n);
        if (f.empty()) return fail(err, "dfly: " + n + " has no name in this file's layout");
        auto it = g.tensors.find(f);
        if (it == g.tensors.end()) return fail(err, "dfly: no tensor " + f);
        t = it->second;
        return true;
    };
    TensorRef fn, gp, qa, lg, hg;
    if (!mj("model.dfly_layers.final_norm.weight", fn) || !mj(lp("0.", "mlp.gate_proj.weight"), gp) ||
        !mj(lp("0.", "self_attn.q_a_proj.weight"), qa) || !mj(lp("0.", "self_attn.linear_gate.weight"), lg) ||
        !mj("model.dfly_layers.hidden_correction.gate_proj.weight", hg))
        return false;
    if (fn.ne.size() != 1 || gp.ne.size() != 2 || qa.ne.size() != 2 || lg.ne.size() != 2 || hg.ne.size() != 2)
        return fail(err, path + ": the drafter's anchor tensors are not shaped as expected");
    d.E = (int) fn.ne[0];
    d.ffn = (int) gp.ne[1];
    d.q_lora = (int) qa.ne[1];
    d.heads = (int) (lg.ne[1] / 256);
    d.hc_ffn = (int) hg.ne[1];
    if (d.E < 1 || d.heads < 1 || d.heads > 128 || d.ffn < 1 || d.q_lora < 1 || d.hc_ffn < 1)
        return fail(err, path + ": drafter dims out of range");
    if (gi("embedding_length", d.E) != d.E) return fail(err, path + ": embedding_length disagrees with final_norm");
    if (gi("attention.head_count", d.heads) != d.heads) return fail(err, path + ": attention.head_count disagrees");
    if (gi("attention.q_lora_rank", d.q_lora) != d.q_lora) return fail(err, path + ": q_lora_rank disagrees");
    const int kv_rank = (int) gi("attention.kv_lora_rank", kDflyRank);
    if (kv_rank != kDflyRank) return fail(err, path + ": kv_lora_rank " + std::to_string(kv_rank) + ", not 512");
    // what the target must agree on
    if (d.E != target.n_embd) return fail(err, "dfly: the drafter's width is not the target's");
    if (target.n_vocab < 1) return fail(err, "dfly: the target's vocabulary is empty");
    if (d.mask_id >= target.n_vocab) return fail(err, "dfly: the drafter's mask token is outside the target's vocabulary");
    if (d.topk != target.topk) return fail(err, "dfly: target_index_topk is not the target's indexer width");
    const int nl = (int) target.heads.size();
    if (nl < 1 || (int) target.sparse_indexed.size() < nl) return fail(err, "dfly: the target's layer table is missing");
    for (int i = 0; i < d.layers; ++i) {
        const int k = d.kv_layers[(size_t) i], x = d.idx_layers[(size_t) i];
        if (k < 0 || k >= nl || x < 0 || x >= nl || !target.sparse_indexed[(size_t) k] ||
            !target.sparse_indexed[(size_t) x] || target.heads[(size_t) k] != d.heads)
            return fail(err, "dfly: draft layer " + std::to_string(i) +
                                 " names a target layer that is not a sparse one, or heads disagree");
    }
    d.vocab = target.n_vocab;
    d.rms_eps = target.rms_eps;
    // the rotary base: the drafter's own key when it has one (AngelSlim's build: 1e7), else the target's
    {
        auto it = g.floats.find(ap + "rope.freq_base");
        d.rope_base = (float) (it == g.floats.end() ? (double) target.rope_base : it->second);
    }
    sp = specs(d);
    for (Spec& s : sp) {
        s.file = s.name;
        if (next) {
            if (!alias_next(s.name, s.file, s.alt, s.split_kv))
                return fail(err, "dfly: " + s.name + " has no name in this file's layout");
        }
        auto it = g.tensors.find(s.file);
        if (it == g.tensors.end()) return fail(err, "dfly: no tensor " + s.file);
        const TensorRef& t = it->second;
        if (t.type != (s.kind == Q8M ? kQ8_0 : kF32))
            return fail(err, "dfly: " + s.file + " is not " + (s.kind == Q8M ? "Q8_0" : "F32"));
        bool shape = false;
        uint64_t file_bytes = s.bytes();
        if (s.split_kv) {
            // the split: attn_k_b is [192 nope][512 rank][heads] (out contiguous, so it is read transposed), attn_v_b
            // [512 rank][256 v][heads] (rank contiguous: its rows copy straight in)
            auto ia = g.tensors.find(s.file), ib = g.tensors.find(s.alt);
            if (ib == g.tensors.end()) return fail(err, "dfly: no tensor " + s.alt);
            const TensorRef& ka = ia->second;
            const TensorRef& vb = ib->second;
            shape = ka.ne.size() == 3 && ka.ne[0] == kDflyNope && ka.ne[1] == kDflyRank && ka.ne[2] == s.b / kDflyKvbHead &&
                    vb.ne.size() == 3 && vb.ne[0] == kDflyRank && vb.ne[1] == kDflyV && vb.ne[2] == s.b / kDflyKvbHead &&
                    vb.type == kQ8_0;
            if (!shape) return fail(err, "dfly: " + s.file + " / " + s.alt + " are not shaped as a split kv_b");
            file_bytes = (uint64_t) ka.ne[0] * ka.ne[1] * ka.ne[2] / 32 * 34;
            if (ka.off + file_bytes > g.file_bytes - g.data_start ||
                vb.off + (uint64_t) vb.ne[0] * vb.ne[1] * vb.ne[2] / 32 * 34 > g.file_bytes - g.data_start)
                return fail(err, "dfly: " + s.file + " runs past the file");
            continue;
        }
        if (s.kind == F32V) shape = elems_of(t) == s.a && t.ne.size() <= 2;
        else shape = t.ne.size() == 2 && t.ne[0] == s.a && t.ne[1] == s.b;             // ggml: [in][out]
        if (!shape) return fail(err, "dfly: " + s.file + " has another shape than the drafter's");
        if (t.off + file_bytes > g.file_bytes - g.data_start) return fail(err, "dfly: " + s.file + " runs past the file");
    }
    return true;
}

// ---- kernels ------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float dfly_block_sum(float s) {
    __shared__ float part[8];
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) part[wp] = s;
    __syncthreads();
    if (threadIdx.x != 0) return 1e30f;   // (only thread 0 uses the sum)
    float a = 0.0f;
    for (int i = 0; i < 8; ++i) a += part[i];
    return a;
}

// q_abs[vec][c] = sum_{d < 192} q[vec][d] * kv_b[(head * 448 + d)][c]: the query's nope part carried to the latent through
// the key half of kv_b (Q8_0 rows of 512). One block an output c and a block vector.
__global__ void absorb_kernel(const uint8_t* __restrict__ kvb, const float* __restrict__ q, int heads,
                              float* __restrict__ qabs) {
    const int c = blockIdx.x, vec = blockIdx.y, h = vec % heads;
    const float* qv = q + (size_t) vec * 256;
    const int cb = c >> 5, ci = c & 31;
    float s = 0.0f;
    for (int d = threadIdx.x; d < kDflyNope; d += blockDim.x) {
        const uint8_t* row = kvb + ((size_t) h * kDflyKvbHead + (size_t) d) * (kDflyRank / 32 * 34);
        const uint8_t* b = row + (size_t) cb * 34;
        const float sc = __half2float(*(const __half*) b);
        const int8_t qq = (int8_t) b[2 + ci];
        s = fmaf(sc * (float) qq, qv[d], s);
    }
    s = dfly_block_sum(s);
    if (threadIdx.x == 0) qabs[(size_t) vec * kDflyRank + c] = s;
}

// out[vec][v] = sum_{c < 512} olat[vec][c] * kv_b[(head * 448 + 192 + v)][c]: the value half, used absorbed.
__global__ void expand_kernel(const uint8_t* __restrict__ kvb, const float* __restrict__ olat, int heads,
                              float* __restrict__ out) {
    const int v = blockIdx.x, vec = blockIdx.y, h = vec % heads;
    const uint8_t* w = kvb + ((size_t) h * kDflyKvbHead + (size_t) (kDflyNope + v)) * (kDflyRank / 32 * 34);
    const float* x = olat + (size_t) vec * kDflyRank;
    float s = 0.0f;
    for (int i = threadIdx.x; i < kDflyRank; i += blockDim.x) {
        const uint8_t* b = w + (size_t) (i >> 5) * 34;
        const float sc = __half2float(*(const __half*) b);
        const int8_t qq = (int8_t) b[2 + (i & 31)];
        s = fmaf(sc * (float) qq, x[i], s);
    }
    s = dfly_block_sum(s);
    if (threadIdx.x == 0) out[(size_t) vec * kDflyV + v] = s;
}

__global__ void gate_add_kernel(const float* __restrict__ base, const float* __restrict__ fused, const float* __restrict__ g,
                                int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[i] + g[i] * fused[i];
}

// the block's per-row key lists: row t sees the selected rows [0, nctx) and its own [nctx, nctx + t]
__global__ void keys_fill_kernel(int32_t* __restrict__ keys, int ldk, int32_t* __restrict__ nk, int nctx, int rows) {
    const int t = blockIdx.x;
    if (t >= rows) return;
    int32_t* k = keys + (size_t) t * ldk;
    for (int i = threadIdx.x; i <= nctx + t; i += blockDim.x) k[i] = i;
    if (threadIdx.x == 0) nk[t] = nctx + t + 1;
}

// the target's selected rows, raw cache rows copied in their own format (row_bytes each)
__global__ void gather_rows_kernel(const uint8_t* __restrict__ cache, const int32_t* __restrict__ cells, int n,
                                   uint8_t* __restrict__ dst, int row_bytes, int ring) {
    const int j = blockIdx.x;
    if (j >= n) return;
    int p = cells ? cells[j] : j;
    if (ring > 0) p %= ring;
    const uint8_t* s = cache + (size_t) p * row_bytes;
    uint8_t* d = dst + (size_t) j * row_bytes;
    for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
}

}  // namespace

// =========================================================================================================================
struct Dfly::Impl {
    DflyDims d;
    DflyTarget target;
    int sel_cap = 0;
    int64_t ctx = 0;
    int device = 0;
    Head head;
    Embed embed;

    std::vector<void*> walloc;
    std::unordered_map<std::string, void*> wslot;
    struct Layer {
        const void *q_a = nullptr, *q_b = nullptr, *kv_a = nullptr, *kv_b = nullptr, *gate = nullptr, *o = nullptr;
        const void *mlp_gate = nullptr, *mlp_up = nullptr, *mlp_down = nullptr;
        const float *in_ln = nullptr, *q_a_ln = nullptr, *kv_a_ln = nullptr, *post_proj_ln = nullptr, *post_attn_ln = nullptr,
                    *sink = nullptr;
    };
    std::vector<Layer> layer;
    const void *eh_proj = nullptr, *hc_gate = nullptr, *hc_up = nullptr, *hc_down = nullptr;
    const float *enorm = nullptr, *hnorm = nullptr, *fuse_gate = nullptr, *hc_hidden_norm = nullptr, *hc_embed_norm = nullptr,
                *final_norm = nullptr, *conf_w = nullptr, *conf_b = nullptr;
    float bias = 0.f;
    uint64_t wbytes = 0;

    std::vector<void*> salloc, spins;
    float *X = nullptr, *xn = nullptr, *o = nullptr, *o2 = nullptr, *mlp = nullptr, *z = nullptr;
    float *qa = nullptr, *qan = nullptr, *q = nullptr, *kvraw = nullptr, *qabs = nullptr, *qcat = nullptr, *olat = nullptr;
    float *vout = nullptr, *gate = nullptr, *g = nullptr, *u = nullptr, *h = nullptr;
    float *cat = nullptr, *fused = nullptr, *emb0 = nullptr, *pe = nullptr, *hg = nullptr, *hu = nullptr, *hh = nullptr,
          *hd = nullptr, *hcout = nullptr, *logits = nullptr, *craw = nullptr, *top = nullptr;
    void* ctxbuf = nullptr;
    int32_t *keys = nullptr, *nk = nullptr, *posd = nullptr;
    int8_t* actq = nullptr;
    float* actd = nullptr;
    void* attw = nullptr;
    float *hid = nullptr, *h_emb = nullptr, *h_pe = nullptr, *h_conf = nullptr;
    float* xmask = nullptr;        // the block's mask rows on the device (constant, written once)
    int32_t* h_pos = nullptr;      // padded host row of the block's rotary positions
    int32_t* h_tok = nullptr;
    uint64_t sbytes = 0;

    // the tap, by absolute position: a verifier window may replay positions, so a row is found by its position and a
    // replayed position overwrites its slot; the slot before a new window's start (draft_begin's source row) stays.
    int32_t* tsel = nullptr;
    int32_t tsel_n[kDflyTapRows][kDflyMaxLayers] = {};
    int64_t at[kDflyTapRows] = {};
    unsigned hid_ok = 0, sel_ok[kDflyTapRows] = {};
    bool mask_ready = false;
    std::vector<float> mask;

    ~Impl() {
        int prev = 0;
        const bool had = cudaGetDevice(&prev) == cudaSuccess;
        cudaSetDevice(device);
        for (void* p : salloc) cudaFree(p);
        for (void* p : spins) cudaFreeHost(p);
        for (void* p : walloc) cudaFree(p);
        if (had) cudaSetDevice(prev);
    }

    // several projections of the same input: one quantization of it, one mv call (the layer's q_a/kv_a/gate read xn, the
    // MLP's gate/up too, so the input is rounded once instead of once a projection)
    bool mm3(const void* const* W, const int* ns, float* const* ys, int nW, const float* x, int ldx, int k, int rows,
             cudaStream_t s, std::string& err) {
        k::q8_rows(x, (size_t) ldx, rows, k, actq, actd, s);
        iq::MvJob jobs[4];
        for (int i = 0; i < nW; ++i) {
            iq::MvJob& j = jobs[i];
            j.w = (const uint8_t*) W[i];
            j.ld = (size_t) (k / 32 * 34);
            j.type = kQ8_0;
            j.k = k;
            j.n = ns[i];
            j.ncol = rows;
            j.xq = actq;
            j.xd = actd;
            j.y = ys[i];
            j.ycs = (size_t) ns[i];
            j.alpha = 1.0f;
        }
        if (!iq::mv(jobs, nW, s)) return fail(err, "dfly: projections the kernels do not take");
        return true;
    }

    bool mm(const void* W, const float* x, int ldx, int k, int n, float* y, int ldy, int rows, cudaStream_t s,
            std::string& err) {
        k::q8_rows(x, (size_t) ldx, rows, k, actq, actd, s);
        iq::MvJob j;
        j.w = (const uint8_t*) W;
        j.ld = (size_t) (k / 32 * 34);
        j.type = kQ8_0;
        j.k = k;
        j.n = n;
        j.ncol = rows;
        j.xq = actq;
        j.xd = actd;
        j.y = y;
        j.ycs = (size_t) ldy;
        j.alpha = 1.0f;
        if (!iq::mv(&j, 1, s)) return fail(err, "dfly: a projection the kernels do not take");
        return true;
    }

    void tap_advance(int64_t pos0) {
        for (int i = 0; i < kDflyTapRows; ++i)
            if (at[i] >= pos0) {
                at[i] = -1;
                hid_ok &= ~(1u << i);
                sel_ok[i] = 0;
            }
    }
    void tap_clear() {
        std::memset(at, 0xff, sizeof at);   // (-1)
        hid_ok = 0;
        std::memset(sel_ok, 0, sizeof sel_ok);
    }
    void tap_hidden(int64_t pos, const float* fin, cudaStream_t s) {
        if (pos < 0) return;
        const int slot = (int) (pos & (kDflyTapRows - 1));
        if (at[slot] != pos) {
            at[slot] = pos;
            hid_ok &= ~(1u << slot);
            sel_ok[slot] = 0;
        }
        cudaMemcpyAsync(hid + (size_t) slot * d.E, fin, (size_t) d.E * 4, cudaMemcpyDeviceToDevice, s);
        hid_ok |= 1u << slot;
    }
    void tap_selection(int layer, int64_t pos, const int32_t* cells, int n, int cells_device, cudaStream_t s) {
        if (pos < 0) return;
        const int slot = (int) (pos & (kDflyTapRows - 1));
        for (int i = 0; i < d.layers; ++i) {
            if (d.idx_layers[(size_t) i] != layer) continue;
            if (at[slot] != pos) {
                at[slot] = pos;
                hid_ok &= ~(1u << slot);
                sel_ok[slot] = 0;
            }
            if (n > sel_cap) {   // the row cannot be held: leave the layer's bit clear (no draft from it)
                sel_ok[slot] &= ~(1u << i);
                continue;
            }
            tsel_n[slot][i] = n;
            if (n > 0) {
                const int32_t* src = cells;
                void* dst = tsel + ((size_t) i * kDflyTapRows + (size_t) slot) * sel_cap;
                if (cells_device >= 0 && cells_device != device)
                    cudaMemcpyPeerAsync(dst, device, src, cells_device, (size_t) n * 4, s);
                else
                    cudaMemcpyAsync(dst, src, (size_t) n * 4, cudaMemcpyDeviceToDevice, s);
            }
            sel_ok[slot] |= 1u << i;
        }
    }
    int tap_row(int64_t pos) const {
        const int64_t p = pos - 1;
        if (p < 0) return -1;
        const int slot = (int) (p & (kDflyTapRows - 1));
        if (at[slot] != p || !(hid_ok & (1u << slot)) || sel_ok[slot] != (1u << d.layers) - 1u) return -1;
        for (int i = 0; i < d.layers; ++i)
            if (tsel_n[slot][i] > sel_cap) return -1;
        return slot;
    }
    int tap_count(int slot, int i, int64_t pos) const {
        const int n = tsel_n[slot][i];
        return n < 0 ? (int) pos : n;   // n < 0: every position before the anchor (0 .. pos - 1)
    }

    bool gather(const DflyCache& c, const int32_t* cells, int n, void* dst, cudaStream_t s, std::string& err) {
        if (n <= 0) return true;
        const int src_dev = c.device < 0 ? device : c.device;
        if (c.gather) {
            if (!c.gather(cells, n, dst, s)) return fail(err, "dfly: gathering a draft layer's target rows");
            return true;
        }
        if (src_dev == device) {
            gather_rows_kernel<<<n, 128, 0, s>>>((const uint8_t*) c.rows, cells, n, (uint8_t*) dst, (int) c.row_bytes, c.ring);
            return true;
        }
        // another device: coalesce the (index-ordered) cells into consecutive runs, one peer copy a run. The cells are on the
        // head GPU (the tap's copy), so read them back on the same stream before coalescing.
        std::vector<int32_t> hc;
        if (cells) {
            hc.resize((size_t) n);
            if (cudaMemcpyAsync(hc.data(), cells, (size_t) n * 4, cudaMemcpyDeviceToHost, s) != cudaSuccess)
                return fail(err, "dfly: reading a selection");
            if (cudaStreamSynchronize(s) != cudaSuccess) return fail(err, "dfly: reading a selection");
        }
        int i = 0;
        while (i < n) {
            const int64_t p0 = cells ? hc[(size_t) i] : i;
            int j = i + 1;
            if (cells)
                while (j < n && hc[(size_t) j] == hc[(size_t) j - 1] + 1) ++j;
            else
                j = n;   // every position 0 .. n-1
            int run = j - i;
            if (p0 < 0) return fail(err, "dfly: a selected cell outside the cache");
            const int64_t slot = c.ring > 0 ? p0 % c.ring : p0;
            if (c.ring > 0) run = std::min(run, c.ring - (int) slot);
            const void* src = (const uint8_t*) c.rows + (size_t) slot * (size_t) c.row_bytes;
            void* out = (uint8_t*) dst + (size_t) i * (size_t) c.row_bytes;
            const size_t bytes = (size_t) run * (size_t) c.row_bytes;
            if (cudaMemcpyPeerAsync(out, device, src, src_dev, bytes, s) != cudaSuccess)
                return fail(err, "dfly: a peer copy of a draft layer's rows");
            i += run;
        }
        return true;
    }
};

Dfly::Dfly() = default;
Dfly::~Dfly() = default;

bool Dfly::on() const { return p_ != nullptr; }
const DflyDims& Dfly::dims() const {
    static const DflyDims none;
    return p_ ? p_->d : none;
}
uint64_t Dfly::weights_bytes() const { return p_ ? p_->wbytes : 0; }
uint64_t Dfly::scratch_bytes() const { return p_ ? p_->sbytes : 0; }

bool Dfly::probe(const std::string& gguf, const DflyTarget& target, int sel_cap, int64_t ctx, DflyInfo& info, std::string& err) {
    (void) ctx;
    if (sel_cap < 1) return fail(err, "dfly: a selection cap of " + std::to_string(sel_cap));
    Gguf g;
    DflyDims d;
    std::vector<Spec> sp;
    if (!parse(gguf, target, d, g, sp, err)) return false;
    info.dims = d;
    info.weights = 0;
    for (const Spec& s : sp) info.weights += s.bytes();
    info.scratch = scratch_bytes_for(d, target.kv_q8, sel_cap) + 4096;   // (the allocations' 16-byte rounding)
    return true;
}

bool Dfly::load(const std::string& gguf, const DflyTarget& target, int sel_cap, int64_t ctx, int device, Head head, Embed embed,
                std::string& err) {
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) return fail(err, "dfly: no CUDA device");
    if (cudaSetDevice(device) != cudaSuccess) return fail(err, "dfly: device " + std::to_string(device) + " is not usable");
    struct Restore {
        int prev;
        ~Restore() { cudaSetDevice(prev); }
    } restore{prev};
    if (sel_cap < 1) return fail(err, "dfly: a selection cap of " + std::to_string(sel_cap));
    p_.reset();
    auto p = std::make_unique<Impl>();
    p->device = device;
    p->target = target;
    p->sel_cap = sel_cap;
    p->ctx = ctx;
    p->head = std::move(head);
    p->embed = std::move(embed);
    Gguf g;
    std::vector<Spec> sp;
    if (!parse(gguf, target, p->d, g, sp, err)) return false;
    std::ifstream f(gguf, std::ios::binary);
    if (!f) return fail(err, gguf + ": cannot open");
    auto read_at = [&](const TensorRef& t, uint64_t bytes, std::vector<uint8_t>& out) -> bool {
        out.resize((size_t) bytes);
        f.seekg((std::streamoff) (g.data_start + t.off), std::ios::beg);
        f.read((char*) out.data(), (std::streamsize) out.size());
        return (bool) f;
    };
    for (const Spec& spc : sp) {
        void* dev = nullptr;
        if (cudaMalloc(&dev, std::max<uint64_t>(spc.bytes(), 16)) != cudaSuccess)
            return fail(err, "dfly: out of device memory for " + spc.name);
        p->walloc.push_back(dev);
        p->wslot[spc.name] = dev;
        p->wbytes += spc.bytes();
        std::vector<uint8_t> host;
        if (!spc.split_kv) {
            const TensorRef& t = g.tensors.at(spc.file);
            if (!read_at(t, spc.bytes(), host)) return fail(err, gguf + ": " + spc.file + " is truncated");
        } else {
            // build the kv_b_proj this engine reads: per head, 192 key rows then 256 value rows, each a Q8_0 row of the
            // 512 latents. attn_k_b is stored with its outputs contiguous (transpose it), attn_v_b with the latents
            // contiguous (its rows are the same bytes already).
            const TensorRef& ka = g.tensors.at(spc.file);
            const TensorRef& vb = g.tensors.at(spc.alt);
            const int H = (int) (spc.b / kDflyKvbHead);
            std::vector<uint8_t> kraw, vraw;
            if (!read_at(ka, (uint64_t) kDflyNope * kDflyRank * (uint64_t) H / 32 * 34, kraw) ||
                !read_at(vb, (uint64_t) kDflyRank * kDflyV * (uint64_t) H / 32 * 34, vraw))
                return fail(err, gguf + ": " + spc.file + " is truncated");
            std::vector<float> kd((size_t) kDflyNope * kDflyRank * (size_t) H);
            const size_t kb_row = (size_t) kDflyNope / 32 * 34;   // a run of 192 values: 6 Q8_0 blocks
            for (int h = 0; h < H; ++h)
                for (int i = 0; i < kDflyRank; ++i)
                    dequantize_row_q8_0(kraw.data() + ((size_t) h * kDflyRank + (size_t) i) * kb_row,
                                        kd.data() + ((size_t) h * kDflyRank + (size_t) i) * kDflyNope, kDflyNope);
            const size_t vrow = (size_t) kDflyRank / 32 * 34, row = (size_t) kDflyRank / 32 * 34;
            host.resize((size_t) spc.bytes());
            std::vector<float> x((size_t) kDflyRank);
            for (int h = 0; h < H; ++h) {
                for (int o = 0; o < kDflyNope; ++o) {
                    for (int i = 0; i < kDflyRank; ++i)
                        x[(size_t) i] = kd[((size_t) h * kDflyRank + (size_t) i) * kDflyNope + (size_t) o];
                    quantize_row_q8_0_ref(x.data(), host.data() + ((size_t) h * kDflyKvbHead + (size_t) o) * row, kDflyRank);
                }
                for (int o = 0; o < kDflyV; ++o)
                    std::memcpy(host.data() + ((size_t) h * kDflyKvbHead + kDflyNope + (size_t) o) * row,
                                vraw.data() + ((size_t) h * kDflyV + (size_t) o) * vrow, vrow);
            }
        }
        if (cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "dfly: copying " + spc.name);
    }
    auto q = [&](const std::string& n) { return p->wslot.at(n); };
    auto f32 = [&](const std::string& n) { return (const float*) p->wslot.at(n); };
    for (int i = 0; i < p->d.layers; ++i) {
        const std::string pr = "model.dfly_layers." + std::to_string(i) + ".";
        Impl::Layer l;
        l.q_a = q(pr + "self_attn.q_a_proj.weight");
        l.q_b = q(pr + "self_attn.q_b_proj.weight");
        l.kv_a = q(pr + "self_attn.kv_a_proj_with_mqa.weight");
        l.kv_b = q(pr + "self_attn.kv_b_proj.weight");
        l.gate = q(pr + "self_attn.linear_gate.weight");
        l.o = q(pr + "self_attn.o_proj.weight");
        l.mlp_gate = q(pr + "mlp.gate_proj.weight");
        l.mlp_up = q(pr + "mlp.up_proj.weight");
        l.mlp_down = q(pr + "mlp.down_proj.weight");
        l.in_ln = f32(pr + "input_layernorm.weight");
        l.q_a_ln = f32(pr + "self_attn.q_a_layernorm.weight");
        l.kv_a_ln = f32(pr + "self_attn.kv_a_layernorm.weight");
        l.post_proj_ln = f32(pr + "self_attn.post_proj_norm.weight");
        l.post_attn_ln = f32(pr + "post_attention_layernorm.weight");
        l.sink = f32(pr + "self_attn.learnable_sink_param");
        p->layer.push_back(l);
    }
    p->eh_proj = q("model.dfly_layers.mtp_fusion.eh_proj.weight");
    p->hc_gate = q("model.dfly_layers.hidden_correction.gate_proj.weight");
    p->hc_up = q("model.dfly_layers.hidden_correction.up_proj.weight");
    p->hc_down = q("model.dfly_layers.hidden_correction.down_proj.weight");
    p->enorm = f32("model.dfly_layers.mtp_fusion.enorm.weight");
    p->hnorm = f32("model.dfly_layers.mtp_fusion.hnorm.weight");
    p->fuse_gate = f32("model.dfly_layers.mtp_fusion.gate");
    p->hc_hidden_norm = f32("model.dfly_layers.hidden_correction.hidden_norm.weight");
    p->hc_embed_norm = f32("model.dfly_layers.hidden_correction.embed_norm.weight");
    p->final_norm = f32("model.dfly_layers.final_norm.weight");
    p->conf_w = f32("model.dfly_layers.confidence_head.proj.weight");
    p->conf_b = f32("model.dfly_layers.confidence_head.proj.bias");
    if (cudaMemcpy(&p->bias, p->conf_b, 4, cudaMemcpyDeviceToHost) != cudaSuccess)
        return fail(err, "dfly: the confidence bias");

    const DflyDims& d = p->d;
    const uint64_t R = (uint64_t) d.block, E = (uint64_t) d.E, H = (uint64_t) d.heads, F = (uint64_t) d.ffn,
                   C = (uint64_t) d.hc_ffn, Q = (uint64_t) d.q_lora;
    auto dev = [&](float*& ptr, size_t n) -> bool {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * 4, 16)) != cudaSuccess) return false;
        p->salloc.push_back(v);
        p->sbytes += std::max<size_t>(n * 4, 16);
        ptr = (float*) v;
        return true;
    };
    auto devi = [&](int32_t*& ptr, size_t n) -> bool {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * 4, 16)) != cudaSuccess) return false;
        p->salloc.push_back(v);
        p->sbytes += std::max<size_t>(n * 4, 16);
        ptr = (int32_t*) v;
        return true;
    };
    auto devb = [&](void*& ptr, size_t n) -> bool {
        if (cudaMalloc(&ptr, std::max<size_t>(n, 16)) != cudaSuccess) return false;
        p->salloc.push_back(ptr);
        p->sbytes += std::max<size_t>(n, 16);
        return true;
    };
    bool ok = true;
    ok = ok && dev(p->X, (size_t) (R * E)) && dev(p->xn, (size_t) (R * E)) && dev(p->o, (size_t) (R * E)) &&
         dev(p->o2, (size_t) (R * E)) && dev(p->mlp, (size_t) (R * E)) && dev(p->z, (size_t) (R * E));
    ok = ok && dev(p->qa, (size_t) (R * Q)) && dev(p->qan, (size_t) (R * Q)) && dev(p->q, (size_t) (R * H * 256)) &&
         dev(p->kvraw, (size_t) (R * kDflyRowWidth)) && dev(p->qabs, (size_t) (R * H * kDflyRank)) &&
         dev(p->qcat, (size_t) (R * H * kDflyRowWidth)) && dev(p->olat, (size_t) (R * H * kDflyRank)) &&
         dev(p->vout, (size_t) (R * H * kDflyV)) && dev(p->gate, (size_t) (R * H * kDflyV));
    ok = ok && dev(p->g, (size_t) (R * F)) && dev(p->u, (size_t) (R * F)) && dev(p->h, (size_t) (R * F));
    ok = ok && dev(p->cat, (size_t) (2 * E)) && dev(p->fused, (size_t) E) && dev(p->emb0, (size_t) E) &&
         dev(p->pe, (size_t) E) && dev(p->hg, (size_t) C) && dev(p->hu, (size_t) C) && dev(p->hh, (size_t) C) &&
         dev(p->hd, (size_t) E) && dev(p->hcout, (size_t) E) && dev(p->logits, (size_t) d.vocab) && dev(p->craw, 1) &&
         dev(p->top, 2);
    const uint64_t rb = drb(target.kv_q8);
    ok = ok && devb(p->ctxbuf, (size_t) ((uint64_t) sel_cap + R) * rb);
    ok = ok && devi(p->keys, (size_t) (R * ((uint64_t) sel_cap + R))) && devi(p->nk, (size_t) R) && devi(p->posd, (size_t) R);
    const uint64_t maxk = std::max({2 * E, Q, F, C, (uint64_t) H * kDflyV});
    if (ok) {
        void* v = nullptr;
        if (cudaMalloc(&v, (size_t) (R * maxk)) != cudaSuccess) ok = false;
        else {
            p->salloc.push_back(v);
            p->sbytes += (size_t) (R * maxk);
            p->actq = (int8_t*) v;
        }
        v = nullptr;
        if (ok && cudaMalloc(&v, (size_t) (R * maxk / 32 * 4)) != cudaSuccess) ok = false;
        else if (ok) {
            p->salloc.push_back(v);
            p->sbytes += (size_t) (R * maxk / 32 * 4);
            p->actd = (float*) v;
        }
    }
    if (ok) {
        const size_t aw = k::attn_ws_floats((int) R, (int) H, kDflyRank, (int) (sel_cap + R));
        ok = devb(p->attw, aw * 4);
    }
    // the tap
    ok = ok && dev(p->hid, (size_t) (kDflyTapRows * E));
    if (ok) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>((size_t) d.layers * kDflyTapRows * (uint64_t) sel_cap * 4, 16)) != cudaSuccess)
            ok = false;
        else {
            p->salloc.push_back(v);
            p->sbytes += (size_t) d.layers * kDflyTapRows * (uint64_t) sel_cap * 4;
            p->tsel = (int32_t*) v;
        }
    }
    auto pin = [&](size_t b) -> void* {
        void* v = nullptr;
        if (cudaMallocHost(&v, std::max<size_t>(b, 16)) != cudaSuccess) return nullptr;
        p->spins.push_back(v);
        return v;
    };
    if (void* v = pin((size_t) (R * E * 4))) p->h_emb = (float*) v;
    else ok = false;
    if (void* v = pin((size_t) E * 4)) p->h_pe = (float*) v;
    else ok = false;
    if (void* v = pin(4)) p->h_conf = (float*) v;
    else ok = false;
    if (void* v = pin(4)) p->h_tok = (int32_t*) v;
    else ok = false;
    if (!ok) return fail(err, "dfly: out of device memory for the drafter's scratch");
    // the mask rows are the same every round: one device copy of a block's worth, and a pinned position row
    ok = ok && dev(p->xmask, (size_t) ((kDflyMaxBlock - 1) * E));
    if (void* v = pin(sizeof(int32_t) * (size_t) kDflyMaxBlock)) p->h_pos = (int32_t*) v;
    else ok = false;
    if (!ok) return fail(err, "dfly: out of device memory for the drafter's scratch");
    p->tap_clear();
    p->mask.assign((size_t) d.E, 0.0f);
    p_ = std::move(p);
    row0_ = -1;
    pending_ = 0;
    return true;
}

// ---- the tap ------------------------------------------------------------------------------------------------------------
void Dfly::begin_window(int64_t pos0) {
    if (p_) p_->tap_advance(pos0);
}
void Dfly::invalidate() {
    if (p_) p_->tap_clear();
}
void Dfly::hidden(int first, int n, const float* fin, cudaStream_t s) {
    if (!p_ || row0_ < 0) return;
    for (int i = 0; i < n; ++i) p_->tap_hidden(row0_ + first + i, fin + (size_t) i * p_->d.E, s);
}
void Dfly::selection(int layer, int t, const int32_t* cells, int n, cudaStream_t s) {
    if (!p_ || row0_ < 0) return;
    p_->tap_selection(layer, row0_ + t, cells, n, p_->device, s);
}
void Dfly::hidden_at(int64_t pos, const float* fin, cudaStream_t s) {
    if (p_) p_->tap_hidden(pos, fin, s);
}
void Dfly::selection_at(int layer, int64_t pos, const int32_t* cells, int n, int cells_device, cudaStream_t s) {
    if (p_) p_->tap_selection(layer, pos, cells, n, cells_device, s);
}
int Dfly::source_row(int64_t pos) const {
    if (!p_) return -1;
    const int slot = p_->tap_row(pos);
    return slot < 0 ? -1 : (int) (pos - 1);
}

// ---- the block pass -----------------------------------------------------------------------------------------------------
bool Dfly::draft_begin(int32_t anchor, int64_t pos, int n, const DflyCache* caches, cudaStream_t s, int64_t rope_pos,
                       std::string& err) {
    if (!p_) return fail(err, "dfly: no drafter");
    Impl& p = *p_;
    if (cudaSetDevice(p.device) != cudaSuccess) return fail(err, "dfly: the drafter's device is not usable");
    if (pending_ > 0) {
        if (cudaStreamSynchronize(s) != cudaSuccess) return fail(err, "dfly: the previous block did not finish");
        pending_ = 0;
    }
    if (!p.head || !p.embed) return fail(err, "dfly: no head or embedding");
    const int slot = p.tap_row(pos);
    if (slot < 0 || pos + p.d.block > p.ctx) return true;   // no hidden state to draft from: the window goes plain
    if (n < 1) return fail(err, "dfly: a block of " + std::to_string(n) + " drafts");
    n = std::min(n, p.d.block - 1);
    if (!caches) return fail(err, "dfly: no target caches");
    const DflyDims& d = p.d;
    const int R = n + 1, E = d.E;
    if (!p.mask_ready) {
        if (!p.embed(d.mask_id, p.mask.data())) return fail(err, "dfly: no embedding row for the mask token");
        std::vector<float> mr((size_t) (kDflyMaxBlock - 1) * E);
        for (int r = 0; r < kDflyMaxBlock - 1; ++r)
            std::memcpy(mr.data() + (size_t) r * E, p.mask.data(), (size_t) E * 4);
        if (cudaMemcpyAsync(p.xmask, mr.data(), mr.size() * 4, cudaMemcpyHostToDevice, s) != cudaSuccess)
            return fail(err, "dfly: the mask rows");
        p.mask_ready = true;
    }
    if (!p.embed(anchor, p.h_emb)) return fail(err, "dfly: the anchor token has no embedding row");
    cudaMemcpyAsync(p.emb0, p.h_emb, (size_t) E * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(p.X + E, p.xmask, (size_t) (R - 1) * E * 4, cudaMemcpyDeviceToDevice, s);
    k::rmsnorm(p.emb0, E, 1, E, p.enorm, d.rms_eps, p.cat, 2 * E, s);
    k::rmsnorm(p.hid + (size_t) slot * E, E, 1, E, p.hnorm, d.rms_eps, p.cat + E, 2 * E, s);
    if (!p.mm(p.eh_proj, p.cat, 2 * E, 2 * E, E, p.fused, E, 1, s, err)) return false;
    gate_add_kernel<<<(E + 255) / 256, 256, 0, s>>>(p.emb0, p.fused, p.fuse_gate, E, p.X);
    const int64_t rp = rope_pos < 0 ? pos : rope_pos;
    for (int r = 0; r < R; ++r) p.h_pos[r] = (int32_t) (rp + r);
    cudaMemcpyAsync(p.posd, p.h_pos, sizeof(int32_t) * (size_t) R, cudaMemcpyHostToDevice, s);
    const uint64_t rb = drb(p.target.kv_q8);
    for (int i = 0; i < d.layers; ++i) {
        const int nctx = p.tap_count(slot, i, pos);
        if (nctx < 0 || nctx > p.sel_cap || nctx + R > 4096)
            return fail(err, "dfly: " + std::to_string(nctx) + " selected rows");
        const int kvl = d.kv_layers[(size_t) i];
        const DflyCache& c = caches[kvl];
        if (!c.rows || c.row_bytes != (int64_t) rb || c.q8 != p.target.kv_q8)
            return fail(err, "dfly: draft layer " + std::to_string(i) + " has no usable cache rows");
        const int32_t* cells = p.tsel_n[slot][i] < 0
                                   ? nullptr
                                   : p.tsel + ((size_t) i * kDflyTapRows + (size_t) slot) * p.sel_cap;
        if (!p.gather(c, cells, nctx, p.ctxbuf, s, err)) return false;
        const Impl::Layer& L = p.layer[(size_t) i];
        k::rmsnorm(p.X, E, R, E, L.in_ln, d.rms_eps, p.xn, E, s);
        {   // q_a, kv_a and the attention's gate all read xn: one rounding of it
            const void* W[3] = {L.q_a, L.kv_a, L.gate};
            const int ns[3] = {d.q_lora, kDflyRowWidth, d.heads * kDflyV};
            float* ys[3] = {p.qa, p.kvraw, p.gate};
            if (!p.mm3(W, ns, ys, 3, p.xn, E, E, R, s, err)) return false;
        }
        k::rmsnorm(p.qa, d.q_lora, R, d.q_lora, L.q_a_ln, d.rms_eps, p.qan, d.q_lora, s);
        if (!p.mm(L.q_b, p.qan, d.q_lora, d.q_lora, d.heads * 256, p.q, d.heads * 256, R, s, err)) return false;
        absorb_kernel<<<dim3(kDflyRank, (unsigned) (R * d.heads)), 256, 0, s>>>((const uint8_t*) L.kv_b, p.q, d.heads,
                                                                               p.qabs);
        // the block's own rows, in the target's cache form, after the selected ones (position rp + r -> row nctx + r)
        const uintptr_t base = (uintptr_t) p.ctxbuf + (uintptr_t) ((int64_t) ((int64_t) nctx - rp) * (int64_t) rb);
        k::kv_store(p.kvraw, R, kDflyRank, kDflyRope, L.kv_a_ln, d.rms_eps, p.posd, d.rope_base, (uint16_t*) base, 0, s,
                    p.target.kv_q8);
        k::rope(p.q, R, d.heads, 256, (size_t) d.heads * 256, kDflyNope, kDflyRope, p.posd, d.rope_base, s);
        k::copy_cols(p.qabs, kDflyRank, p.qcat, kDflyRowWidth, R * d.heads, kDflyRank, s);
        k::copy_cols(p.q + kDflyNope, 256, p.qcat + kDflyRank, kDflyRowWidth, R * d.heads, kDflyRope, s);
        const int maxk = nctx + R;
        keys_fill_kernel<<<R, 256, 0, s>>>(p.keys, maxk, p.nk, nctx, R);
        const float scale = 1.0f / std::sqrt((float) (kDflyNope + kDflyRope));
        k::attention(p.qcat, R, d.heads, kDflyRank, kDflyRope, (const uint16_t*) p.ctxbuf, 0, p.keys, maxk, nullptr, p.nk,
                     maxk, scale, L.sink, (float*) p.attw, p.olat, s, p.target.kv_q8);
        expand_kernel<<<dim3(kDflyV, (unsigned) (R * d.heads)), 256, 0, s>>>((const uint8_t*) L.kv_b, p.olat, d.heads,
                                                                             p.vout);
        k::sigmoid_mul(p.vout, p.gate, (size_t) R * d.heads * kDflyV, s);
        if (!p.mm(L.o, p.vout, d.heads * kDflyV, d.heads * kDflyV, E, p.o, E, R, s, err)) return false;
        k::rmsnorm(p.o, E, R, E, L.post_proj_ln, d.rms_eps, p.o2, E, s);
        k::add(p.X, p.o2, (size_t) R * E, s);
        k::rmsnorm(p.X, E, R, E, L.post_attn_ln, d.rms_eps, p.xn, E, s);
        {
            const void* W[2] = {L.mlp_gate, L.mlp_up};
            const int ns[2] = {d.ffn, d.ffn};
            float* ys[2] = {p.g, p.u};
            if (!p.mm3(W, ns, ys, 2, p.xn, E, E, R, s, err)) return false;
        }
        k::swiglu(p.g, p.u, (size_t) d.ffn, R, d.ffn, p.h, (size_t) d.ffn, s);
        if (!p.mm(L.mlp_down, p.h, d.ffn, d.ffn, E, p.mlp, E, R, s, err)) return false;
        k::add(p.X, p.mlp, (size_t) R * E, s);
    }
    k::rmsnorm(p.X, E, R, E, p.final_norm, d.rms_eps, p.z, E, s);
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return fail(err, std::string("dfly: CUDA: ") + cudaGetErrorString(ce));
    pending_ = n;
    made_ = n;
    anchor_ = anchor;
    return true;
}

bool Dfly::draft_end(int32_t* ids, float* conf, int max_n, int& n, float floor, cudaStream_t s, std::string& err) {
    n = 0;
    if (!p_) return fail(err, "dfly: no drafter");
    if (pending_ == 0) return true;
    Impl& p = *p_;
    if (cudaSetDevice(p.device) != cudaSuccess) return fail(err, "dfly: the drafter's device is not usable");
    const DflyDims& d = p.d;
    const int E = d.E, C = d.hc_ffn;
    pending_ = 0;
    int32_t prev = anchor_;
    for (int j = 0; j < made_ && j < max_n; ++j) {
        if (!p.embed(prev, p.h_pe)) return fail(err, "dfly: a draft token has no embedding row");
        cudaMemcpyAsync(p.pe, p.h_pe, (size_t) E * 4, cudaMemcpyHostToDevice, s);
        const float* zrow = p.z + (size_t) (j + 1) * E;
        k::rmsnorm(zrow, E, 1, E, p.hc_hidden_norm, d.rms_eps, p.cat, 2 * E, s);
        k::rmsnorm(p.pe, E, 1, E, p.hc_embed_norm, d.rms_eps, p.cat + E, 2 * E, s);
        {
            const void* W[2] = {p.hc_gate, p.hc_up};
            const int ns[2] = {C, C};
            float* ys[2] = {p.hg, p.hu};
            if (!p.mm3(W, ns, ys, 2, p.cat, 2 * E, 2 * E, 1, s, err)) return false;
        }
        k::swiglu(p.hg, p.hu, (size_t) C, 1, C, p.hh, (size_t) C, s);
        if (!p.mm(p.hc_down, p.hh, C, C, E, p.hd, E, 1, s, err)) return false;
        cudaMemcpyAsync(p.hcout, zrow, (size_t) E * 4, cudaMemcpyDeviceToDevice, s);
        k::add(p.hcout, p.hd, (size_t) E, s);
        if (!p.head(p.hcout, p.logits, s)) return fail(err, "dfly: the target's head failed");
        k::top1(p.logits, d.vocab, p.top, s);
        k::gemv_f32w(p.conf_w, E, E, 1, p.hcout, E, 1, p.craw, 1, 1.0f, s);
        cudaMemcpyAsync(p.h_tok, p.top, 4, cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(p.h_conf, p.craw, 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess)
            return fail(err, std::string("dfly: CUDA: ") + cudaGetErrorString(cudaGetLastError()));
        ids[j] = *p.h_tok;
        conf[j] = 1.0f / (1.0f + std::exp(-(*p.h_conf + p.bias)));
        prev = ids[j];
        n = j + 1;
        if (floor > 0.0f && conf[j] < floor) break;   // (the caller drops this slot and every later one)
    }
    return true;
}

}  // namespace hy4
