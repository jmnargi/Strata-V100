// src/kernels/cpu/nvfp4_avx2.cpp - see include/strata/kernels/cpu/nvfp4_avx2.hpp.
//
// Per 64-value block, once: its E2M1 values in element order as two 32-byte registers (ggml's reordering: sub-block
// s keeps value j in the low nibble of qs[8 s + j] and value j + 8 in the high one), their magnitudes, and the four
// sub-block scales. Per token, as ggml_vec_dot_nvfp4_q8_0's AVX2 path: vpsignb moves the sign onto the activation,
// vpmaddubsw + vpmaddwd give 8 int32 lanes (four per 16-value sub-block), one FMA per 32 values applies
// ue4m3(d[s]) * d_q8, and hsum_float_8's order sums the lanes.
//
// Both registers come from ONE 32-byte load (two 256-bit shuffles, unpacklo/hi and permute2x128 re-pair the four
// sub-blocks), and a row's four scale bytes are read once. At one token - the commonest group in decode - a gate row
// and its up row are multiplied together: they share the activation loads, while each keeps its own accumulator and
// its own operation order, so every row is still ggml's bit for bit.
//
// Needs AVX2, FMA and F16C - what cpu_avx2_ok() checks before anything here is called.
#include "strata/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// A compile-time table, not a constructor: this file is compiled for a wide ISA, and a namespace-scope object with a
// constructor runs at program start on EVERY CPU - the compiler vectorises the loop with AVX/FMA, and a CPU without them
// (Westmere, Sandy Bridge, ...) died with an illegal instruction before main (#2324).  Powers of two: exact.
constexpr float pow2i(int e) {
    float r = 1.0f;
    for (; e > 0; --e) r *= 2.0f;
    for (; e < 0; ++e) r *= 0.5f;
    return r;
}
struct Ue4m3Table {   // ggml_ue4m3_to_fp32 (ggml-impl.h): halved for the doubled kvalues_fp4, 0x7F read as 0
    float v[256];
    constexpr Ue4m3Table() : v() {
        for (int x = 0; x < 256; ++x) {
            if (x == 0 || x == 0x7F) { v[x] = 0.0f; continue; }
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? (float) m * pow2i(-9) : (1.0f + (float) m / 8.0f) * pow2i(e - 7);
            v[x] = raw * 0.5f;
        }
    }
};
constexpr Ue4m3Table kUe4m3;

// software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows
// (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above)
inline int prefetch_distance() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_NVFP4_PREFETCH");
        return e ? std::atoi(e) : 2048;
    }();
    return v;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

inline __m256i table() { return _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kvalues_fp4)); }

// The block's four sub-blocks in element order: sub-blocks 0, 1 in v01 and 2, 3 in v23. Sub-block s keeps its eight
// codes in qs[8 s .. 8 s + 7], byte j holding value j low and value j + 8 high, so the 32 bytes split into low and
// high nibbles; unpacklo/hi pair a sub-block's two halves and the two permutes gather the pairs (as codes64 does in
// 512-bit lanes, without the insert: one 32-byte load here, not two 16-byte ones).
inline void values32x2(const uint8_t* qs, const __m256i kv, __m256i& v01, __m256i& v23) {
    const __m256i q = _mm256_loadu_si256((const __m256i*) qs);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_shuffle_epi8(kv, _mm256_and_si256(q, m4));
    const __m256i hi = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
    const __m256i ul = _mm256_unpacklo_epi64(lo, hi);    // [sub-block 0 | sub-block 2]
    const __m256i uh = _mm256_unpackhi_epi64(lo, hi);    // [sub-block 1 | sub-block 3]
    v01 = _mm256_permute2x128_si256(ul, uh, 0x20);       // sub-blocks 0, 1
    v23 = _mm256_permute2x128_si256(ul, uh, 0x31);       // sub-blocks 2, 3
}

// lanes 0-3 ue4m3(d[0]), 4-7 d[1] (and d[2], d[3]): times a broadcast d_q8, each lane is ggml's scalar
// ue4m3(d[s]) * d_q8, rounded the same. All four bytes are one load: the block stores them at offset 0, and this
// file is AVX2-only, so the host is little-endian (d[0] is the low byte).
inline void sub_scales(const uint8_t* d, __m256& u01, __m256& u23) {
    uint32_t d4;
    std::memcpy(&d4, d, sizeof d4);
    u01 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[(d4 >> 8) & 0xFF]), _mm_set1_ps(kUe4m3.v[d4 & 0xFF]));
    u23 = _mm256_set_m128(_mm_set1_ps(kUe4m3.v[d4 >> 24]), _mm_set1_ps(kUe4m3.v[(d4 >> 16) & 0xFF]));
}

inline float hsum8(const __m256 x) {   // ggml's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// one block of one token against one row, in ggml's exact operation order; acc comes back in the same lane order
inline __m256 dot_block_token(const __m256i v01, const __m256i v23, const __m256i a01, const __m256i a23,
                              const __m256 u01, const __m256 u23, const __m256i q01, const __m256i q23,
                              const float* dy, const __m256 acc, const __m256i ones) {
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q01, v01)), ones);
    const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q23, v23)), ones);
    const __m256 s01 = _mm256_mul_ps(u01, _mm256_broadcast_ss(dy));
    const __m256 s23 = _mm256_mul_ps(u23, _mm256_broadcast_ss(dy + 1));
    __m256 r = _mm256_fmadd_ps(s01, _mm256_cvtepi32_ps(p1), acc);
    r = _mm256_fmadd_ps(s23, _mm256_cvtepi32_ps(p2), r);
    return r;
}

constexpr int kMaxBlocks = 64;   // rows of up to 4096 values (n_embd 2560)
// tokens per pass: the pool's MAXT, so a window is one pass. Spilled accumulators cost less than another pass over
// the weights: one expert's gate+up, 8 tokens, 0.415 ms in passes of 4 + 4, 0.352 in one of 8 (5 6 7 alike)
constexpr int kMaxW = 8;

struct Acts {   // per token and block, the two Q8_0 blocks' scales; built once per call, reused by every row
    int nt = 0, nb = 0;
    const block_q8_0* y[kMaxW] = {};
    float dy[kMaxW * kMaxBlocks * 2];
    Acts(const void* const* act, int nt_, int nb_) : nt(nt_), nb(nb_) {
        for (int t = 0; t < nt; ++t) {
            y[t] = (const block_q8_0*) act[t];
            for (int ib = 0; ib < 2 * nb; ++ib) dy[(size_t) t * 2 * nb + ib] = h2f(y[t][ib].d);
        }
    }
};

template <int NT>
inline void row_dot(const uint8_t* row, const Acts& a, const __m256i kv, const __m256i ones, float* res) {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    const block_nvfp4* x = (const block_nvfp4*) row;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) _mm_prefetch((const char*) (x + ib) + prefetch_ahead, _MM_HINT_T0);
        __m256i v01, v23;
        values32x2(x[ib].qs, kv, v01, v23);
        const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
        __m256 u01, u23;
        sub_scales(x[ib].d, u01, u23);
        for (int t = 0; t < NT; ++t) {
            const block_q8_0* yb = a.y[t] + 2 * ib;
            const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
            const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
            const float* dy = a.dy + (size_t) t * 2 * a.nb + 2 * ib;
            acc[t] = dot_block_token(v01, v23, a01, a23, u01, u23, q01, q23, dy, acc[t], ones);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]);
}

// Two rows against the same one-token activations. What is shared is only the activation load: each row has its own
// decoded block, scales and accumulator, and dot_block_token runs its operations in the same order, so each row is
// bit-equal to what row_dot gives for it. One token only: at two, both rows' state plus the four accumulators no
// longer fits the sixteen YMM registers, and row_dot<2> already reaches its ~1.6x.
inline void row_dot_pair(const uint8_t* row0, const uint8_t* row1, const Acts& a, const __m256i kv,
                         const __m256i ones, float* res0, float* res1) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    const block_nvfp4* x0 = (const block_nvfp4*) row0;
    const block_nvfp4* x1 = (const block_nvfp4*) row1;
    const int prefetch_ahead = prefetch_distance();
    for (int ib = 0; ib < a.nb; ++ib) {
        if (prefetch_ahead > 0) {
            _mm_prefetch((const char*) (x0 + ib) + prefetch_ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) (x1 + ib) + prefetch_ahead, _MM_HINT_T0);
        }
        const block_q8_0* yb = a.y[0] + 2 * ib;
        const __m256i q01 = _mm256_loadu_si256((const __m256i*) yb[0].qs);
        const __m256i q23 = _mm256_loadu_si256((const __m256i*) yb[1].qs);
        const float* dy = a.dy + 2 * ib;
        {   // one row's decoded state at a time: both rows plus two accumulators would not fit the YMM file
            __m256i v01, v23;
            values32x2(x0[ib].qs, kv, v01, v23);
            const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
            __m256 u01, u23;
            sub_scales(x0[ib].d, u01, u23);
            acc0 = dot_block_token(v01, v23, a01, a23, u01, u23, q01, q23, dy, acc0, ones);
        }
        {
            __m256i v01, v23;
            values32x2(x1[ib].qs, kv, v01, v23);
            const __m256i a01 = _mm256_sign_epi8(v01, v01), a23 = _mm256_sign_epi8(v23, v23);
            __m256 u01, u23;
            sub_scales(x1[ib].d, u01, u23);
            acc1 = dot_block_token(v01, v23, a01, a23, u01, u23, q01, q23, dy, acc1, ones);
        }
    }
    res0[0] = hsum8(acc0);
    res1[0] = hsum8(acc1);
}

template <int NT>
inline void swiglu(float* const* ff, const float* g, const float* u, int r, float sg, float su) {
    for (int t = 0; t < NT; ++t) {   // native_gu_rows' ggml path, operation for operation
        const float gs = g[t] * sg, us = u[t] * su;
        ff[t][r] = (gs / (1.f + std::exp(-gs))) * us;
    }
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, const Acts& a, float* const* ff, int r0, int r1,
             float sg, float su) {
    const __m256i kv = table();
    const __m256i ones = _mm256_set1_epi16(1);
    float g[NT], u[NT];
    if constexpr (NT == 1) {
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gr = blob + (size_t) r * gu_row;
            const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
            row_dot_pair(gr, ur, a, kv, ones, g, u);
            swiglu<1>(ff, g, u, r, sg, su);
        }
    } else {
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gr = blob + (size_t) r * gu_row;
            const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
            row_dot<NT>(gr, a, kv, ones, g);
            row_dot<NT>(ur, a, kv, ones, u);
            swiglu<NT>(ff, g, u, r, sg, su);
        }
    }
}

template <int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, const Acts& a, float* const* out, int r0, int r1, float scale) {
    const __m256i kv = table();
    const __m256i ones = _mm256_set1_epi16(1);
    if constexpr (NT == 1) {
        int r = r0;
        for (; r + 1 < r1; r += 2) {   // two adjacent rows share one token's activation load
            float lo[1], hi[1];
            row_dot_pair(w + (size_t) r * row_bytes, w + (size_t) (r + 1) * row_bytes, a, kv, ones, lo, hi);
            out[0][r] = lo[0] * scale;
            out[0][r + 1] = hi[0] * scale;
        }
        if (r < r1) {
            float res[1];
            row_dot<1>(w + (size_t) r * row_bytes, a, kv, ones, res);
            out[0][r] = res[0] * scale;
        }
    } else {
        float res[NT];
        for (int r = r0; r < r1; ++r) {
            row_dot<NT>(w + (size_t) r * row_bytes, a, kv, ones, res);
            for (int t = 0; t < NT; ++t) out[t][r] = res[t] * scale;
        }
    }
}

// windows wider than kMaxW (not from the pool) run in balanced slices of at most kMaxW
template <typename F>
inline void slices(int nt, F&& run) {
    const int k = (nt + kMaxW - 1) / kMaxW;
    for (int i = 0, t0 = 0; i < k; ++i) {
        const int w = nt / k + (i < nt % k ? 1 : 0);
        run(t0, w);
        t0 += w;
    }
}

}  // namespace

bool nvfp4_256_fits(int n) { return n % QK_NVFP4 == 0 && n / QK_NVFP4 <= kMaxBlocks; }

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                       float* const* ff, int r0, int r1, float s_gate, float s_up) {
    slices(nt, [&](int t0, int w) {
        const Acts a(act + t0, w, n / QK_NVFP4);
        float* const* f = ff + t0;
        switch (w) {
            case 1: gu_rows<1>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
            default: gu_rows<8>(blob, gu_row, up_off, a, f, r0, r1, s_gate, s_up); break;
        }
    });
}

void nvfp4_256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                    int r0, int r1, float scale) {
    slices(nt, [&](int t0, int k) {
        const Acts a(act + t0, k, n / QK_NVFP4);
        float* const* o = out + t0;
        switch (k) {
            case 1: dot_rows<1>(w, row_bytes, a, o, r0, r1, scale); break;
            case 2: dot_rows<2>(w, row_bytes, a, o, r0, r1, scale); break;
            case 3: dot_rows<3>(w, row_bytes, a, o, r0, r1, scale); break;
            case 4: dot_rows<4>(w, row_bytes, a, o, r0, r1, scale); break;
            case 5: dot_rows<5>(w, row_bytes, a, o, r0, r1, scale); break;
            case 6: dot_rows<6>(w, row_bytes, a, o, r0, r1, scale); break;
            case 7: dot_rows<7>(w, row_bytes, a, o, r0, r1, scale); break;
            default: dot_rows<8>(w, row_bytes, a, o, r0, r1, scale); break;
        }
    });
}

}  // namespace strata::kernels::cpu
