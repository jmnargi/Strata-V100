// src/kernels/cuda/qpn2_nvfp4.cu - NVFP4 routed experts in decode as W4A16 on Volta's mma.sync.m8n8k4.
//
// The dataflow is v100-skinny's QPN2 (github.com/dnv2003/v100-skinny, kernels/skinny_kernels.cu, Apache-2.0): one
// mma.m8n8k4 call is four independent 8x8x4 products ("quadpairs") that share one 8x4 A tile, and the B fragments are
// the e2m1 codes decoded in registers.  What differs is the weight layout: v100-skinny reads weights prepacked in
// fragment order; this engine keeps the checkpoint's own rows (ggml's block_nvfp4: four UE4M3 scale bytes then 32
// code bytes, 36 bytes per 64 values).  The fragment a lane needs is the 16 codes of one 16-value sub-block, and a
// sub-block IS 8 contiguous bytes there (`qs[8j..8j+7]`, value 16j+i in the low nibble of byte i and 16j+i+8 in its
// high one), so the fragment is a nibble pairing of two 4-byte loads instead of a repack at load time.
//
// M (the rows one expert serves in a verify window) is 1..8: a group is one expert and an entry one (row, expert)
// pair, and the engine's window holds at most kVerifyMaxT = 8 rows, so a group has at most 8 entries.  The switch is
// an environment variable, so the default path stays byte-identical.
// Activations are the caller's q8_1 image, dequantized to half in registers: decode's activation quantization stays
// the engine's (no new buffer, no change in verify.cpp), and the weight side is the value the checkpoint stores.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/qpn2_nvfp4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QPN2_SPLITK = 8;   // warps a CTA, splitting K on one 32-column tile (v100-skinny: 8 for gate/up)

// e2m1 code -> fp16, exactly: the magnitudes' fp16 high bytes {0,.5,1,1.5} and {2,3,4,6} as two 32-bit words, so one
// PRMT picks either code's magnitude; the sign bits are OR'd in after.  v100-skinny's LUT_LO/LUT_HI (Apache-2.0).
constexpr unsigned LUT_LO = 0x3E3C3800u, LUT_HI = 0x46444240u;

// code0 -> the low half's byte 1, code1 -> the high half's byte 3 (the halves' byte 0 is zero in LUT_LO/LUT_HI)
__device__ __forceinline__ half2 e2m1_pair(unsigned code0, unsigned code1) {
    const unsigned sel = ((code0 & 0x7u) << 4) | ((code1 & 0x7u) << 12);
    unsigned h = __byte_perm(LUT_LO, LUT_HI, sel);
    h |= ((code0 & 0x8u) << 12) | ((code1 & 0x8u) << 28);   // the low half's sign is bit 15, the high half's bit 31
    return *reinterpret_cast<half2*>(&h);
}

// UE4M3 block scale -> half.  UNSIGNED: four exponent bits (bias 7) and three mantissa bits, and bit 7 is not a sign
// (ggml's ggml_ue4m3_to_fp32 masks 0x7F; the packers write it as 0).  Every finite value is exact in fp16; the one NaN
// pattern (0x7F) reads as 0, which is how ggml reads it too.
__device__ __forceinline__ half ue4m3_to_half(unsigned b) {
    b &= 0x7Fu;
    const unsigned e = (b >> 3) & 0xFu, m = b & 0x7u;
    if (e == 0) return __float2half_rn((float) m * 0.001953125f);            // m * 2^-9
    if (e == 15 && m == 7) return __ushort_as_half(0);
    return __ushort_as_half((unsigned short) (((e + 8u) << 10) | (m << 7)));
}

// b[0..7] = the B registers (adjacent k pairs (2i, 2i+1)) of one 16-value group, from its eight native code bytes:
// w0 holds bytes 0..3, w1 bytes 4..7, value 16j+i in the low nibble of byte i and 16j+i+8 in its high one.
__device__ __forceinline__ void qpn2_b(unsigned w0, unsigned w1, half2 sc2, half2 b[8]) {
    const unsigned n0 = w0 & 0xFFu, n1 = (w0 >> 8) & 0xFFu, n2 = (w0 >> 16) & 0xFFu, n3 = w0 >> 24;
    const unsigned n4 = w1 & 0xFFu, n5 = (w1 >> 8) & 0xFFu, n6 = (w1 >> 16) & 0xFFu, n7 = w1 >> 24;
    b[0] = __hmul2(e2m1_pair(n0 & 0xFu, n1 & 0xFu), sc2);
    b[1] = __hmul2(e2m1_pair(n2 & 0xFu, n3 & 0xFu), sc2);
    b[2] = __hmul2(e2m1_pair(n4 & 0xFu, n5 & 0xFu), sc2);
    b[3] = __hmul2(e2m1_pair(n6 & 0xFu, n7 & 0xFu), sc2);
    b[4] = __hmul2(e2m1_pair(n0 >> 4, n1 >> 4), sc2);
    b[5] = __hmul2(e2m1_pair(n2 >> 4, n3 >> 4), sc2);
    b[6] = __hmul2(e2m1_pair(n4 >> 4, n5 >> 4), sc2);
    b[7] = __hmul2(e2m1_pair(n6 >> 4, n7 >> 4), sc2);
}

__device__ __forceinline__ void mma8n8k4(float c[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// q8_1's 16 int8 values at `qs` (4-byte aligned; block_q8_1 is 36 bytes) -> 8 half2 in natural k order, times its d
__device__ __forceinline__ void q8_1_16_to_half(const int8_t* qs, half d, half2 out[8]) {
    const half2 d2 = __half2half2(d);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = *reinterpret_cast<const int*>(qs + 4 * i);
        out[2 * i + 0] = __hmul2(__floats2half2_rn((float) (signed char) (v & 0xFF),
                                                   (float) (signed char) ((v >> 8) & 0xFF)), d2);
        out[2 * i + 1] = __hmul2(__floats2half2_rn((float) (signed char) ((v >> 16) & 0xFF),
                                                   (float) (signed char) (v >> 24)), d2);
    }
}

// GU:   Y[e * n_ff + n]            = s_proj * X[ent_tok[e]] . W[n], n < 2 * n_ff (gate rows, then up rows), tail[0/1]
// DOWN: Y[ent_dst[e] * n_embd + n] = s_down * Hq[e] . W[n],         n < n_embd,                        tail[2]
template <bool GU>
__global__ void __launch_bounds__(32 * QPN2_SPLITK) qpn2_nvfp4_kernel(
        const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
        const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent,
        const block_q8_1* __restrict__ xq, NativeExpertLayout L, float* __restrict__ out0, float* __restrict__ out1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3;
    const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);            // the group entry (A row) this lane holds
    const int bcol = qp * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);   // the weight row this lane holds

    const int tile = blockIdx.x;                                    // 32 weight rows
    const bool is_up = GU && tile * 32 >= (int) L.n_ff;
    const int row0 = GU ? (is_up ? tile * 32 - (int) L.n_ff : tile * 32) : tile * 32;
    const int K = (int) (GU ? L.n_embd : L.n_ff);
    const int row_bytes = (int) (GU ? L.gu_row : L.d_row);
    const int xb = (int) (GU ? L.n_embd / 32 : L.n_ff / 32);        // q8_1 blocks a token row
    const int SB = K >> 6;                                          // 64-value blocks (K/64)

    __shared__ float cs[QPN2_SPLITK][256];

    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        const int nent = e1 - e0;
        if (nent <= 0) continue;
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        float scale = 1.0f;
        if (L.tail_off) {
            const float* tail = (const float*) (blob + L.tail_off);
            scale = GU ? (is_up ? tail[1] : tail[0]) : tail[2];
        }
        float c[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = 0.f;

        const bool has_a = arow < nent;
        const int trow = has_a ? (GU ? ent[e0 + arow] : (e0 + arow)) : 0;
        const uint8_t* wr = blob + (GU ? (is_up ? L.up_off : 0) : L.down_off) + (size_t) (row0 + bcol) * row_bytes;
        for (int s = warp; s < SB; s += QPN2_SPLITK) {
            const uint8_t* bp = wr + (size_t) s * 36;
            const unsigned d4 = __ldcs(reinterpret_cast<const unsigned*>(bp));   // the block's four UE4M3 scale bytes
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned w0 = __ldcs(reinterpret_cast<const unsigned*>(bp + 4 + 8 * j));   // no reuse: streaming
                const unsigned w1 = __ldcs(reinterpret_cast<const unsigned*>(bp + 8 + 8 * j));
                const half sc = ue4m3_to_half((d4 >> (8 * j)) & 0xFFu);
                half2 b[8];
                qpn2_b(w0, w1, __halves2half2(sc, sc), b);
                if (!has_a) continue;
                // this group's 16 activation values: half of a q8_1 block
                const int k0 = s * 64 + j * 16;
                const block_q8_1* bq = xq + (size_t) trow * xb + (k0 >> 5);
                half2 av[8];
                q8_1_16_to_half(bq->qs + (k0 & 31), __low2half(bq->ds), av);
                const unsigned* A = reinterpret_cast<const unsigned*>(av);
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                mma8n8k4(c, A[0], A[1], B[0], B[1]);
                mma8n8k4(c, A[2], A[3], B[2], B[3]);
                mma8n8k4(c, A[4], A[5], B[4], B[5]);
                mma8n8k4(c, A[6], A[7], B[6], B[7]);
            }
        }

        // the C map (v100-skinny's, roles as in its mma8_probe.cu): reg i of lane L -> row / col inside the 8x8 quadpair
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
            const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][row * 32 + qp * 8 + col] = c[i];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < QPN2_SPLITK; ++w) v += cs[w][i];
            const int li = i >> 5, col = i & 31;                // entry inside the group, column of the tile
            if (li >= nent) continue;
            const int row = row0 + col;
            const int dst = GU ? (e0 + li) * (int) L.n_ff + row : ent[e0 + li] * (int) L.n_embd + row;
            (GU ? (is_up ? out1 : out0) : out0)[dst] = v * scale;
        }
        __syncthreads();
    }
#endif
}

}  // namespace

bool qpn2_nvfp4_supported(const NativeExpertLayout& L, int64_t cap_entries) noexcept {
    if (L.gu_type != 40 || L.d_type != 40 || L.tail_off == 0) return false;
    if (cap_entries <= 0) return false;
    // The 8-row bound is on ENTRIES PER GROUP, not on cap_entries: a group is one expert, an entry one (window row,
    // expert) pair, and the window holds at most kVerifyMaxT = 8 rows (verify.cpp bounds max_t by it), so a group has
    // at most 8 entries.  cap_entries is rows * top_k there (40 for the daily --spec 4), which is why it is not the
    // tile bound.  The kernel takes nent from grp_start on the device and refuses nothing: the invariant is the engine's.
    if (L.n_embd <= 0 || L.n_ff <= 0 || L.n_embd % 128 != 0 || L.n_ff % 16 != 0) return false;
    int dev = 0, maj = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return false; }
    if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return maj >= 7;
}

void qpn2_nvfp4_gu(const NativeExpertLayout& L, int64_t cap_groups, int64_t cap_entries, void* stream,
                   const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                   const int32_t* ent_tok, const void* x_q8_1, float* gate, float* up, int64_t grid_groups) {
    (void) cap_entries;
    const int64_t gy = (grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const dim3 grid((unsigned) (2 * L.n_ff / 32), (unsigned) gy);
    qpn2_nvfp4_kernel<true><<<grid, 32 * QPN2_SPLITK, 0, (cudaStream_t) stream>>>(
        grp_ptr, grp_start, n_groups, ent_tok, (const block_q8_1*) x_q8_1, L, gate, up);
}

void qpn2_nvfp4_down(const NativeExpertLayout& L, int64_t cap_groups, int64_t cap_entries, void* stream,
                     const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                     const int32_t* ent_dst, const void* hq_q8_1, float* out, int64_t grid_groups) {
    (void) cap_entries;
    const int64_t gy = (grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const dim3 grid((unsigned) (L.n_embd / 32), (unsigned) gy);
    qpn2_nvfp4_kernel<false><<<grid, 32 * QPN2_SPLITK, 0, (cudaStream_t) stream>>>(
        grp_ptr, grp_start, n_groups, ent_dst, (const block_q8_1*) hq_q8_1, L, out, out);
}

}  // namespace strata::kernels
