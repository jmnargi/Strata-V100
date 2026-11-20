// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include "strata/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;      // the main session's flag (entry 0 of the device table)
int g_flags_host[kCvecFlags] = {};   // the whole table: [0] = g_on_host, [1 + b] = batch slot b's
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    g_flags_host[0] = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.s, g_s_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.on, sizeof(g_flags_host)) != cudaSuccess ||
        cudaMemcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, g_flags_host, sizeof(g_flags_host), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// one block per (stream, token): the pending write, then h . v over the stream, then the update
__global__ void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                            const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                            const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj,
                            int64_t inj_ld, int write, uint64_t sel) {
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    // the row's own flag: byte t of `sel` is its entry of the table (0 = the main session's, cvec_apply's every row;
    // 1 + b = batch slot b's, cvec_apply_rows)
    const int entry = (int) ((sel >> (8 * (int) (t & 7))) & 0xffu);
    const bool steer = on[entry] != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        __shared__ float part[THREADS / 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;   // a new vector replaces the old one on every device
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    for (int& f : g_flags_host) f = 1;   // every slot starts ON, like the main session
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

namespace {
void set_flag(int entry, bool on) {   // one entry of the table, on every device that holds it
    const int v = on ? 1 : 0;
    if (!g_cvec.loaded() || g_flags_host[entry] == v) return;
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on + entry, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_flags_host[entry] = v;
}
}  // namespace

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    set_flag(0, on);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_set_slot_enabled(int slot, bool on) {
    if (slot < 0 || slot >= kCvecSlots) throw std::runtime_error("cvec_set_slot_enabled: no such slot");
    set_flag(1 + slot, on);
}

bool cvec_slot_enabled(int slot) {
    return g_cvec.loaded() && slot >= 0 && slot < kCvecSlots && g_flags_host[1 + slot] != 0;
}

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) return false;
    *dir = t.dir; *s = t.s; *on = t.on;
    return true;
}

namespace {
void launch(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
            int64_t inj_ld, bool write, uint64_t sel, void* stream) {
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    cvec_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(R, t.dir, t.s, t.on, g_cvec.mode, layer,
                                                            (int) g_cvec.n_embd, (int) g_cvec.hc, r_ld, bo, bo_ld,
                                                            inj, inj_ld, write ? 1 : 0, sel);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
}
}  // namespace

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    launch(R, layer, T, r_ld, bo, bo_ld, inj, inj_ld, write, 0, stream);   // every row: the main session's flag
}

void cvec_apply_rows(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
                     const float* inj, int64_t inj_ld, bool write, const int* slots, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    if (T > 8) throw std::runtime_error("cvec_apply_rows: more than 8 rows");
    uint64_t sel = 0;
    for (int64_t t = 0; t < T; ++t) {
        if (slots[t] < 0 || slots[t] >= kCvecSlots) throw std::runtime_error("cvec_apply_rows: no such slot");
        sel |= (uint64_t) (1 + slots[t]) << (8 * t);
    }
    launch(R, layer, T, r_ld, bo, bo_ld, inj, inj_ld, write, sel, stream);
}

}  // namespace strata::kernels
