// sm_70: native_mmvq / native_mmvq_il timed as the engine runs them - captured in a graph over cold weight copies.
//   mmvq_graph_bench <ggml type> <n_in> <n_out> [T lo] [T hi] [il 0/1]
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace K = strata::kernels;

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: type n_in n_out [Tlo] [Thi] [il]\n"); return 1; }
    const int type = std::atoi(argv[1]), n_in = std::atoi(argv[2]), n_out = std::atoi(argv[3]);
    const int t_lo = argc > 4 ? std::atoi(argv[4]) : 1, t_hi = argc > 5 ? std::atoi(argv[5]) : 4;
    const int il = argc > 6 ? std::atoi(argv[6]) : 1;
    const size_t wb = K::native_mmvq_weight_bytes(type, n_in, n_out);
    const int sets = (int) std::max<size_t>(8, (size_t) 600e6 / wb), calls = 96;
    std::mt19937 rng(3);
    std::vector<uint8_t> h(wb);
    for (auto& b : h) b = (uint8_t) (rng() & 0xff);
    for (size_t i = 0; i + 1 < wb; i += 2) if ((h[i + 1] & 0x7c) == 0x7c) h[i + 1] &= 0xbf;   // fewer inf/nan halves
    std::vector<void*> w(sets);
    for (auto& p : w) { cudaMalloc(&p, wb); cudaMemcpy(p, h.data(), wb, cudaMemcpyHostToDevice); }
    float *x, *y;
    void *xq, *xil;
    cudaMalloc(&x, (size_t) 8 * n_in * 4);
    cudaMalloc(&y, (size_t) 8 * n_out * 4);
    cudaMalloc(&xq, (size_t) 8 * n_in * 2);
    cudaMalloc(&xil, K::native_q8_1_il_bytes(n_in, 4) * 2);
    std::vector<float> hx((size_t) 8 * n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : hx) v = nd(rng);
    cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::printf("type %d %d x %d: %.2f MB per matrix, %d cold sets\n", type, n_in, n_out, wb / 1e6, sets);
    for (int T = t_lo; T <= t_hi; ++T) {
        if (T >= 2 && T <= 4 && il) K::native_quantize_q8_1_il(x, xq, xil, n_in, T, s);
        else K::native_quantize_q8_1(x, xq, n_in, T, s);
        cudaStreamSynchronize(s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int c = 0; c < calls; ++c) {
            if (T >= 2 && T <= 4 && il) K::native_mmvq_il(type, w[c % sets], xq, xil, y, n_in, n_out, T, s);
            else K::native_mmvq(type, w[c % sets], xq, y, n_in, n_out, T, s);
        }
        if (cudaStreamEndCapture(s, &g) != cudaSuccess || cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) {
            std::printf("capture failed %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
        }
        for (int i = 0; i < 3; ++i) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            best = std::min(best, ms);
        }
        const double us = 1e3 * best / (5 * calls);
        std::printf("T %d: %7.2f us per call  %6.0f GB/s %s\n", T, us, wb / us / 1e3, cudaGetErrorString(cudaGetLastError()));
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    return 0;
}
