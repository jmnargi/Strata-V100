// sm_70: the hyper-connection read timed as the engine runs it - captured into one CUDA graph, 96 reads (48
// layers x 2 halves) over 48 cold weight sets, replayed.  Prints us per read for T 1..6 and dumps the outputs of the
// last read so two builds / env settings can be compared (STRATA_BENCH_DUMP=file).
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv) {
    const int t_lo = argc > 1 ? std::atoi(argv[1]) : 1, t_hi = argc > 2 ? std::atoi(argv[2]) : 6;
    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8, SETS = 48, READS = 96;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.0f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float *dR, *dR0, *dbo, *dinj, *dxn, *dlo, *drs, *dio, *dmix;
    cudaMalloc(&dR, R.size() * 4); cudaMalloc(&dR0, R.size() * 4);
    cudaMalloc(&dbo, bo.size() * 4); cudaMalloc(&dinj, inj.size() * 4);
    cudaMalloc(&dxn, (size_t) TM * D * 4); cudaMalloc(&dlo, (size_t) TM * LR * 4); cudaMalloc(&drs, (size_t) TM * HC * 4);
    cudaMalloc(&dio, (size_t) TM * HC * 4); cudaMalloc(&dmix, (size_t) TM * N * 4);
    cudaMemcpy(dR0, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    std::vector<uint16_t*> cwd(SETS), cwu(SETS), cwi(SETS);
    std::vector<float*> cwn(SETS);
    for (int c = 0; c < SETS; ++c) {
        cudaMalloc(&cwd[c], wd.size() * 2); cudaMemcpy(cwd[c], wd.data(), wd.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&cwu[c], wu.size() * 2); cudaMemcpy(cwu[c], wu.data(), wu.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&cwi[c], wi.size() * 2); cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&cwn[c], wn.size() * 4); cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const char* dump = std::getenv("STRATA_BENCH_DUMP");
    if (dump) { FILE* f = std::fopen(dump, "wb"); if (f) std::fclose(f); }
    for (int T = t_lo; T <= t_hi; ++T) {
        auto args = [&](int rd, K::FusedGrArgs* a) {
            const int c = rd % SETS;
            for (int t = 0; t < T; ++t) {
                a[t] = K::FusedGrArgs();
                a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = (rd % 3) != 0;
                a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                a[t].w_norm = cwn[c]; a[t].w_down = cwd[c]; a[t].w_up = cwu[c]; a[t].w_inject = (rd % 2) ? cwi[c] : nullptr;
                a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;
            }
        };
        // correctness run (direct, no graph), from the same R
        cudaMemcpy(dR, dR0, R.size() * 4, cudaMemcpyDeviceToDevice);
        for (int rd = 0; rd < 6; ++rd) { K::FusedGrArgs a[8]; args(rd, a); K::fused_gr_read_multi(a, T, dxn, s); }
        cudaStreamSynchronize(s);
        if (dump) {
            std::vector<float> o((size_t) TM * (D + LR + HC + HC + N));
            float* p = o.data();
            cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * D;
            cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * LR;
            cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
            cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) TM * HC;
            cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost);
            for (int t = T; t < TM; ++t) {   // rows past T are scratch: blank them
                for (int i = 0; i < D; ++i) o[(size_t) t * D + i] = 0.f;
            }
            FILE* f = std::fopen(dump, "ab");
            if (f) { std::fwrite(o.data(), 4, o.size(), f); std::fclose(f); }
        }
        // the graph
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int rd = 0; rd < READS; ++rd) { K::FusedGrArgs a[8]; args(rd, a); K::fused_gr_read_multi(a, T, dxn, s); }
        if (cudaStreamEndCapture(s, &g) != cudaSuccess || cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) {
            std::printf("capture failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        for (int w = 0; w < 3; ++w) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            cudaEventRecord(e0, s);
            for (int i = 0; i < 10; ++i) cudaGraphLaunch(ge, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        const cudaError_t err = cudaGetLastError();
        std::printf("T %d | %.2f us per read (graph of %d reads, %d cold sets) %s\n", T, 1e3 * best / (10 * READS), READS, SETS,
                    err == cudaSuccess ? "" : cudaGetErrorString(err));
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
