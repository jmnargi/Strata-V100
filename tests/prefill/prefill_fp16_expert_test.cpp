// #100: execute Prefill's FP16 fallback with unequal expert slices. Out-of-order execution
// must not read another expert's activations, including H rows still poisoned with NaNs.
#include "fp16_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(__float2half_rn(x)); }
float value(uint16_t x) { return __half2float(std::bit_cast<__half>(x)); }
template<class T> struct Device {
    T* ptr = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&ptr, n * sizeof(T))); }
    ~Device() { (void) cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& x) { check(cudaMemcpy(ptr, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    void download(std::vector<T>& x) { check(cudaMemcpy(x.data(), ptr, x.size() * sizeof(T), cudaMemcpyDeviceToHost)); }
};

int run() {
    constexpr int N = 2560, F = 640, rows = 10, capacity = rows + 2;
    constexpr std::array<int, 3> offsets{0, 2, 7}, counts{2, 5, 3};
    std::vector<uint16_t> x(capacity * N);
    std::array<std::vector<uint16_t>, 3> wgu, wd;
    for (int r = 0; r < capacity; ++r)
        for (int k = 0; k < N; ++k) x[r * N + k] = half(((r * 11 + k * 3) % 23 - 11) / 16.0f);
    for (int e = 0; e < 3; ++e) {
        wgu[e].resize(2 * F * N);
        wd[e].resize(N * F);
        for (int f = 0; f < F; ++f) {
            wgu[e][(2 * f) * N + f % N] = half((e + 1) / 4.0f);
            wgu[e][(2 * f + 1) * N + (f + 7) % N] = half((e + 2) / 8.0f);
        }
        for (int n = 0; n < N; ++n)
            for (int f = 0; f < F; ++f)
                wd[e][n * F + f] = half(((n * 7 + f * 3 + e * 5) % 17 - 8) / 64.0f);
    }
    // Independent scalar reference, including the FP16 SwiGLU store before the down product.
    std::vector<float> expected(rows * N);
    for (int e = 0; e < 3; ++e) {
        for (int r = offsets[e]; r < offsets[e] + counts[e]; ++r) {
            std::vector<float> h(F);
            for (int f = 0; f < F; ++f) {
                double gate = 0, up = 0;
                for (int k = 0; k < N; ++k) {
                    gate += value(x[r * N + k]) * value(wgu[e][2 * f * N + k]);
                    up += value(x[r * N + k]) * value(wgu[e][(2 * f + 1) * N + k]);
                }
                h[f] = value(half(static_cast<float>(gate / (1.0 + std::exp(-gate)) * up)));
            }
            for (int n = 0; n < N; ++n) {
                double sum = 0;
                for (int f = 0; f < F; ++f) sum += h[f] * value(wd[e][n * F + f]);
                expected[r * N + n] = static_cast<float>(sum);
            }
        }
    }
    Device<uint16_t> dx(x.size()), dgu(2 * F * N), dd(N * F), dh(capacity * F);
    Device<float> gu(capacity * 2 * F), dm(capacity * N);
    dx.upload(x);
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(nullptr, 0, error)) throw std::runtime_error(error);
    int failures = 0;
    float max_error = 0;
    // Ascending catches finite cross-expert contamination. Starting at offset 7 also catches
    // an unwritten prefix; the second pass reuses the same allocations with different poison.
    const std::array<std::array<int, 3>, 2> orders{{{0, 1, 2}, {2, 0, 1}}};
    for (int pass = 0; pass < 2; ++pass) {
        const uint16_t poison = half(pass ? std::numeric_limits<float>::quiet_NaN() : 13.0f);
        dh.upload(std::vector<uint16_t>(capacity * F, poison));
        std::vector<float> actual(capacity * N, -123.0f);
        dm.upload(actual);
        for (int e : orders[pass]) {
            dgu.upload(wgu[e]);
            dd.upload(wd[e]);
            strata::prefill::detail::fp16_expert(gemm, dx.ptr, dgu.ptr, dd.ptr, gu.ptr, dh.ptr, dm.ptr,
                                                offsets[e], counts[e], N, false, 0, nullptr, nullptr, [] {});
            // Weight scratch is reused only after this expert has completed.
            check(cudaDeviceSynchronize());
        }
        dm.download(actual);
        for (int i = 0; i < rows * N; ++i) {
            const float delta = std::fabs(actual[i] - expected[i]);
            if (!std::isfinite(actual[i]) || delta > 2e-4f + 2e-3f * std::fabs(expected[i])) ++failures;
            if (std::isfinite(delta)) max_error = std::fmax(max_error, delta);
        }
        for (int i = rows * N; i < capacity * N; ++i)
            if (actual[i] != -123.0f) ++failures;
        std::vector<uint16_t> h(capacity * F);
        dh.download(h);
        for (int i = rows * F; i < capacity * F; ++i)
            if (h[i] != poison) ++failures;
    }
    std::printf("FP16 expert fallback: 2 orders, 3 experts, %d output checks, failures=%d, max_abs_error=%.9g\n",
                2 * rows * N, failures, static_cast<double>(max_error));
    return failures ? 1 : 0;
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU device");
        return 77;
    }
    try { return run(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "FP16 expert fallback: %s\n", e.what());
        return 1;
    }
}
