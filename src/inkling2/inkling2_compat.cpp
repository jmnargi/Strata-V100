// src/inkling2/inkling2_compat.cpp - the two hooks the Inkling-2 engine asks this build for, which has neither a BF16
// tensor-core path in prefill::Gemm nor the grouped WMMA expert kernel: the reservation is unnecessary, and the grouped
// call answers "not available" so the engine takes its own fallback (native_expert_grouped).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/gemm.hpp"

namespace strata::prefill {
void Gemm::reserve_bf16_tc(int64_t, int64_t) {}
}  // namespace strata::prefill

namespace strata::kernels {
bool native_expert_grouped_wmma(const NativeExpertLayout&, const unsigned long long*, const int32_t*, const int32_t*,
                                const int32_t*, const int32_t*, int64_t, int64_t, const void*, void*, float*, void*,
                                int64_t) {
    return false;
}
}  // namespace strata::kernels
