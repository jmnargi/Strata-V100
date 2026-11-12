// The FP16 fallback after expert weights have been dequantized. Kept internal so the numerical
// regression exercises the same row slicing as Prefill, without loading a model.
#pragma once

#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace strata::prefill::detail {

template<class BeforeDown>
inline void fp16_expert(Gemm& gemm, const uint16_t* xs, const uint16_t* wgu, const uint16_t* wd,
                        float* gu, uint16_t* h, float* dm, int64_t offset, int64_t rows, int64_t width,
                        bool had_down, int had_block, const float* had_signs, void* stream,
                        BeforeDown before_down) {
    gemm.f16(xs + offset * width, wgu, gu + offset * 1280, rows, 1280, width);
    swiglu_interleaved(gu + offset * 1280, h + offset * 640, rows, stream);
    // Rotate in FP16 storage: the GEMM reads FP16 either way. This adds only the SwiGLU
    // store's rounding, rather than requiring a T*K*640 float buffer for the fallback.
    if (had_down)
        strata::kernels::hadamard_rows_f16(h + offset * 640, rows, 640, had_block, had_signs, stream);
    before_down();
    gemm.f16(h + offset * 640, wd, dm + offset * width, rows, width, 640);
}

}  // namespace strata::prefill::detail
