// src/inkling2/inkling2_quant_compat.cpp - the four host-side quant entry points the Inkling-2 engine uses, over the
// decoder the Strata Inkling engine already carries for exactly these block types (Q8_0, Q5_K, Q6_K, IQ3_XXS, IQ4_XS,
// NVFP4).  A type outside that set answers "cannot dequantise" rather than decoding it wrongly.
#include "strata/quant/dequant.hpp"

#include "ik2_kernels.cuh"

namespace strata::quant {
namespace {
const TypeInfo kTable[] = {
    {Type::F32, "f32", 1, 4, Type::F32, false, false},
    {Type::F16, "f16", 1, 2, Type::F16, false, false},
    {Type::BF16, "bf16", 1, 2, Type::BF16, false, false},
    {Type::Q8_0, "q8_0", 32, 34, Type::Q8_1, true, true},
    {Type::Q8_1, "q8_1", 32, 36, Type::Q8_1, false, true},
    {Type::Q8_K, "q8_K", 256, 292, Type::Q8_K, false, true},
    {Type::Q5_K, "q5_K", 256, 176, Type::Q8_K, true, false},
    {Type::Q6_K, "q6_K", 256, 210, Type::Q8_K, true, false},
    {Type::IQ3_XXS, "iq3_xxs", 256, 98, Type::Q8_K, true, false},
    {Type::IQ4_XS, "iq4_xs", 256, 136, Type::Q8_K, true, false},
    {Type::NVFP4, "nvfp4", 64, 36, Type::Q8_1, true, false},
};
}  // namespace

const TypeInfo* type_info(int type) noexcept {
    for (const auto& e : kTable)
        if ((int) e.type == type) return &e;
    return nullptr;
}

size_t row_bytes(int type, int64_t n) noexcept {
    const TypeInfo* t = type_info(type);
    if (!t || n <= 0 || n % t->blck) return 0;
    return (size_t) (n / t->blck) * t->bytes;
}

bool can_dequant(Type t) noexcept {
    switch (t) {
        case Type::Q8_0: case Type::Q5_K: case Type::Q6_K: case Type::IQ3_XXS: case Type::IQ4_XS: case Type::NVFP4:
            return true;
        default:
            return false;
    }
}

bool dequant_row(Type t, const void* src, float* dst, int64_t n) {
    if (!can_dequant(t) || n <= 0) return false;
    ik2::k::decode_host((int) t, (const uint8_t*) src, (size_t) n, dst);
    return true;
}
}  // namespace strata::quant
