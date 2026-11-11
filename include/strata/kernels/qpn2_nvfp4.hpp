// include/strata/kernels/qpn2_nvfp4.hpp - the optional W4A16 NVFP4 decode experts for Volta (see qpn2_nvfp4.cu).
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>

namespace strata::kernels {

/// Whether the QPN2 kernels take this layout: NVFP4 gate/up and down, an M of 1..8 rows a group, dimensions the
/// 32-column tiles and the 64-value K split divide, and the instruction (sm_70+).  The caller still decides on the
/// environment variable: the default path stays byte-identical.
bool qpn2_nvfp4_supported(const NativeExpertLayout& L, int64_t cap_entries) noexcept;

/// Gate/up of every group into `gate` / `up` (row e * n_ff + n, entries indexed from the group's grp_start), with the
/// expert's s_gate / s_up applied - the same buffers `launch_gu` fills.
void qpn2_nvfp4_gu(const NativeExpertLayout& L, int64_t cap_groups, int64_t cap_entries, void* stream,
                   const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                   const int32_t* ent_tok, const void* x_q8_1, float* gate, float* up, int64_t grid_groups = 0);

/// Down of every group from `hq_q8_1` (q8_1, one row an entry) into `out` at row ent_dst[e] * n_embd, s_down applied.
void qpn2_nvfp4_down(const NativeExpertLayout& L, int64_t cap_groups, int64_t cap_entries, void* stream,
                     const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                     const int32_t* ent_dst, const void* hq_q8_1, float* out, int64_t grid_groups = 0);

}  // namespace strata::kernels
