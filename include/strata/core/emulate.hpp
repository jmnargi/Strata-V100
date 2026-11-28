// strata/core/emulate.hpp - STRATA_EMULATE_CC: another GPU generation's code paths, run on this card (tests only).
//
// A build whose device code is PTX for an older architecture (CMAKE_CUDA_ARCHITECTURES=86-virtual) runs on a newer
// card through the driver's JIT, and every kernel then sees __CUDA_ARCH__ = 860.  What the host decides from the
// device must agree with that, or it picks paths and shared-memory sizes the device code was not built for:
// STRATA_EMULATE_CC=75|80|86|89 makes each such query (compute capability, shared memory per block) answer as that
// card would.  Speed under it says nothing about the real card; results do - the kernels are the real card's.
#pragma once

#include "strata/core/volta_forever.hpp"

#include <cstdlib>

namespace strata {

/// the emulated compute capability as 10 * major + minor, or 0 (off)
inline int emulated_cc() {
    static const int cc = [] {
        const char* e = std::getenv("STRATA_EMULATE_CC");
        const int v = e ? std::atoi(e) : 0;
        return v >= 75 && v < 120 ? v : 0;
    }();
    return cc;
}
inline int cc_major_of(int real_major) { return emulated_cc() ? emulated_cc() / 10 : real_major; }
inline int cc_minor_of(int real_minor) {
    if (const int e = emulated_cc()) return e % 10;
    // volta-forever retargets this image for a Tesla V100 and tells the program it runs a 7.5 card (the shim only
    // exists for sm_70).  7.5 becomes 7.0 here, so the host chooses the Volta kernels the image carries instead of
    // Turing's, which that card has to emulate (docs/VOLTA_FOREVER.md).  The shim runs Volta images only, so no card
    // it reports is a real Turing.
    if (real_minor == 5 && core::volta_forever_active()) return 0;
    return real_minor;
}
/// cudaDevAttrMaxSharedMemoryPerBlockOptin as the emulated card reports it: 64 KB on Turing, 163 KB on the A100,
/// 99 KB on the other Ampere and Ada cards
inline int smem_optin_of(int real) {
    switch (emulated_cc()) {
        case 0: return real;
        case 75: return 65536;
        case 80: return 166912;
        default: return real < 101376 ? real : 101376;
    }
}

}  // namespace strata
