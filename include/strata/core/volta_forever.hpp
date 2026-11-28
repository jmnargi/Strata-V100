// include/strata/core/volta_forever.hpp - detect nbald/volta-forever, the libcuda shim that runs a CUDA 13 image on a
// Tesla V100 (sm_70), and the two settings the engine has to make under it (docs/VOLTA_FOREVER.md).
//
// Why this exists:
//   1. CUDA 13 dropped sm_70.  The shim interposes libcuda.so.1 and retargets the image's sm_75 PTX (or translates its
//      sm_75 machine code) to sm_70 when the module loads.  It tells the program it runs a 7.5 card - the card is a
//      Volta - so strata/core/emulate.hpp remaps that back to 7.0 (see cc_minor_of there).
//   2. Strata defaults CUDA_MODULE_LOADING to EAGER to reserve the kernel code before the expert cache takes the free
//      VRAM.  Under the shim EAGER translates EVERY kernel of the image before the banner prints: a minute or more of
//      startup, and kernels the model never calls are translated too.  LAZY is the setting the layer wants.
//
// Detection, in order: STRATA_VOLTA_FOREVER (1 forces on, 0 forces off: the way to exercise this path without the
// layer), VOLTA_FOREVER, the shim's own `volta_forever_version` symbol, then "libvolta_forever" in LD_PRELOAD (an older
// shim).
#pragma once

#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace strata::core {

/// The shim's version string, or nullptr when it is not loaded.
inline const char* volta_forever_version_str() noexcept {
#if !defined(_WIN32)
    using Fn = const char* (*)();
    void* sym = dlsym(RTLD_DEFAULT, "volta_forever_version");
    if (sym != nullptr) {
        Fn fn = nullptr;
        std::memcpy(&fn, &sym, sizeof(sym));
        return fn ? fn() : "loaded";
    }
#endif
    return nullptr;
}

/// The detection itself, from the values it reads (so a test can pass its own).
inline bool volta_forever_active_from(const char* explicit_env, const char* ld_preload_env, const char* dlsym_version,
                                     const char* vf_env = nullptr) noexcept {
    if (explicit_env != nullptr && explicit_env[0] != '\0') return explicit_env[0] != '0';
    if (vf_env != nullptr && vf_env[0] != '\0' && std::strcmp(vf_env, "0") != 0) return true;
    if (dlsym_version != nullptr && dlsym_version[0] != '\0') return true;
    if (ld_preload_env != nullptr && std::strstr(ld_preload_env, "libvolta_forever") != nullptr) return true;
    return false;
}

inline bool volta_forever_active() noexcept {
    static const bool on = volta_forever_active_from(std::getenv("STRATA_VOLTA_FOREVER"), std::getenv("LD_PRELOAD"),
                                                     volta_forever_version_str(), std::getenv("VOLTA_FOREVER"));
    return on;
}

/// CUDA_MODULE_LOADING's default: LAZY under the shim, EAGER otherwise.
constexpr const char* default_cuda_module_loading(bool vf_active) noexcept { return vf_active ? "LAZY" : "EAGER"; }

/// Sets CUDA_MODULE_LOADING when the caller left it alone.  Returns true when the shim was detected (and LAZY set).
inline bool apply_default_cuda_module_loading() noexcept {
    const bool vf = volta_forever_active();
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", default_cuda_module_loading(vf));
#else
        setenv("CUDA_MODULE_LOADING", default_cuda_module_loading(vf), 0);
#endif
        return vf;
    }
    return false;
}

}  // namespace strata::core
