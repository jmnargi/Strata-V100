// include/strata/kernels/volta_isa.hpp - "is this a Volta (sm_70) target?" as a compile-time constant.
//
// The kernels were written for compute capability 7.5 and newer, and some paths are better on Volta: the prompt
// attention's `mma.sync.m8n8k4`, fewer `__dp4a` calls in the i-quant dots, shuffles instead of `ldmatrix` and plain
// copies instead of `cp.async`.  Each of those is chosen at compile time from `__CUDA_ARCH__`, because a build for
// 7.5+ must not contain instructions its cards cannot run.
//
// Two macros answer the question, and every site that used to test `__CUDA_ARCH__` directly uses one of them:
//
//   STRATA_VOLTA_ISA        the Volta sequences: sm_70, or a STRATA_VOLTA_PTX build.
//   STRATA_PRE_TURING_ISA   the pre-Turing sequences: anything below 7.5, or a STRATA_VOLTA_PTX build.
//
// STRATA_VOLTA_PTX (-DSTRATA_VOLTA_PTX=ON) is for a CUDA 13 build on a V100.  CUDA 13 cannot compile for sm_70, so
// the image is built for sm_75 and carries our Volta sequences as PTX; a layer that retargets that PTX to sm_70 when
// the module loads (volta-forever, docs/VOLTA_FOREVER.md) then runs them on the card's own tensor cores and byte
// dots instead of emulating Turing's.  The build also sets STRATA_EXPERIMENTAL_SM60, so the Volta-only kernel set
// that a sm_70 build compiles is present here too.
//
// The host must agree with the image.  A kernel whose Volta body was compiled out holds a `__trap()`, so the host
// may only select it where the image really carries Volta code: STRATA_HOST_VOLTA_ISA, which CMakeLists.txt sets
// from the architecture list (1 for a sm_70 target, 70 in a multi-arch list, or any STRATA_VOLTA_PTX build; 0
// otherwise).  A sm_75-only build run on Volta - through a PTX layer or STRATA_EMULATE_CC=70 - then keeps the
// generic kernel instead of launching a trap.
#pragma once

#if !defined(STRATA_HOST_VOLTA_ISA)
// .cpp translation units and any file built without the CUDA rules: CMake always defines it for the engine's
// targets, so this is only the portable default for a stray include.
#    define STRATA_HOST_VOLTA_ISA 0
#endif

#if defined(STRATA_VOLTA_PTX)
#    define STRATA_VOLTA_ISA 1
#    define STRATA_PRE_TURING_ISA 1
#else
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
#        define STRATA_VOLTA_ISA 1
#    else
#        define STRATA_VOLTA_ISA 0
#    endif
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#        define STRATA_PRE_TURING_ISA 1
#    else
#        define STRATA_PRE_TURING_ISA 0
#    endif
#endif
