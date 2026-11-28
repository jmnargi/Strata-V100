# cmake/strata_volta_ptx_ggml.cmake - the ggml MMQ kernels for a STRATA_VOLTA_PTX build (docs/VOLTA_FOREVER.md).
#
# A STRATA_VOLTA_PTX image is sm_75 PTX that a PTX-retargeting layer (volta-forever) compiles for sm_70, so ggml must
# carry the Volta instruction set: `__dp4a` dots, not Turing's `mma.m16n8k8` int8 tensor cores and `ldmatrix`, which a
# V100 has to emulate - the layer says so when it loads (`128 integer tensor-core products, 32 ldmatrix` per kernel),
# and one of those emulated MMQ kernels was measured to fault (`mul_mat_q<GGML_TYPE_Q8_0, 64, false>`:
# CUDA_ERROR_ILLEGAL_ADDRESS on the first prompt read).
#
# ggml decides this in two places that must agree, or the host launches a kernel whose tile layout the device does not
# compute:
#   common.cuh: TURING_MMA_AVAILABLE (device) and turing_mma_available/volta_mma_available (host)
#   mmq.cuh:   ggml_cuda_mmq_get_config's NVIDIA branch (the SMEM layout and tile sizes), host and device
#
# The patch is anchored on the pinned llama.cpp (its commit is in the FetchContent declaration in CMakeLists.txt); a
# missing anchor stops the configure rather than shipping an image that emulates what the card does natively.
#
#   strata_volta_ptx_ggml(<llama.cpp dir>)   ->  nothing; FATAL_ERROR when the sources are not the pinned ones

function(strata_volta_ptx_ggml LLAMA)
  set(_cu "${LLAMA}/ggml/src/ggml-cuda")
  if(NOT EXISTS "${_cu}/mmq.cuh")
    message(FATAL_ERROR "STRATA_VOLTA_PTX: ${_cu}/mmq.cuh is missing (the llama.cpp sources the engine builds against)")
  endif()
  file(READ "${_cu}/mmq.cuh" _probe)
  string(FIND "${_probe}" "STRATA_VOLTA_PTX" _already)
  if(NOT _already LESS 0)
    message(STATUS "Strata: STRATA_VOLTA_PTX - ggml's MMQ kernels are already patched in ${_cu}")
    return()
  endif()
  macro(_vp_read VAR PATH)
    if(NOT EXISTS "${PATH}")
      message(FATAL_ERROR "STRATA_VOLTA_PTX: ${PATH} is missing (the llama.cpp sources the engine builds against)")
    endif()
    file(READ "${PATH}" ${VAR})
    string(REPLACE "\r\n" "\n" ${VAR} "${${VAR}}")
  endmacro()
  macro(_vp_edit VAR ANCHOR TEXT)     # ANCHOR -> TEXT (the anchor must occur exactly once)
    string(FIND "${${VAR}}" "${ANCHOR}" _p)
    if(_p LESS 0)
      message(FATAL_ERROR "STRATA_VOLTA_PTX: ggml anchor not found, the pinned llama.cpp changed: ${ANCHOR}")
    endif()
    string(LENGTH "${ANCHOR}" _alen)
    math(EXPR _after "${_p} + ${_alen}")
    string(SUBSTRING "${${VAR}}" ${_after} -1 _tail)
    string(FIND "${_tail}" "${ANCHOR}" _p2)
    if(NOT _p2 LESS 0)
      message(FATAL_ERROR "STRATA_VOLTA_PTX: ggml anchor is not unique: ${ANCHOR}")
    endif()
    string(SUBSTRING "${${VAR}}" 0 ${_p} _head)
    set(${VAR} "${_head}${TEXT}${_tail}")
  endmacro()

  # ---- common.cuh: the device macros --------------------------------------------------------------------------------
  _vp_read(_c "${_cu}/common.cuh")
  _vp_edit(_c "#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA\n#define VOLTA_MMA_AVAILABLE"
    "#if !defined(GGML_USE_HIP) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || defined(STRATA_VOLTA_PTX))\n#define VOLTA_MMA_AVAILABLE")
  _vp_edit(_c "#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING\n#define TURING_MMA_AVAILABLE"
    "#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(STRATA_VOLTA_PTX)\n#define TURING_MMA_AVAILABLE")
  # ---- common.cuh: the host-side answers, which the dispatch reads --------------------------------------------------
  _vp_edit(_c "static bool volta_mma_available(const int cc) {\n    return GGML_CUDA_CC_IS_NVIDIA(cc)"
    "static bool volta_mma_available(const int cc) {\n#if defined(STRATA_VOLTA_PTX)\n    (void) cc;\n    return true;\n#endif\n    return GGML_CUDA_CC_IS_NVIDIA(cc)")
  _vp_edit(_c "static bool turing_mma_available(const int cc) {\n    return GGML_CUDA_CC_IS_NVIDIA(cc)"
    "static bool turing_mma_available(const int cc) {\n#if defined(STRATA_VOLTA_PTX)\n    (void) cc;\n    return false;\n#endif\n    return GGML_CUDA_CC_IS_NVIDIA(cc)")
  file(WRITE "${_cu}/common.cuh" "${_c}")

  # ---- mmq.cuh: the SMEM layout and tile sizes, host and device ------------------------------------------------------
  _vp_read(_m "${_cu}/mmq.cuh")
  _vp_edit(_m "    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {\n        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);\n    }\n    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {\n        return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);\n    }"
    "#if defined(STRATA_VOLTA_PTX)\n    // the image carries the Volta instruction set (dp4a), not Turing's mma\n    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {\n        return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);\n    }\n#else\n    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {\n        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);\n    }\n    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {\n        return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);\n    }\n#endif")
  _vp_edit(_m "#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA\n    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);"
    "#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && !defined(STRATA_VOLTA_PTX)\n    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);")
  file(WRITE "${_cu}/mmq.cuh" "${_m}")

  message(STATUS "Strata: STRATA_VOLTA_PTX - ggml's MMQ kernels take their Volta (dp4a) path in ${_cu}")
endfunction()
