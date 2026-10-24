// strata/core/nv_defaults.hpp - per-architecture defaults for NVIDIA sm_86, sm_70 and sm_60.
//
// 148-nv measured every switch of the 0.1.45 bit-exact ON set, one at a time and in the stacked set, on an RTX 3060, an RTX A4000
// (sm_86) and a Tesla P100 (sm_60): a switch is in a table only when its token ids equal the default's and it was faster on that
// architecture (5+ interleaved pairs of fresh engines, noise arm, equal expert-cache slots; notes-148-nv.md).  The identity gate
// (10 prompts + exact_pp 4K/20K, fixed flags) is IDENTICAL with that original table on. Switches that changed bits,
// hung or lost speed were rejected from that measured set.
//
// Two sm_70 entries extend that set with a different contract:
//   * STRATA_ADAPT_OVERLAP=1 on Linux CUDA moves the adaptive copies under the next window. The implicit copy budget
//     can change cache residency; it does not guarantee identical token IDs. An explicit --adapt-swaps N fixes the count.
//   * STRATA_PF_WMMA2=1 selects the second prompt-expert tile inside the opt-in STRATA_PF_WMMA=1 path. It keeps
//     FP16 operands and FP32 sums; a run without WMMA is unchanged.
//
//   STRATA_SM86_PERF=0   turns the sm_86 table off (cards of compute capability 8.6: RTX 3050-3090, RTX A2000-A6000)
//   STRATA_SM70_PERF=0   turns the sm_70 table off (Volta: Tesla V100, Titan V)
//   STRATA_SM60_PERF=0   turns the sm_60 table off (Pascal GP100: Tesla P100)
//
// The sm_70 table on Linux CUDA also turns on adaptive overlap (STRATA_ADAPT_OVERLAP=1): the copies land while the next
// verify window runs. The serve log names the state and the reason when it cannot run. A layer
// split, an unusable buffer, a resident RAM source without STRATA_ADAPT_OVERLAP_RESIDENT=1, or the user's own
// STRATA_ADAPT_OVERLAP setting (0 included) keeps the previous behaviour.  Without --adapt-swaps the round's count
// follows the window (include/strata/core/adapt_budget.hpp); an explicit --adapt-swaps N stays fixed.
//
// A switch the user set (to anything, including 0) is never overridden.  Applied once at start, before any engine code reads its
// switches; the table is used only when every visible CUDA device has the same compute capability.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of a card with compute capability major.minor (raw, as the driver reports it);
/// empty for every other card and when its off switch is 0.
std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor);

/// Sets `nv_default_env(major, minor)` in this process's environment where the user has not set the variable, prints one line to
/// stderr when it set anything, and returns the names it set.
std::vector<std::string> apply_nv_defaults(int major, int minor);

}  // namespace strata::core

namespace strata::core {

/// dflash2-kern: the speculative runs' faster start-up adaptation of the expert tier (opt-in on sm_86 with STRATA_SPEC_ADAPT_SM86=1 since owner round 9; STRATA_SPEC_ADAPT_EVERY / _WARM, read by
/// generate.cpp): sm_86 only (z440 A4000 Q2_0; the P100 is opt-in through the two variables, notes-dflash2-kern.md).  Not part of
/// the exact table above: the residency it changes decides which experts the GPU rounds, as the adaptive tier always does.  The user's
/// own settings and the cards' off switches (STRATA_SM86_PERF=0) is kept.
void apply_spec_adapt_default(int major, int minor);

}  // namespace strata::core
