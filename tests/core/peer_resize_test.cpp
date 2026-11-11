#include "strata/core/peer_experts.hpp"
#include "strata/core/vmm.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int L = 2, E = 32;
constexpr int64_t MiB = 1ll << 20;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
void cuda_ok(cudaError_t status, const char* what) {
    require(status == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(status));
}
struct Source : strata::core::ExpertSource {
    std::vector<uint8_t> data;
    size_t bytes = strata::kernels::cpu::BLOB;
    Source() : data((size_t) L * E * strata::kernels::cpu::BLOB) {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (uint8_t) (((i / bytes) * 31 + i % bytes) % 251);
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        return data.data() + (size_t) (layer * E + expert) * bytes;
    }
};
int check_rows(strata::core::PeerExperts& peer, Source& src, const std::vector<int32_t>& primary) {
    cuda_ok(cudaSetDevice(peer.device()), "peer device");
    std::vector<uint8_t> back(src.bytes);
    int count = 0;
    for (int l = 0; l < L; ++l) {
        for (int e = 0; e < E; ++e) {
            const bool has = peer.has(l, e);
            const uint8_t* ptr = peer.slot_ptr(l, e);
            require(has == (ptr != nullptr), "slot address and residency disagree");
            if (!has) continue;
            require(primary[(size_t) l * E + e] < 0, "expert duplicated on primary and peer");
            cuda_ok(cudaMemcpy(back.data(), ptr, back.size(), cudaMemcpyDeviceToHost), "read resident expert");
            require(std::memcmp(back.data(), src.blob(l, e), back.size()) == 0, "resident expert has stale/wrong bytes");
            ++count;
        }
    }
    require(count == peer.resident(), "resident counter disagrees with routable experts");
    require(count <= peer.slots(), "residency points beyond live backing");
    cuda_ok(cudaSetDevice(0), "primary device");
    return count;
}
void queue_swap(strata::core::PeerExperts& peer, const std::vector<int32_t>& primary) {
    std::vector<float> usage(L * E, 0.f);
    int candidate = -1;
    for (int l = 0; l < L && candidate < 0; ++l)
        for (int e = 0; e < E; ++e)
            if (primary[(size_t) l * E + e] < 0 && !peer.has(l, e)) {
                candidate = l * E + e;
                break;
            }
    require(candidate >= 0, "test needs a nonresident candidate");
    usage[(size_t) candidate] = 1000.f;
    const int64_t before = peer.swaps();
    std::string err;
    require(peer.adapt(usage.data(), primary.data(), 1, err), err);
    require(peer.swaps() == before + 1, "adaptive refill was not queued");
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: peer resize needs two CUDA GPUs");
        return 77;
    }
    for (int device = 0; device < 2; ++device) {
        cudaSetDevice(device);
        if (!strata::core::vmm_available()) {
            std::puts("SKIP: peer resize needs CUDA VMM");
            return 77;
        }
    }
    try {
        cuda_ok(cudaSetDevice(0), "primary device");
        strata::core::ExpertCache::set_vmm(false);
        Source src;
        strata::core::ExpertCache primary;
        std::string err;
        require(primary.open(4, L, E, (int64_t) src.bytes, err), err);
        std::vector<int32_t> res0(L * E, -1);
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int e = 0; e < E; ++e)
            for (int l = 0; l < L; ++l) ranked.emplace_back(l, e);
        for (size_t i = 0; i < 4; ++i) {
            const auto [l, e] = ranked[i];
            const int32_t slot = primary.admit(l, e);
            require(slot >= 0 && primary.fill_slot_blocking(slot, src.blob(l, e), err), err);
            res0[(size_t) l * E + e] = slot;
        }
        strata::core::PeerExperts peer;
        require(peer.open(1, ranked, primary, src, L, E, 128, 32, err, 4), err);
        const int64_t full = peer.full_slots();
        require(full == 32 && check_rows(peer, src, res0) == full, "startup residency incorrect");

        // Shrink while an adaptive H2D is pending: no slot may be unmapped under that copy.
        queue_swap(peer, res0);
        cuda_ok(cudaSetDevice(1), "peer device");
        size_t free = 0, total = 0;
        cuda_ok(cudaMemGetInfo(&free, &total), "free memory");
        const int64_t reserve = (int64_t) (free / MiB) + 16;
        cuda_ok(cudaSetDevice(0), "primary device");
        require(peer.resize_reserve(reserve, res0.data(), err), err);
        require(peer.slots() < full, "reserve did not release backing");
        check_rows(peer, src, res0);

        // Adapt into the remaining cache, then restore: no duplicate or wrong-layer rows.
        queue_swap(peer, res0);
        peer.apply_pending(true);
        check_rows(peer, src, res0);
        require(peer.resize_reserve(0, res0.data(), err), err);
        require(peer.slots() == full && check_rows(peer, src, res0) == full, "restore lost resident experts");

        // A peer with no backed slots must safely route every expert as a miss, then recover.
        cuda_ok(cudaSetDevice(1), "peer device");
        cuda_ok(cudaMemGetInfo(&free, &total), "free memory");
        const int64_t all = (int64_t) ((free + (size_t) peer.mapped_bytes() - MiB) / MiB);
        cuda_ok(cudaSetDevice(0), "primary device");
        require(peer.resize_reserve(all, res0.data(), err), err);
        require(peer.slots() == 0 && check_rows(peer, src, res0) == 0, "empty cache still routes a resident");
        require(peer.resize_reserve(0, res0.data(), err), err);
        require(peer.slots() == full && check_rows(peer, src, res0) == full, "empty-cache restore failed");

        const int64_t prior_slots = peer.slots();
        const int64_t prior_resident = peer.resident();
        require(!peer.resize_reserve((int64_t) (total / MiB) + 1024, res0.data(), err), "impossible reserve accepted");
        require(peer.slots() == prior_slots && peer.resident() == prior_resident, "rejected reserve damaged residency");
        check_rows(peer, src, res0);

        peer.close();
        require(peer.open(1, ranked, primary, src, L, E, 128, 8, err), err);
        require(!peer.resize_reserve(0, res0.data(), err), "ordinary unsegmented peer unexpectedly resizes");
        require(check_rows(peer, src, res0) == 8, "ordinary peer changed after rejected resize");
        std::puts("PASS: pending refill, shrink, adapt, regrow, empty cache, rejected reserve, old mode");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
