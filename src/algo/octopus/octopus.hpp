// Conflux Octopus CPU reference, ported from conflux-rust crates/cfxcore/pow (compute.rs, shared.rs, cache.rs;
// GPL-3.0 reference, re-implemented from the algorithm). Per nonce: 32 values of a degree-1023 polynomial mod
// 1032193 whose coefficients come from SipHash of the 32 nonces of the nonce's aligned group, evaluated at points
// fixed by the header hash; their FNV digest seeds an Ethash-style mix of 32 random 256-byte DAG pages.
// The light cache and DAG items are Ethash's (256 parents) with Octopus sizes: 2^19-block stages, a 4 GB + 16 MB per
// stage dataset of prime 256-byte pages and a 16 MB + 64 KB per stage cache.
#pragma once

#include <cstdint>
#include <memory>

#include "algo/kawpow/kawpow.hpp"

namespace xm::octopus {

using kawpow::Hash256;
using kawpow::Light;

constexpr uint32_t kMod = 1032193;  // 63 * 2^14 + 1
constexpr uint32_t kModB = 11;
constexpr uint64_t kStageLength = 1ull << 19;
constexpr int kN = 1024;             // polynomial coefficients
constexpr int kGroup = 32;           // nonces sharing one polynomial (the reference's warp)
constexpr int kPoints = kN / kGroup; // values per nonce
constexpr int kAccesses = 32;
constexpr int kDatasetParents = 256;

inline int stage_for_height(uint64_t height) {
  return height / kStageLength > 0x7fffffffull ? 0x7fffffff : static_cast<int>(height / kStageLength);
}
int cache_items(int stage);    // 512-bit items
int dataset_pages(int stage);  // 256-byte pages (the DAG is this many x 256 bytes)
std::shared_ptr<const Light> make_light(int stage);  // Light::dataset_items = 2 x pages (1024-bit items)

// The 1024 evaluation points of a header: x_k = a w^2k + b w^k + c (mod p); nonce n uses x_{n%32 + 32t}, t < 32.
// xs_k = floor(x_k 2^32 / p) for Shoup's modular multiply.
struct Points {
  uint32_t x[kN];
  uint32_t xs[kN];
  uint64_t v[4];  // the header as four little-endian words (the SipHash key)
};
void make_points(const Hash256& header, Points& out);

// Octopus hash (a light evaluation: about 128 dataset items from the cache, ~1 ms).
Hash256 hash(const Light& light, const Points& pts, const Hash256& header, uint64_t nonce);

// The pool's share rule for nonces below 2^128: hash < boundary, both 256-bit big-endian.
bool below(const Hash256& hash, const Hash256& boundary);

}  // namespace xm::octopus
