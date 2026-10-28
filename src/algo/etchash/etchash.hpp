// Etchash (Ethereum Classic) CPU reference: Ethash's hashimoto over the light cache, with ECIP-1099's 60000-block
// epochs from block 11,700,000 (epoch E: Ethash sizes of epoch E, the cache built from the seed of Ethash epoch 2E).
// Spec: core-geth consensus/ethash (algorithm.go, ecip1099), etclabscore go-etchash. Shares the Keccak, light cache
// and dataset code of the ProgPoW family (algo/kawpow, ported from cpp-kawpow / ethash, Apache-2.0).
#pragma once

#include <cstdint>
#include <memory>

#include "algo/kawpow/kawpow.hpp"

namespace xm::etchash {

using kawpow::Hash256;
using kawpow::Light;

constexpr int kEcip1099Block = 11700000;
constexpr int kDatasetParents = 256;

int epoch_for_height(int height);    // 60000-block epochs from the ECIP-1099 block, 30000 before
int seed_epoch_for_height(int height);  // Ethash epoch whose seed hash the pool sends (height / 30000)
std::shared_ptr<const Light> make_light(int height);

struct Result {
  Hash256 final_hash{};  // keccak256(seed || mix digest); a share when <= the target (bytes big-endian)
  Hash256 mix_hash{};
};
// hashimoto with dataset items computed from the light cache (about 1 ms).
Result hash(const Light& light, const Hash256& header_hash, uint64_t nonce);

}  // namespace xm::etchash
