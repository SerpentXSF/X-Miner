#include "algo/etchash/etchash.hpp"

#include <cstring>

namespace xm::etchash {

int epoch_for_height(int height) { return height >= kEcip1099Block ? height / 60000 : height / 30000; }

int seed_epoch_for_height(int height) {
  // core-geth seedHash(epoch * epochLength + 1): keccak256 applied block / 30000 times.
  return height >= kEcip1099Block ? 2 * (height / 60000) : height / 30000;
}

std::shared_ptr<const Light> make_light(int height) {
  return kawpow::make_light(epoch_for_height(height), kDatasetParents, 1ull << 30, seed_epoch_for_height(height));
}

Result hash(const Light& light, const Hash256& header_hash, uint64_t nonce) {
  uint8_t in[40];
  std::memcpy(in, header_hash.data(), 32);
  for (int i = 0; i < 8; ++i) in[32 + i] = static_cast<uint8_t>(nonce >> (8 * i));  // little-endian
  const auto seed = kawpow::keccak512(in, 40);
  uint32_t s[16], mix[32];
  std::memcpy(s, seed.data(), 64);
  for (int i = 0; i < 32; ++i) mix[i] = s[i % 16];
  const uint32_t rows = uint32_t(light.dataset_items);  // 1024-bit items
  uint32_t item[32];
  for (uint32_t i = 0; i < 64; ++i) {
    const uint32_t p = kawpow::fnv1(i ^ s[0], mix[i % 32]) % rows;
    kawpow::dataset_item_512(light, uint64_t(p) * 2, item);
    kawpow::dataset_item_512(light, uint64_t(p) * 2 + 1, item + 16);
    for (int j = 0; j < 32; ++j) mix[j] = kawpow::fnv1(mix[j], item[j]);
  }
  Result r;
  for (int i = 0; i < 8; ++i)
    r.mix_hash[i] = kawpow::fnv1(kawpow::fnv1(kawpow::fnv1(mix[4 * i], mix[4 * i + 1]), mix[4 * i + 2]), mix[4 * i + 3]);
  uint8_t fin[96];
  std::memcpy(fin, seed.data(), 64);
  std::memcpy(fin + 64, r.mix_hash.data(), 32);
  const auto h = kawpow::keccak256(fin, 96);
  std::memcpy(r.final_hash.data(), h.data(), 32);
  return r;
}

}  // namespace xm::etchash
