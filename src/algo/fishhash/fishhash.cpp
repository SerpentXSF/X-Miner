#include "algo/fishhash/fishhash.hpp"

#include <cstring>

#include "crypto/blake3.hpp"

namespace xm::fishhash {

namespace {

constexpr uint8_t kSeed[32] = {0xeb, 0x01, 0x63, 0xae, 0xf2, 0xab, 0x1c, 0x5a, 0x66, 0x31, 0x0c,
                               0x1c, 0x14, 0xd6, 0x0f, 0x42, 0x55, 0xa9, 0xb3, 0x9b, 0x0e, 0xdf,
                               0x26, 0x53, 0x98, 0x44, 0xf1, 0x17, 0xad, 0x67, 0x21, 0x19};

void item_1024(const Light& light, uint32_t index, uint32_t out[32]) {
  kawpow::dataset_item_512(light, uint64_t(index) * 2, out);
  kawpow::dataset_item_512(light, uint64_t(index) * 2 + 1, out + 16);
}

}  // namespace

std::shared_ptr<const Light> make_light() {
  kawpow::Hash256 seed;
  std::memcpy(seed.data(), kSeed, 32);
  return kawpow::make_light_seeded(0, kParents, kCacheItems, kDatasetItems, seed);
}

Bytes32 kernel(const Light& light, const uint8_t seed[64], Variant v) {
  uint32_t mix[32];
  std::memcpy(mix, seed, 64);
  std::memcpy(mix + 16, seed, 64);
  uint32_t f0[32], f1[32], f2[32];
  for (uint32_t r = 0; r < kRounds; ++r) {
    uint32_t p0, p1, p2;
    if (v == Variant::Iron) {
      p0 = mix[0], p1 = mix[4], p2 = mix[8];
    } else {
      uint32_t g[8];
      for (int c = 0; c < 8; ++c) g[c] = mix[4 * c] ^ mix[4 * c + 1] ^ mix[4 * c + 2] ^ mix[4 * c + 3];
      p0 = g[0] ^ g[3] ^ g[6], p1 = g[1] ^ g[4] ^ g[7], p2 = g[2] ^ g[5] ^ r;
    }
    item_1024(light, p0 % kDatasetItems, f0);
    item_1024(light, p1 % kDatasetItems, f1);
    item_1024(light, p2 % kDatasetItems, f2);
    for (int j = 0; j < 32; ++j) {
      f1[j] = kawpow::fnv1(mix[j], f1[j]);
      f2[j] = mix[j] ^ f2[j];
    }
    for (int j = 0; j < 16; ++j) {
      uint64_t a, b, c;
      std::memcpy(&a, f0 + 2 * j, 8);
      std::memcpy(&b, f1 + 2 * j, 8);
      std::memcpy(&c, f2 + 2 * j, 8);
      const uint64_t m = a * b + c;
      std::memcpy(mix + 2 * j, &m, 8);
    }
  }
  uint32_t out[8];
  for (int i = 0; i < 8; ++i)
    out[i] = kawpow::fnv1(kawpow::fnv1(kawpow::fnv1(mix[4 * i], mix[4 * i + 1]), mix[4 * i + 2]), mix[4 * i + 3]);
  Bytes32 r;
  std::memcpy(r.data(), out, 32);
  return r;
}

Bytes32 iron_hash(const Light& light, const uint8_t* header, size_t len) {
  const auto seed = blake3::hash_xof(header, len, 64);
  const Bytes32 mix = kernel(light, seed.data(), Variant::Iron);
  uint8_t fin[96];
  std::memcpy(fin, seed.data(), 64);
  std::memcpy(fin + 64, mix.data(), 32);
  return blake3::hash(fin, 96);
}

Bytes32 karlsen_h1(const Bytes32& pre_pow, uint64_t timestamp, uint64_t nonce) {
  uint8_t in[80] = {};
  std::memcpy(in, pre_pow.data(), 32);
  std::memcpy(in + 32, &timestamp, 8);  // little-endian
  std::memcpy(in + 72, &nonce, 8);
  return blake3::hash(in, 80);
}

Bytes32 karlsen_hash(const Light& light, const Bytes32& pre_pow, uint64_t timestamp, uint64_t nonce) {
  uint8_t seed[64] = {};
  const Bytes32 h1 = karlsen_h1(pre_pow, timestamp, nonce);
  std::memcpy(seed, h1.data(), 32);
  const Bytes32 h2 = kernel(light, seed, Variant::Karlsen);
  return blake3::hash(h2.data(), 32);
}

bool be_less_equal(const Bytes32& a, const Bytes32& b) { return std::memcmp(a.data(), b.data(), 32) <= 0; }

bool le_less_equal(const Bytes32& a, const Bytes32& b) {
  for (int i = 31; i >= 0; --i)
    if (a[i] != b[i]) return a[i] < b[i];
  return true;
}

}  // namespace xm::fishhash
