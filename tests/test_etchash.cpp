// Etchash / Ethash hashimoto. Vector: core-geth consensus/ethash/algorithm_test.go TestHashimoto (a 1 KB cache from
// a zero seed and a 32 KB dataset).
#include <cstring>
#include <string>

#include "algo/etchash/etchash.hpp"
#include "test.hpp"

using namespace xm;

TEST(etchash_epochs) {
  CHECK_EQ(etchash::epoch_for_height(11699999), 389);
  CHECK_EQ(etchash::epoch_for_height(11700000), 195);
  CHECK_EQ(etchash::seed_epoch_for_height(11700000), 390);
  CHECK_EQ(etchash::epoch_for_height(25624706), 427);
}

TEST(etchash_hashimoto_small) {
  kawpow::Light L;
  L.parents = 256;
  L.cache_items = 1024 / 64;
  L.dataset_items = 32 * 1024 / 128;
  L.cache.resize(size_t(L.cache_items) * 16);
  // Ethash cache generation (as kawpow::make_light) from a zero seed.
  auto item = [&](int i) { return reinterpret_cast<uint8_t*>(&L.cache[size_t(i) * 16]); };
  uint8_t zero[32] = {};
  auto h = kawpow::keccak512(zero, 32);
  std::memcpy(item(0), h.data(), 64);
  for (int i = 1; i < L.cache_items; ++i) {
    h = kawpow::keccak512(item(i - 1), 64);
    std::memcpy(item(i), h.data(), 64);
  }
  const uint32_t n = uint32_t(L.cache_items);
  for (int round = 0; round < 3; ++round)
    for (int i = 0; i < L.cache_items; ++i) {
      const uint32_t v = L.cache[size_t(i) * 16] % n, w = uint32_t(L.cache_items + i - 1) % n;
      uint8_t x[64];
      for (int k = 0; k < 64; ++k) x[k] = item(int(v))[k] ^ item(int(w))[k];
      h = kawpow::keccak512(x, 64);
      std::memcpy(item(i), h.data(), 64);
    }
  const auto r = etchash::hash(L, kawpow::hash_from_hex("c9149cc0386e689d789a1c2f3d5d169a61a6218ed30e74414dc736e442ef3d1f"), 0);
  CHECK_EQ(kawpow::hash_to_hex(r.mix_hash), std::string("e4073cffaef931d37117cefd9afd27ea0f1cad6a981dd2605c4a1ac97c519800"));
  CHECK_EQ(kawpow::hash_to_hex(r.final_hash), std::string("d3539235ee2e6f8db665c0a72169f55b7f6c605712330b778ec3944f0eb5a557"));
}
