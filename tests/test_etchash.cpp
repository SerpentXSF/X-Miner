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

TEST(ethash_epochs_without_ecip1099) {
  CHECK_EQ(etchash::epoch_for_height(25358333, false), 845);
  CHECK_EQ(etchash::seed_epoch_for_height(25358333, false), 845);
  CHECK_EQ(etchash::epoch_for_height(11700000, false), 390);
}

// EthereumPoW (plain Ethash) main-chain blocks from mainnet.ethereumpow.org: {height, seal hash (keccak256 of the header
// RLP without mix and nonce; the full RLP re-hashed to each block's hash), nonce, mix digest, difficulty}.
TEST(ethash_ethw_blocks) {
  struct Block {
    int height;
    const char* seal;
    uint64_t nonce;
    const char* mix;
    double difficulty;
  };
  const Block blocks[] = {
      {15600000, "a7a24e032a2200002d5054895ee381193c69c1d28ac83c9388c6631ea9664222", 0x58581d000c8349ccull,
       "3bb0050d6de5b6ad9e2dffce1d4e7e2bd607b2cae9262fa3acac290c1d3af879", 446883286259446.0},
      {20000000, "2a4601bc104845675596ba5517d7db33633df5211c99da3cd732c6c1e3433fae", 0x1e9af4983d363252ull,
       "6aa39c185bffd5ab88fc79e887b011c7c6d13ae0255e66766e997b96d3bf5832", 245787976649898.0},
      {25358000, "bf430426bcdbae5b7ab461c03aa264565ec4081007405e2b3e621d892fe558a2", 0x295138d24c191fd6ull,
       "e5463580cbc4415d6512db8d2b8aebcf2d322f778b99a20e1eb971c2ae952d2e", 27281007939820.0},
  };
  for (const auto& b : blocks) {
    const auto light = etchash::make_light(b.height, false);
    const auto r = etchash::hash(*light, kawpow::hash_from_hex(b.seal), b.nonce);
    CHECK_EQ(kawpow::hash_to_hex(r.mix_hash), std::string(b.mix));
    // The result meets the block's difficulty: top 64 bits (big-endian) <= 2^64 / difficulty.
    const uint8_t* f = reinterpret_cast<const uint8_t*>(r.final_hash.data());
    uint64_t top = 0;
    for (int i = 0; i < 8; ++i) top = (top << 8) | f[i];
    CHECK(double(top) <= 18446744073709551616.0 / b.difficulty);
  }
}
