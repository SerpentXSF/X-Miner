// FishHash / KarlsenHashV2 against the published vectors (light-cache evaluation, ~3 s for the shared cache):
//   Iron Fish: ironfish-rust-nodejs/tests/fish_hash.test.slow.ts ("the quick brown fox..."), ironfish-rust
//   src/mining/mine.rs test_mine_batch_match_fish_hash (nonce 45 is the first of 43.. under the target).
//   Karlsen: rusty-karlsen crypto/hashes/src/pow_hashers.rs test_powb3hash / test_powfishhash / test_khashv2.
#include <cstring>
#include <string>

#include "algo/fishhash/fishhash.hpp"
#include "pool/kls_pool.hpp"
#include "test.hpp"

using namespace xm;

namespace {

const fishhash::Light& light() {
  static const auto L = fishhash::make_light();
  return *L;
}

std::string hex(const fishhash::Bytes32& h) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : h) s += d[b >> 4], s += d[b & 15];
  return s;
}

}  // namespace

TEST(fishhash_iron_vector) {
  const char* msg = "the quick brown fox jumps over the lazy dog";
  const auto h = fishhash::iron_hash(light(), reinterpret_cast<const uint8_t*>(msg), std::strlen(msg));
  CHECK_EQ(hex(h), std::string("6f4429716dc009d5d3b9775a4d6a5d58bccd9f73386bf88da7d5afdf5deb50f1"));
}

TEST(fishhash_iron_mine_batch) {
  uint8_t header[180];
  for (int i = 0; i < 180; ++i) header[i] = static_cast<uint8_t>(i % 10);
  const fishhash::Bytes32 target = {59,  125, 43,  4,   254, 19,  32,  88, 203, 188, 220, 43,  193, 139, 194, 164,
                                    61,  99,  44,  90,  151, 122, 236, 65, 253, 171, 148, 82,  130, 54,  122, 195};
  int first = -1;
  for (uint64_t n = 43; n < 46 && first < 0; ++n) {
    for (int b = 0; b < 8; ++b) header[172 + b] = static_cast<uint8_t>(n >> (56 - 8 * b));
    if (fishhash::be_less_equal(fishhash::iron_hash(light(), header, 180), target)) first = int(n);
  }
  CHECK_EQ(first, 45);
}

TEST(fishhash_karlsen_vectors) {
  fishhash::Bytes32 pre;
  pre.fill(42);
  const auto h1 = fishhash::karlsen_h1(pre, 5435345234ull, 432432432ull);
  CHECK_EQ(hex(h1), std::string("f0afbcd9401f24cf8374418e391e1458a6df7645441dd5dc9cfe8dc9e761e67d"));
  uint8_t seed[64] = {};
  std::memcpy(seed, h1.data(), 32);
  CHECK_EQ(hex(fishhash::kernel(light(), seed, fishhash::Variant::Karlsen)),
           std::string("f57e96fd7fef6accc5daacc9eaa1d012f9145da61488d884a8fa4ce6b57288be"));
  CHECK_EQ(hex(fishhash::karlsen_hash(light(), pre, 5435345234ull, 432432432ull)),
           std::string("71e8a7ff50f4eba67fbf00af449c12e6e74b1edfc1577b59c41c77922e546f87"));
}

// karlsen-stratum-bridge stratum_test.go / example_header.json: pre-PoW hash 853a0bb2...2ed2 as four little-endian
// u64 words, and the BzMiner form (hash hex + timestamp 1662696346 as 8 LE bytes).
TEST(karlsen_notify_forms) {
  fishhash::Job a, b;
  CHECK(kls_parse_notify("[\"7\", [2769687437080476293, 2642455852654975590, 16370749824715673019, "
                         "15145064868898544117], 1662696346]", a));
  CHECK(kls_parse_notify("[\"7\", \"853a0bb20ce86f2666da260099e3ab24bb4df7c83a9630e3f519f29a41142ed29abb1a6300000000\"]", b));
  CHECK_EQ(hex(a.pre_pow), std::string("853a0bb20ce86f2666da260099e3ab24bb4df7c83a9630e3f519f29a41142ed2"));
  CHECK(a.pre_pow == b.pre_pow);
  CHECK_EQ(a.timestamp, 1662696346ull);
  CHECK_EQ(b.timestamp, 1662696346ull);
  CHECK_EQ(a.id, std::string("7"));
}
