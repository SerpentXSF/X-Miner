// Pearl reference against the official pinned values (pearl-research-labs/pearl, ISC).
#include <string>

#include "algo/pearl/pearl.hpp"
#include "test.hpp"

using namespace xm;
using namespace xm::pearl;

namespace {

std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}
std::string hex(const Hash& h) { return hex(h.data(), 32); }

Hash fill(uint8_t v) {
  Hash h;
  h.fill(v);
  return h;
}

Pattern from(std::vector<uint32_t> v) {
  Pattern p;
  bool ok = Pattern::from_list(v, p);
  CHECK(ok);
  return p;
}

}  // namespace

TEST(pearl_seed_salts) {  // seed.rs SEED_SALT_A / SEED_SALT_B
  CHECK_EQ(hex(seed_salt_a()), std::string("8249406ca0ed15169616f692fcf076f892dbdb2a7023b852f0d47719c390017b"));
  CHECK_EQ(hex(seed_salt_b()), std::string("11300632ec6301ca2be2af718b3f4d4f1ae9c63988e8cc044844301d71b89aa9"));
}

TEST(pearl_seed_chain_pinned_vectors) {  // seed.rs commitment_hash_pinned_vectors, (m, n) = (192, 320)
  Seeds legacy = commitment_seeds(fill(0x11), fill(0xAA), fill(0xBB), 192, 320, false);
  CHECK_EQ(hex(legacy.b_noise_seed), std::string("add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48971861d12fcb0177"));
  CHECK_EQ(hex(legacy.a_noise_seed), std::string("483b07b6f73105030b9482255f37723f3fed69ae916724ee8291848b8c28794b"));
  Seeds salted = commitment_seeds(fill(0x11), fill(0xAA), fill(0xBB), 192, 320, true);
  CHECK_EQ(hex(salted.b_noise_seed), std::string("60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5"));
  CHECK_EQ(hex(salted.a_noise_seed), std::string("301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf"));
}

TEST(pearl_pattern_bytes) {
  auto b = Pattern::contiguous(16).to_bytes();
  CHECK_EQ(hex(b.data(), 6), std::string("000f00000000"));
  auto p = from({0, 32});
  CHECK_EQ(hex(p.to_bytes().data(), 6), std::string("1f0100000000"));
  CHECK_EQ(p.period(), 64u);
  std::vector<uint32_t> c64;
  for (uint32_t i = 0; i < 64; ++i) c64.push_back(i);
  CHECK_EQ(hex(from(c64).to_bytes().data(), 6), std::string("003f00000000"));
  CHECK_EQ(hex(from({0, 8}).to_bytes().data(), 6), std::string("070100000000"));
  // [0,32] offsets: valid iff t mod 64 < 32
  CHECK(p.offset_is_valid(0) && p.offset_is_valid(31) && !p.offset_is_valid(32) && p.offset_is_valid(64));
  Pattern back;
  auto pb = p.to_bytes();
  CHECK(Pattern::from_bytes(pb.data(), back) && back == p);
  uint8_t bad[6] = {0, 0, 1, 0, 0, 0};  // non-canonical
  CHECK(!Pattern::from_bytes(bad, back));
}

TEST(pearl_config_bytes) {
  Config c;
  c.k = 4096;
  c.rank = 128;
  c.rows = from({0, 32});
  std::vector<uint32_t> c64;
  for (uint32_t i = 0; i < 64; ++i) c64.push_back(i);
  c.cols = from(c64);
  auto b = c.to_bytes();
  CHECK_EQ(hex(b.data(), 20), std::string("00100000800000001f0100000000003f00000000"));
  for (int i = 20; i < 52; ++i) CHECK_EQ(int(b[i]), 0);
}

TEST(pearl_nbits) {  // proof_utils.rs nbits tests
  CHECK_EQ(nbits_to_target(0x1d00ffff).hex_be(),
           std::string("00000000ffff0000000000000000000000000000000000000000000000000000"));
  CHECK_EQ(nbits_to_target(0x1b0404cb).hex_be(),
           std::string("00000000000404cb000000000000000000000000000000000000000000000000"));
  CHECK_EQ(nbits_to_target(0x03123456).hex_be(),
           std::string("0000000000000000000000000000000000000000000000000000000000123456"));
  CHECK(nbits_to_target(0x1d000000).is_zero());  // zero mantissa
  CHECK(nbits_to_target(0x1d800000).is_zero());  // sign bit
  CHECK_EQ(target_to_nbits(nbits_to_target(0x1b0404cb)), 0x1b0404cbu);
  CHECK_EQ(target_to_nbits(nbits_to_target(0x1d00ffff)), 0x1d00ffffu);
  // HeroMiners' fixed share target 0xFFFF * 2^208 / 2^21 is exactly compact 0x1a07fff8 (spec 4.4).
  U256 t = nbits_to_target(0x1d00ffff);  // 0xffff * 2^208
  U256 d21;  // divide by 2^21 = shift right 21
  for (int i = 0; i < 8; ++i) {
    uint64_t v = (uint64_t(i + 1 < 8 ? t.w[i + 1] : 0) << 32 | t.w[i]) >> 21;
    d21.w[i] = static_cast<uint32_t>(v);
  }
  CHECK_EQ(target_to_nbits(d21), 0x1a07fff8u);
}

TEST(pearl_bound_scales_with_tile_and_rank) {
  Config c;  // 16x16 contiguous, k 4096, r 128 -> factor 256*4096
  U256 b = difficulty_bound(0x1d00ffff, c);
  CHECK(b == nbits_to_target(0x1d00ffff).mul_sat(256ull * 4096));
  Config r256 = c;
  r256.rank = 256;  // rank penalty: (dpl/r)*128 = half of dpl
  CHECK(difficulty_bound(0x1d00ffff, r256) == nbits_to_target(0x1d00ffff).mul_sat(256ull * 16 * 128));
}

TEST(pearl_base64) {
  CHECK_EQ(base64({}), std::string(""));
  CHECK_EQ(base64({'f'}), std::string("Zg=="));
  CHECK_EQ(base64({'f', 'o'}), std::string("Zm8="));
  CHECK_EQ(base64({'f', 'o', 'o'}), std::string("Zm9v"));
}

TEST(pearl_merkle_root_equals_keyed_hash) {
  // The proof tree root must equal the plain keyed BLAKE3 of the padded matrix (merkle.rs test), for many sizes.
  Hash key = fill(0x5a);
  for (size_t rows : {1u, 2u, 3u, 5u, 6u, 7u, 9u, 17u}) {
    std::vector<int8_t> m(rows * 1024);
    for (size_t i = 0; i < m.size(); ++i) m[i] = static_cast<int8_t>((i * 31 + rows) & 0x7f) - 64;
    auto p = matrix_proof(key, m.data(), rows, 1024, {0});
    CHECK_EQ(hex(p.root), hex(matrix_root(key, m.data(), rows, 1024)));
    CHECK_EQ(p.total_leaves, uint64_t(rows));
  }
}
