// Quantus QPoW reference. Vectors: qp-poseidon src/poseidon2.rs (P3_ZERO_RESULT, P3_SEQ_RESULT) and the official
// miner's NONCE_HASH_KVS (quantus-miner crates/pow-core/src/lib.rs), both checked there against qpow-math.
#include <string>

#include "algo/quantus/quantus.hpp"
#include "test.hpp"

using namespace xm::qtc;

namespace {

std::string hexs(const uint8_t* p, size_t n) {
  static const char* x = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += x[p[i] >> 4];
    s += x[p[i] & 15];
  }
  return s;
}

template <size_t N>
std::array<uint8_t, N> unhex(const std::string& h) {
  std::array<uint8_t, N> v{};
  for (size_t i = 0; i < N; ++i) v[i] = static_cast<uint8_t>(std::stoi(h.substr(2 * i, 2), nullptr, 16));
  return v;
}

}  // namespace

TEST(quantus_permutation_vectors) {
  uint64_t z[12] = {};
  permute(z);
  const uint64_t zero[12] = {0xc9bc9432e1686884, 0x03ecbab0dcdd2189, 0x5e7ac885b3dc1215, 0x6ac07513801d191f,
                             0xca5c593fb184dcfc, 0x414dec5f3e455287, 0x1a17df170127ae41, 0xe7e592bd0af9b0a5,
                             0xc71a9b27edc66a4c, 0x2728671759ac43c2, 0xb9969c20f7f672f9, 0xc5140b586823b92f};
  for (int i = 0; i < 12; ++i) CHECK_EQ(z[i], zero[i]);
  uint64_t q[12];
  for (int i = 0; i < 12; ++i) q[i] = i + 1;
  permute(q);
  const uint64_t seq[12] = {0x7e9574e2a3d6c48b, 0x9d7bc16d282d2f2b, 0x798826626d94a498, 0x0831011bb22304c7,
                            0xbdccb5931fffd16c, 0xe98687714dacbefc, 0xc6a1ed29dd75e027, 0x1aec96681d15f765,
                            0xc74b2c710b170a23, 0x5fb4aff45e9c24fb, 0x1fb3d228db0127eb, 0xe201a7e214b16e74};
  for (int i = 0; i < 12; ++i) CHECK_EQ(q[i], seq[i]);
}

TEST(quantus_nonce_hash_vectors) {
  const char* kv[][3] = {
      {"0000000000000000000000000000000000000000000000000000000000000000",
       "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
       "0000000000000000",
       "8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f6"
       "91a0d4f6c607e8cc"},
      {"0101010101010101010101010101010101010101010101010101010101010101",
       "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
       "0000000000000001",
       "e4ef79db80b642a093d7e38e6a6daac7ec1cca7293ddbe3710b4e6781be4add0b93f7e3440626df5bb23b757436787ba8d3fd0a2652e387f"
       "8af5bd8a5cc2ce2f"},
      {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
       "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
       "1234567890abcdef",
       "7f6c410ce62e7fc54811ad3b8b92664ac0f8b1d2cd6eb163129251146e7ee34d27794aaae1b31eaaeb98ab835bae6a2b9ac4ad6f08c95903"
       "e1423f0b865816ff"},
      {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
       "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
       "ffffffffffffffff",
       "822f70999ee517f610b51bbb79ba88bfbb19cb77b316df25d8aeb4c2682673555e9fc6a588ee96dcce4ff0d2b8d97f4aa89d0f46c97f9027"
       "b386d42b14be5100"},
      {"0707070707070707070707070707070707070707070707070707070707070707",
       "00000000000000000000000000000000000000000deadbeefcafe000000000000000000000000000000000000000000000000000000000001"
       "234567890abcdef",
       "eab81cdb33db293aef7e1f10f85cb08d17bc197c284d46a25d9dc57cdb7ba878adf3f5c354318121876795aab08c0941afda21d835383faa"
       "2f139c26be31a30a"},
  };
  for (auto& k : kv) {
    auto header = unhex<32>(k[0]);
    auto nonce = unhex<64>(k[1]);
    Hash h = nonce_hash(header.data(), nonce);
    CHECK_EQ(hexs(h.data(), 64), std::string(k[2]));
  }
}

TEST(quantus_targets) {
  Hash h{};
  Target t{};
  t[4] = 0x3c;
  CHECK(meets_target(h, t));
  h[4] = 0x3c;
  CHECK(!meets_target(h, t));  // equal fails: the chain's check is strict
  h[4] = 0x3b;
  h[63] = 0xff;
  CHECK(meets_target(h, t));
}
