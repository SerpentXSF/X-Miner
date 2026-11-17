// Beam BeamHash III CPU reference: personalised BLAKE2b (vector from Python's hashlib), the SipHash variant (Grin's
// keys-as-state SipHash-2-4, as in our tested Cuckaroo29 code), index packing, the packed difficulty, and the verifier
// rejecting tampered solutions. XM_BEAM_CPU_SOLVE=1 also runs the slow CPU solver (~8 GB) and verifies its output.
#include <cstdlib>
#include <string>
#include <vector>

#include "algo/beam/beam.hpp"
#include "algo/c29/c29.hpp"
#include "test.hpp"

using namespace xm;

namespace {

std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> v;
  for (size_t i = 0; i + 1 < s.size(); i += 2) v.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
  return v;
}

}  // namespace

TEST(beam_prepow) {
  uint8_t msg[44];
  for (int i = 0; i < 44; ++i) msg[i] = static_cast<uint8_t>(i);
  const beam::PrePow p = beam::pre_pow(msg, msg + 32, msg + 40);
  // hashlib.blake2b(bytes(range(44)), digest_size=32, person=b"Beam-PoW" + (448).to_bytes(4, "little") +
  // (5).to_bytes(4, "little"))
  const auto d = unhex("0cf85f0a62f451e4177b3c2e42fd6a2ecb35b6700c268621dd6961e8db382129");
  for (int w = 0; w < 4; ++w) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v |= uint64_t(d[8 * w + b]) << (8 * b);
    CHECK_EQ(p.v[w], v);
  }
}

TEST(beam_siphash_is_grin_siphash) {
  const beam::PrePow p{{0x0123456789abcdefull, 0xfedcba9876543210ull, 0x1111222233334444ull, 0x5555666677778888ull}};
  const c29::Keys k{p.v[0], p.v[1], p.v[2], p.v[3]};
  const uint64_t nonces[] = {0, 1, 7, 0x12345678, (uint64_t(beam::kElements - 1) << 3) + 6};
  for (uint64_t n : nonces)
    CHECK_EQ(beam::siphash(p, n), c29::siphash24(k, n, 21));
}

TEST(beam_index_packing) {
  beam::Indices idx{};
  for (int i = 0; i < beam::kIndices; ++i) idx[i] = (uint32_t(i) * 2654435761u) & (beam::kElements - 1);
  const auto packed = beam::pack_indices(idx);
  const auto back = beam::unpack_indices(packed.data());
  for (int i = 0; i < beam::kIndices; ++i) CHECK_EQ(back[i], idx[i]);
  // Field 0 occupies the low 25 bits of bytes 0..3.
  beam::Indices one{};
  one[0] = 0x1abcdef;
  const auto p1 = beam::pack_indices(one);
  CHECK_EQ(int(p1[0]), 0xef);
  CHECK_EQ(int(p1[1]), 0xcd);
  CHECK_EQ(int(p1[2]), 0xab);
  CHECK_EQ(int(p1[3]), 0x01);
}

TEST(beam_lengths) {
  CHECK_EQ(beam::mix_len(1), 448);
  CHECK_EQ(beam::out_len(1), 424);
  CHECK_EQ(beam::mix_len(4), 376);
  CHECK_EQ(beam::out_len(4), 288);
  CHECK_EQ(beam::mix_len(5), 288);
  CHECK_EQ(beam::out_len(5), 24);
  CHECK_EQ(beam::mix_indices(1), 1);
  CHECK_EQ(beam::mix_indices(2), 2);
  CHECK_EQ(beam::mix_indices(3), 4);
  CHECK_EQ(beam::mix_indices(4), 6);
  CHECK_EQ(beam::mix_indices(5), 9);
}

TEST(beam_difficulty) {
  uint8_t d[32] = {};
  d[0] = 0xff;  // the largest digests
  for (int i = 1; i < 32; ++i) d[i] = 0xff;
  CHECK(beam::target_reached(d, 0));           // difficulty 1: every solution
  CHECK(!beam::target_reached(d, 1u << 24));   // order 1: hv < 2^255
  d[0] = 0x7f;
  CHECK(beam::target_reached(d, 1u << 24));
  d[0] = 0x00;
  d[1] = 0xff;
  CHECK(beam::target_reached(d, 8u << 24));    // hv < 2^248
  d[0] = 0x01;
  CHECK(!beam::target_reached(d, 8u << 24));
  CHECK(!beam::target_reached(d, (233u << 24)));  // above "infinite": invalid
  CHECK_NEAR(beam::difficulty_value(0), 1.0, 1e-12);
  CHECK_NEAR(beam::difficulty_value((10u << 24) | 0x800000), 1536.0, 1e-9);
}

TEST(beam_verify_rejects) {
  uint8_t input[32] = {}, nonce[8] = {};
  beam::Solution s{};
  CHECK(!beam::verify(input, nonce, s));  // all-zero indices: duplicates
  beam::Indices idx{};
  for (int i = 0; i < beam::kIndices; ++i) idx[i] = uint32_t(i);
  const auto packed = beam::pack_indices(idx);
  std::copy(packed.begin(), packed.end(), s.begin());
  CHECK(!beam::verify(input, nonce, s));  // no collisions
}

TEST(beam_cpu_solver) {
  if (!std::getenv("XM_BEAM_CPU_SOLVE")) return;
  uint8_t input[32], nonce[8];
  for (int i = 0; i < 32; ++i) input[i] = static_cast<uint8_t>(0x10 + 3 * i);
  int total = 0;
  for (int n = 0; n < 3; ++n) {
    for (int i = 0; i < 8; ++i) nonce[i] = static_cast<uint8_t>(i == 0 ? n : 0);
    const auto sols = beam::solve_cpu(input, nonce);
    std::printf("  nonce %d: %zu solutions\n", n, sols.size());
    for (const auto& s : sols) {
      CHECK(beam::verify(input, nonce, s));
      beam::Solution t = s;
      t[5] ^= 0x10;  // any changed index breaks it
      CHECK(!beam::verify(input, nonce, t));
      std::printf("    ");
      for (auto c : s) std::printf("%02x", c);
      std::printf("\n");
    }
    total += int(sols.size());
  }
  CHECK(total > 0);
}

// Beam stratum job parsing (message shape from BeamMW/beam pow/stratum.h).
#include "pool/beam_pool.hpp"

TEST(beam_pool_job_parsing) {
  beam::Job j;
  CHECK(beam_parse_job(R"({"method":"job","id":"7731","input":")"
                       "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
                       R"(","difficulty":117440512,"height":3100200,"jsonrpc":"2.0"})", j));
  CHECK_EQ(j.id, std::string("7731"));
  CHECK_EQ(int(j.input[31]), 0x1f);
  CHECK_EQ(j.difficulty, 117440512u);  // order 7: 128 solutions per share
  CHECK_NEAR(beam::difficulty_value(j.difficulty), 128.0, 1e-9);
  CHECK_EQ(j.height, uint64_t(3100200));
  CHECK(beam_parse_job(R"({"method":"job","id":42,"input":")"
                       "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
                       R"(","difficulty":0})", j));
  CHECK_EQ(j.id, std::string("42"));
  CHECK(!beam_parse_job(R"({"method":"job","id":"1","input":"0001","difficulty":0})", j));  // short input
  CHECK(!beam_parse_job(R"({"method":"job","id":"1","input":")"
                        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
                        R"(","difficulty":4278190080})", j));  // order 255: above "infinite"
}
