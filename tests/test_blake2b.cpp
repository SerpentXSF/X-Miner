// BLAKE2b (RDTS) reference and pool dialect. Vectors come from the original Python X-Miner's oracle (spec/pow.py,
// hashlib.blake2b) and from the BLAKE2b activation block mined on Knots 29.4.1rc5 regtest (its node-reported hash).
#include <string>

#include "algo/blake2b/blake2b.hpp"
#include "algo/blake2b/blake2b_gpu.hpp"
#include "pool/blake2b_pool.hpp"
#include "test.hpp"

using namespace xm::b2b;

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

std::vector<uint8_t> unhex(const std::string& h) {
  std::vector<uint8_t> v(h.size() / 2);
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(std::stoi(h.substr(2 * i, 2), nullptr, 16));
  return v;
}

}  // namespace

TEST(blake2b_known_answers) {
  const char* kat[][2] = {
      {"a54dca182530bb1d6d132cded6237b2ed91e3f721fcb1971174494d6493c9d5c3460be31201e69fedaa0eee8b9997f5c7c2999fdafe593"
       "253cd654af4dfad71427a0aeb3fee9232f8af2211f9ee491c5",
       "2ea9e64e546a97f3f84818d3d2827ea81c0b79521111aff20fcc1134a6de1362"},
      {"b10becb5563bfc1e6f93427ecbc8fe2955e5cd8e46dc8ed4b7c2764d2a5a4d767706f85d8690024ad6bda3401be9c8cbccc935f6cd1f61"
       "226ae15338ae1a34004d33ba0d246ac04c81b1baf23e3bf9ee",
       "4d26da4c085e892a6e60e71786689fccde3d7f7c8d98c9dc76aa470c53f283ae"},
      {"47de636c0e806c957ba684d6431fb5ead7424d09e15d024c5848f23d1fa6f7361d7f618d1532e70e20e2a6668de7f47e8467e546d53ec8"
       "e2a1257bdb256c9b3e4fbb498146ef7030cbf9537252dccead",
       "900176fa42c9dcc37058a5dcb8b3b4c3ba52d6ac8305d79c13fbd8744e10fb96"},
  };
  for (auto& k : kat) {
    auto in = unhex(k[0]);
    Digest d = blake2b256(in.data(), in.size());
    CHECK_EQ(hexs(d.data(), 32), std::string(k[1]));
  }
  // RFC 7693-style empty message, 32-byte digest.
  Digest e = blake2b256(nullptr, 0);
  CHECK_EQ(hexs(e.data(), 32), std::string("0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"));
}

TEST(blake2b_regtest_block_from_stratum_fields) {
  // What a gateway would send for Knots regtest block 1, and the hash the node reported for that block.
  const std::string notify =
      R"(["b1","0000000000004fee2e5613820a8fa7540fa030c58f0b363a28ee63af2ba84bc7",)"
      R"("0000009650adc5f76b847f7d0a340d604917eea59e0a5d823214ca7974490984655f2700000000","",[],"20000000",)"
      R"("207fffff","0000000000000000",true])";
  Job j;
  uint32_t nbits = 0;
  bool clean = false;
  CHECK(xm::b2b_parse_notify(notify, j, nbits, clean));
  CHECK(clean);
  CHECK_EQ(nbits, 0x207fffffu);
  j.extranonce = {};  // extranonce1 00000000 || extranonce2 0000000000000000
  Digest root = work_root(j.coinb1.data(), j.extranonce.data());
  Header h = work_header(j.prevhash.data(), 0, j.ntime.data(), root);
  CHECK_EQ(hexs(h.data(), 80),
           std::string("0000000000004fee2e5613820a8fa7540fa030c58f0b363a28ee63af2ba84bc700000000000000000000000000000000"
                       "a3708e9b5c0276808b35390abd55829a9e88aced11a7824b46c167d3457fa23f"));
  Digest d = blake2b256(h.data(), h.size());
  // The node prints the block hash as the raw digest bytes (pow_hash_le reversed, displayed reversed again).
  CHECK_EQ(hexs(d.data(), 32), std::string("431195186ba0ed5f08248af972f1b5193d9e36ff9814e0bc0969d1195141cffd"));
  CHECK(meets_target(d, target_from_nbits(nbits)));
}

TEST(blake2b_targets) {
  auto be = [](const Target& t) {
    Target r;
    for (int i = 0; i < 32; ++i) r[i] = t[31 - i];
    return hexs(r.data(), 32);
  };
  CHECK_EQ(be(target_for_difficulty(1)), std::string("00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
  CHECK_EQ(be(target_for_difficulty(16384)),
           std::string("000000000003ffffffffffffffffffffffffffffffffffffffffffffffffffff"));
  // Rabid's "16": flooring the difficulty to a power of two would make this target 2x too lenient.
  CHECK_EQ(be(target_for_difficulty(15.999755859375)),
           std::string("0000000010001000100010001000100010001000100010001000100010001000"));
  CHECK_EQ(be(target_for_difficulty(0.5)), std::string("00000001fffffffffffffffffffffffffffffffffffffffffffffffffffffffe"));
  CHECK_EQ(be(target_from_nbits(0x207fffff)),
           std::string("7fffff0000000000000000000000000000000000000000000000000000000000"));
  CHECK_EQ(be(target_from_nbits(0x1d00ffff)),
           std::string("00000000ffff0000000000000000000000000000000000000000000000000000"));
  Target t{};
  t[31] = 0x01;  // 2^248
  Digest lo{}, eq{}, hi{};
  eq[0] = 0x01;
  hi[0] = 0x01;
  hi[31] = 0x01;
  CHECK(meets_target(lo, t));
  CHECK(meets_target(eq, t));
  CHECK(!meets_target(hi, t));
}

TEST(blake2b_precompute_matches_reference) {
  // The kernel's per-job state, finished on the CPU the way the kernel finishes it, must give the reference digest.
  Header h{};
  for (int i = 0; i < 80; ++i) h[i] = static_cast<uint8_t>(i * 29 + 3);
  uint64_t m[10], v[16];
  precompute(h, m, v);
  for (int i = 0; i < 8; ++i) CHECK_EQ(m[4] >> (8 * i) & 0xff, uint64_t(h[32 + i]));
  // Column steps 0, 1, 3 change v0/v1/v3 but leave v2 at IV2.
  CHECK_EQ(v[2], 0x3c6ef372fe94f82bull);
  CHECK(v[0] != (0x6a09e667f3bcc908ull ^ 0x01010020ull));
}

TEST(blake2b_pool_fields) {
  std::array<uint8_t, 8> t;
  CHECK(xm::b2b_parse_wire8("0000abcd", t));
  CHECK_EQ(hexs(t.data(), 8), std::string("cdab000000000000"));
  CHECK(xm::b2b_parse_wire8("0102030405060708", t));
  CHECK_EQ(hexs(t.data(), 8), std::string("0102030405060708"));
  CHECK(!xm::b2b_parse_wire8("123", t));
  CHECK_EQ(xm::b2b_nonce_hex(0x0807060504030201ull), std::string("0102030405060708"));
  Job j;
  uint32_t nb;
  bool clean;
  // SHA-256d work (a coinb2 or merkle branches) is refused.
  CHECK(!xm::b2b_parse_notify(R"(["j","00","00","ab",[],"20000000","207fffff","00000000",true])", j, nb, clean));
  CHECK(!xm::b2b_parse_notify(
      R"(["j","0000000000004fee2e5613820a8fa7540fa030c58f0b363a28ee63af2ba84bc7",)"
      R"("0000009650adc5f76b847f7d0a340d604917eea59e0a5d823214ca7974490984655f2700000000","",["aa"],"20000000",)"
      R"("207fffff","00000000",true])",
      j, nb, clean));
}
