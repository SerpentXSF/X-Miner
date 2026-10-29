// QHash (Qubitcoin) against real main-chain headers (explorer-api.superquantum.io: block hash = QHash of the header,
// displayed byte-reversed), covering both angle eras and the zero-byte rules, plus a full 2^16 state-vector
// simulation of the reference circuit (custatevec conventions: exp(i theta P) rotations) for the contraction.
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "algo/qhash/qhash.hpp"
#include "crypto/sha256.hpp"
#include "pool/qhash_pool.hpp"
#include "test.hpp"

using namespace xm;

namespace {

std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> v;
  for (size_t i = 0; i + 1 < s.size(); i += 2) v.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
  return v;
}

std::string display(const qhash::Hash& h) {
  static const char* hx = "0123456789abcdef";
  std::string s;
  for (int i = 31; i >= 0; --i) {
    s += hx[h[i] >> 4];
    s += hx[h[i] & 15];
  }
  return s;
}

// The reference circuit on a full state vector (qubit q = bit q of the index).
std::array<double, 16> statevector(const uint8_t nib[64], bool odd) {
  using cd = std::complex<double>;
  std::vector<cd> v(size_t(1) << 16);
  v[0] = 1;
  const double pi = 3.14159265358979323846;
  auto rot = [&](int q, double th, bool y) {
    const double c = std::cos(th), s = std::sin(th);
    const size_t m = size_t(1) << q;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i & m) continue;
      const cd a0 = v[i], a1 = v[i | m];
      if (y) {
        v[i] = c * a0 + s * a1;
        v[i | m] = -s * a0 + c * a1;
      } else {
        v[i] = a0 * cd(c, s);
        v[i | m] = a1 * cd(c, -s);
      }
    }
  };
  for (int l = 0; l < 2; ++l) {
    for (int i = 0; i < 16; ++i) {
      rot(i, -(2.0 * nib[(2 * l * 16 + i) % 64] + odd) * pi / 32, true);
      rot(i, -(2.0 * nib[((2 * l + 1) * 16 + i) % 64] + odd) * pi / 32, false);
    }
    for (int i = 0; i < 15; ++i) {
      const size_t cm = size_t(1) << i, tm = size_t(1) << (i + 1);
      for (size_t k = 0; k < v.size(); ++k)
        if ((k & cm) && !(k & tm)) std::swap(v[k], v[k | tm]);
    }
  }
  std::array<double, 16> e{};
  for (size_t k = 0; k < v.size(); ++k) {
    const double p = std::norm(v[k]);
    for (int q = 0; q < 16; ++q) e[q] += (k >> q & 1) ? -p : p;
  }
  return e;
}

}  // namespace

TEST(sha256_abc) {
  const uint8_t abc[3] = {'a', 'b', 'c'};
  const auto h = crypto::sha256(abc, 3);
  CHECK(h == (crypto::Sha256Digest{0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                                   0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad}));
  std::vector<uint8_t> m(200, 0x61);  // padding edge cases: 55 (one tail block), 56 (two), 119 (full block + two)
  CHECK(crypto::sha256(m.data(), 55)[0] == 0x9f && crypto::sha256(m.data(), 55)[31] == 0x18);
  CHECK(crypto::sha256(m.data(), 56)[0] == 0xb3 && crypto::sha256(m.data(), 56)[31] == 0x8a);
  CHECK(crypto::sha256(m.data(), 119)[0] == 0x31 && crypto::sha256(m.data(), 119)[31] == 0xfb);
}

TEST(qhash_mainnet_blocks) {
  const struct {
    int height;
    const char *hash, *header;
  } blocks[] = {
      {114200, "00000000000000d4276ddd0b71354aa6919301f6bef46d5166c9986e62935044",
       "0400002054b580c7c802f22204f40cb9606664e697cb0441335b90229e0d000000000000565d61178f21340b4e40958fe9fbbef0e10655c44a433bc08f055f3b2a4644181a40e36a050e0e1a6d380c52"},
      {114100, "0000000000000c67436d80c62b0d5b1198b2b0f7f7fd833a5742d3225c36ec6c",
       "04000020ca2ed03f49ad21075604adb73abd7c30644fe6237ecf6473d506000000000000e36e16826d9fccf132bf214c1fbb3255d889909d2d1f394d11a8f1daee23d281e25ee26a952c0e1a7919ea26"},
      {90000, "000000000000015f508c980e9e9f29275f93ea87f5efe8cd199f268fc625cde0",
       "04000020a961b4e060d51653fc1e183405bcad278b0fe5ddf06598513f05000000000000259be2bba329de617cbe4dac11f3a96d9e3c17b8b79ace5647193d9f56bfb79c5a8e026af829061acae8fb39"},
      {60000, "0000000000000a21630593676b1516dfad401ac9da92db247849f14b1028882d",
       "0400002093893b97b13069764a9677446e113d927bd968b922415e65551400000000000069a865a068d807e57b3d0c50a8fbfabec65ea43ebf81728118ba8a827130fc31a66ff568b36f1a1ac696736f"},
      {30000, "0000001201cfbd8b1bc68ff645d1159dec8d9df4deaaeaf4127b26e055423ae8",  // even angles, no zero rule
       "040000205cec6b9d4278019d7f6180ee8b763b61df5c14e91612485c47a0e68c0c0000004bffcd573eb7d7d5c3a4ce29a81e89991962c7c16d811ef0d8274cf45f4775c4ea982468772e331dd0120000"},
      {5000, "000012396183cd4488300cb08f01ca897ed5f0996904c520eb5b39ac24ab2da9",
       "040000201c71989a8172ccbc83c0dd49d6bfd406dd9d29dc1bed75c11746777b6313000023adb6698d20b5f913f8a878f5313aead41a916320c4d05169d2d44905ecb538da0052671393181e13050000"},
  };
  for (const auto& b : blocks) {
    const auto hdr = unhex(b.header);
    CHECK(hdr.size() == 80);
    CHECK_EQ(display(qhash::hash(hdr.data())), std::string(b.hash));
    // Each block hash meets its own nBits target.
    const uint32_t nbits = uint32_t(hdr[72]) | uint32_t(hdr[73]) << 8 | uint32_t(hdr[74]) << 16 | uint32_t(hdr[75]) << 24;
    CHECK(qhash::meets(qhash::hash(hdr.data()), qhash::target_from_nbits(nbits)));
  }
}

TEST(qhash_contraction_matches_statevector) {
  std::mt19937 rng(11);
  for (int t = 0; t < 12; ++t) {
    uint8_t nib[64];
    for (auto& x : nib) x = static_cast<uint8_t>(rng() & 15);
    const bool odd = t & 1;
    const auto a = statevector(nib, odd);
    const auto b = qhash::expectations(nib, odd);
    for (int q = 0; q < 16; ++q) {
      CHECK(std::fabs(a[q] - b[q]) < 1e-12);
      CHECK(qhash::to_fixed(a[q]) == qhash::to_fixed(b[q]));
    }
  }
}

TEST(qhash_fixed_and_rules) {
  CHECK(qhash::to_fixed(0.5) == 16384);
  CHECK(qhash::to_fixed(-0.5) == -16384);
  CHECK(qhash::to_fixed(1.0) == -32768);   // 32768 wraps in int16, as in the reference
  CHECK(qhash::to_fixed(-1.0) == -32768);
  CHECK(qhash::to_fixed(1.0 / 65536) == 1);  // half away from zero
  CHECK(qhash::to_fixed(-1.0 / 65536) == -1);
  CHECK(qhash::zero_limit(1754220531) == 8);
  CHECK(qhash::zero_limit(1753305380) == 24);
  CHECK(qhash::zero_limit(1753105444) == 32);
  CHECK(qhash::zero_limit(1753105443) == 33);
  CHECK(qhash::odd_angles(1758762000) && !qhash::odd_angles(1758761999));
}

TEST(qhash_targets) {
  // Difficulty 1 = 0x00000000FFFF0000...; 48 -> 0xFFFF * 2^208 / 48.
  const auto t1 = qhash::target_from_difficulty(1);
  CHECK(t1[31] == 0 && t1[30] == 0 && t1[29] == 0 && t1[28] == 0 && t1[27] == 0xff && t1[26] == 0xff && t1[25] == 0);
  CHECK(qhash::target_top64(t1) == 0x00000000ffff0000ull);
  CHECK(qhash::target_top64(qhash::target_from_difficulty(48)) == 0x00000000ffff0000ull / 48);
  CHECK(qhash::target_top64(qhash::target_from_difficulty(1.0 / 1024)) == 0x000003fffc000000ull);
  const auto nb = qhash::target_from_nbits(0x1d00ffff);
  CHECK(qhash::target_top64(nb) == 0x00000000ffff0000ull);
  qhash::Hash h{};
  h[27] = 0xff;
  h[26] = 0xff;
  CHECK(qhash::meets(h, t1));
  h[0] = 1;
  CHECK(!qhash::meets(h, t1));
}

TEST(qhash_notify_and_header) {
  // A LuckyPool notify (2026-10-29) for height 114213; its prevhash is block 114212.
  const std::string p =
      R"(["7acb","5fa4b8f385691b3c3a5bae9ac9d0a1517786724bb9209b1900000d3300000000","01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff200326be01040955e36a08","0d2f6e6f64655374726174756d2f00000000030000000000000000266a24aa21a9ede2f61c3f71d1defd3fa999dfa36953755c690689799962b48bebd836974e8cf980010b270100000016001496a15ea9fb6cc5d0d327e5fa88bb4183170f47ce80f0fa0200000000160014b2d089f1db1205d98baa5b8d9bdeda22813e6fe000000000",[],"20000004","1a0de1e9","6ae35509",true])";
  qhash::Job j;
  bool clean = false;
  CHECK(qhash_parse_notify(p, j, clean));
  CHECK(clean);
  CHECK_EQ(j.id, std::string("7acb"));
  CHECK(j.tpl.version == 0x20000004 && j.tpl.nbits == 0x1a0de1e9 && j.tpl.ntime == 0x6ae35509);
  CHECK(j.tpl.branches.empty());
  // prevhash in header order = block 114212's hash byte-reversed
  const auto want = unhex("0000000000000d33b9209b197786724bc9d0a1513a5bae9a85691b3c5fa4b8f3");
  bool same = true;
  for (int i = 0; i < 32; ++i) same = same && j.tpl.prevhash[i] == want[31 - i];
  CHECK(same);
  j.tpl.extranonce1 = {0x21, 0x65, 0xc7, 0xfe};
  j.tpl.extranonce2_size = 4;
  CHECK_EQ(qhash::extranonce2_hex(j.tpl, 0x01020304), std::string("04030201"));
  uint8_t hdr[80];
  qhash::header(j.tpl, 0, hdr);
  // no branches: the merkle root is the coinbase txid
  std::vector<uint8_t> cb = j.tpl.coinb1;
  cb.insert(cb.end(), {0x21, 0x65, 0xc7, 0xfe, 0, 0, 0, 0});
  cb.insert(cb.end(), j.tpl.coinb2.begin(), j.tpl.coinb2.end());
  const auto txid = crypto::sha256d(cb.data(), cb.size());
  CHECK(std::memcmp(hdr + 36, txid.data(), 32) == 0);
  CHECK(hdr[0] == 0x04 && hdr[3] == 0x20 && hdr[68] == 0x09 && hdr[71] == 0x6a && hdr[72] == 0xe9 && hdr[75] == 0x1a);
  CHECK(!qhash_parse_notify(R"(["x","00",""])", j, clean));
}
