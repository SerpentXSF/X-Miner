// Tari Cuckaroo29 against the Tari node's own test vectors (tari-project/tari cuckaroo_pow.rs and siphash.rs tests):
// siphash, the siphash block, key derivation, edge endpoints, the packing, and a whole 42-cycle proof.
#include <string>
#include <vector>

#include "algo/c29/c29.hpp"
#include "test.hpp"

using namespace xm;

namespace {

std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> v;
  for (size_t i = 0; i + 1 < s.size(); i += 2) v.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
  return v;
}

std::string hex(const uint8_t* p, size_t n) {
  static const char* hx = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += hx[p[i] >> 4];
    s += hx[p[i] & 15];
  }
  return s;
}

c29::Keys keys_le(const std::string& h) {
  const auto b = unhex(h);
  c29::Keys k{};
  for (int w = 0; w < 4; ++w)
    for (int i = 0; i < 8; ++i) k[w] |= uint64_t(b[8 * w + i]) << (8 * i);
  return k;
}

c29::Hash hash32(const std::string& h) {
  const auto b = unhex(h);
  c29::Hash r{};
  for (int i = 0; i < 32; ++i) r[i] = b[i];
  return r;
}

uint64_t be64(const std::string& h) {
  uint64_t v = 0;
  for (uint8_t b : unhex(h)) v = v << 8 | b;
  return v;
}

const c29::Proof kProof4 = {1252665,   4516819,   17265865,  24709305,  89445155,  99203810,  108059490,
                            118448252, 126732226, 147341567, 162827037, 177849034, 183556179, 191103550,
                            231377681, 233321258, 237510921, 257843213, 266606591, 282340500, 288301862,
                            333019502, 344766902, 355446190, 368615551, 371362415, 371729271, 380994474,
                            396711753, 400948255, 402047643, 445581804, 452121724, 460915916, 464509725,
                            472201798, 487709959, 488169697, 499236155, 509929107, 516834413, 534561363};

}  // namespace

TEST(c29_siphash) {
  CHECK_EQ(c29::siphash24({1, 2, 3, 4}, 10), 928382149599306901ull);
  CHECK_EQ(c29::siphash24({1, 2, 3, 4}, 111), 10524991083049122233ull);
  CHECK_EQ(c29::siphash24({9, 7, 6, 7}, 12), 1305683875471634734ull);
  CHECK_EQ(c29::siphash24({9, 7, 6, 7}, 10), 11589833042187638814ull);
  CHECK_EQ(c29::siphash_block({1, 2, 3, 4}, 10), 1182162244994096396ull);
  CHECK_EQ(c29::siphash_block({1, 2, 3, 4}, 123), 11303676240481718781ull);
  CHECK_EQ(c29::siphash_block({9, 7, 6, 7}, 12), 4886136884237259030ull);
}

TEST(c29_keys_and_edges) {
  const auto k1 = c29::sip_keys(hash32("4dbfee3eb7b9a6a27d2a4a8d754eb77cc5493006945c1246e50e9beb4de5ffa5"), be64("9d589ed597ed42d1"));
  uint8_t kb[32];
  for (int w = 0; w < 4; ++w)
    for (int i = 0; i < 8; ++i) kb[8 * w + i] = uint8_t(k1[w] >> (8 * i));
  CHECK_EQ(hex(kb, 32), std::string("49b48f77df94943cf3a422c5a0b528c737cc38a7b6c36076e81abcede5b2be3a"));
  uint32_t u, v;
  c29::edge_nodes(keys_le("a216826b5d2752ccf129eef73e1e02a6b56d13195d7998e6b04d02a136b88f4f"), 24394923, u, v);
  CHECK_EQ(u, 259523165u);
  CHECK_EQ(v, 501211281u);
  const auto k3 = c29::sip_keys(hash32("93e43a39b44f0875af830b6fd4cc69a421b1f5f0c5efd1b6c2e16d439bfd238c"), be64("ab30dc99fd054f57"));
  c29::edge_nodes(k3, 147341567, u, v);
  CHECK_EQ(u, 361363154u);
  CHECK_EQ(v, 255932004u);
}

TEST(c29_proof) {
  const auto k = c29::sip_keys(hash32("93e43a39b44f0875af830b6fd4cc69a421b1f5f0c5efd1b6c2e16d439bfd238c"), be64("ab30dc99fd054f57"));
  CHECK(c29::verify(k, kProof4));
  c29::Proof bad = kProof4;
  bad[5] += 1;
  CHECK(!c29::verify(k, bad));
  // a valid cycle of one graph is not one of another
  const auto k1 = c29::sip_keys(hash32("93e43a39b44f0875af830b6fd4cc69a421b1f5f0c5efd1b6c2e16d439bfd238c"), be64("ab30dc99fd054f58"));
  CHECK(!c29::verify(k1, kProof4));
  // the cycle finder recovers the proof from its edges plus noise
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (uint32_t i = 0; i < 300; ++i) {
    uint32_t u, v;
    c29::edge_nodes(k, i * 1000003u % (1u << 29), u, v);
    edges.push_back({u, v});
  }
  for (uint32_t e : kProof4) {
    uint32_t u, v;
    c29::edge_nodes(k, e, u, v);
    edges.push_back({u, v});
  }
  const auto cycles = c29::find_cycles(edges);
  CHECK_EQ(cycles.size(), size_t(1));
  if (cycles.size() == 1) CHECK_EQ(cycles[0][0], 300u);
}

TEST(c29_packing) {
  const c29::Proof p = {24394923,  46592033,  58968194,  71842682,  75995694,  81022926,  97860763,  105410600,
                        106063142, 138729012, 150559164, 170291280, 197045966, 202539638, 206356582, 215689620,
                        218504800, 232385274, 243238820, 250666150, 268537652, 296239928, 296891268, 315542595,
                        338677422, 341088271, 346272558, 359697897, 364349211, 377758979, 391857369, 403811427,
                        408334826, 413242437, 413564914, 426980322, 433277222, 460056825, 473150750, 473935291,
                        477597924, 497788893};
  const auto packed = c29::pack(p);
  CHECK_EQ(hex(packed.data(), packed.size()),
           std::string("ab3c742104de5808220f0ebd1d24e2a279489c9fa8c9264f754181433226655286c69a08f166e523283813e5ecea"
                       "bbec042598193013a34cd966601e064d5f3dbb911efe39536e7847f3180071865023e18c6c1c627696aece2ff4019"
                       "38abacc8ed2f446b8ba71785b074a086d36d1d61d638dc0eab156b8883214cbff9962f199b96c92349df3d1d7b647"
                       "ed0cdf6dfde1e49077bcfb74b503"));
  const auto r = c29::result(p);  // blake2b-256 of the 153 packed bytes (two blocks), from hashlib
  CHECK_EQ(hex(r.data(), 32), std::string("06d52b90ccfd4db1a52cc133a46dac0dc4577343c1a27d200865a81d717a60e1"));
  CHECK_EQ(c29::target_difficulty("cccccccccccccc0c"), 20ull);
  CHECK_EQ(c29::target_difficulty("ffffffff00000000"), 4294967297ull);
}
