// NexaPow against the Echelon spec vectors (spec.nexa.org/mining/echelon-protocol: commitment and hashes printed
// byte-reversed, nonce raw) and real main-chain blocks (headers from Rostrum blockchain.block.header; commitment and
// nonce raw, powHash displayed byte-reversed), with the signatures in between.
#include <string>
#include <vector>

#include "algo/nexa/nexa.hpp"
#include "crypto/secp256k1.hpp"
#include "crypto/sha256.hpp"
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

std::string display(const nexa::Hash& h) {
  nexa::Hash r;
  for (int i = 0; i < 32; ++i) r[i] = h[31 - i];
  return hex(r.data(), 32);
}

nexa::Hash hash_of(const std::string& s, bool reversed) {
  const auto v = unhex(s);
  nexa::Hash h{};
  for (int i = 0; i < 32; ++i) h[i] = v[reversed ? 31 - i : i];
  return h;
}

struct Block {
  const char *commitment, *nonce, *mining_hash, *sig, *pow_display;
};

const Block kBlocks[] = {
    {"a196578a9aa2f79ef4881a141ca31c18a6e76a6c58c2da4210f8a2b052d98c5d", "1ca286d4ae78d2fece9a3763",
     "172d122abe7d47242adcc06533c1da57a103a43fb2361b25befc080146e78e5d",
     "a1c2a8119bfe98f221826490166f3e17125db3589a3a58460b6c3caa5474a0acfd948f29c09722781a13ad56814f598bb1b2dfd3dc70e23d90b2ddfa381da6a0",
     "00000000000003d9e86fa2a9f4f95dede53285663e7ac9302ca1a0670cc8d6e7"},  // height 300000
    {"f9c430f23b52ada3405ac1e3b39b85084483dfed6b0e3a6d2e507cd5982e829d", "78147b3500002c00d2f5120000000000",
     "e66bf55067bd44a712e07ee7fb4ff0f9c50a9fea31c20c49642e5a8a280f130f",
     "cd0cbdb6fea59c3e7b56bf90b00aba209c64993f1bfb81eb080fef26b57bedf6d1fb785766e44faf6b1fe281c2a9fa5516e7ce8ace672ca02ef084b44e29064e",
     "00000000000091e788b07da69d693efe2a9d3becfed8f835fc9263b3169b3d48"},  // height 1000001
    {"47a397e0f1724a2c9603d77874b8302f602187aa43f4e60b77719b85800ddae7", "2d330ae100005301933374b948f8cc05",
     "5ceb789ac80800f1c8fa7a766d5683cd3e5c8b0f7e7eeac3bf70f83190752956",
     "b79335aea8038cd79265acf4fd430c2eacac01d292444aee7928bbec439f743cde847ccd172d76156312efad8331396ba6291a9cad438a6ab7f44697786467c6",
     "000000000000295b1e2579487e6dea548cc36817a423b9bc4d74868c126e99ab"},  // height 1180001
};

}  // namespace

TEST(secp256k1_generator_multiples) {
  using namespace crypto::secp256k1;
  // 2G and (n - 1)G = -G
  uint8_t two[32] = {};
  two[31] = 2;
  const Point p2 = mul_g(from_be(two));
  uint8_t x[32];
  to_be(p2.x, x);
  CHECK_EQ(hex(x, 32), std::string("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"));
  uint8_t nm1[32];
  to_be(kN, nm1);
  nm1[31] -= 1;
  const Point pm = mul_g(from_be(nm1));
  CHECK(pm.x == generator().x);
  CHECK(fe_add(pm.y, generator().y) == U256{});
}

TEST(nexa_echelon_vectors) {
  const nexa::Hash hc = hash_of("0a4ac49b2d02e3c8d12c7093255ba7c49624f9c374d9f1c2f8e37c58705e74b0", true);
  const char* cases[][4] = {
      {"10000000000000001182dc5800000000", "7077c3549e088001ff681c2291bbc171f6ea681b5bc79bca9f2e0f563ad3dfef",
       "93bb5bd98bb9d0817c5594782d4648a995e76755a3c94fdd96673115a06b8483efe485e73c1e4732ce9c34ef6891e7f25ff7c080aacbf71660bf782b067bc4f1",
       "00000042cbc240375242e14641488a0e2dca7b54458a2cea23dc1d2c178bb188"},
      {"1000000000000000b787915d00000000", "5cddd556e7972ee8b3e4ce01f954336cbb223813c1f01c768f299b82f20a0ea9",
       "fc1894a100e6b33d19abac5289403b4c942d75d061b5453904be740681526cae30ed3a59f1a9d3ca165151ddb1ca3069de832cd5cc5fc95a7339f058f1fd155b",
       "0000005f0b59e110863566e77d85e1b4fc713754e5c75f8fc3df44133866a669"}};
  for (auto& c : cases) {
    const auto nonce = unhex(c[0]);
    const nexa::Hash mh = nexa::mining_hash(hc, nonce.data(), nonce.size());
    CHECK_EQ(display(mh), std::string(c[1]));
    nexa::Hash pw{};
    CHECK(nexa::pow_hash(hc, nonce.data(), nonce.size(), pw));
    CHECK_EQ(display(pw), std::string(c[3]));
  }
  // the signature of vector 1
  const auto nonce = unhex(cases[0][0]);
  const nexa::Hash mh = nexa::mining_hash(hc, nonce.data(), nonce.size());
  const auto h1 = crypto::sha256(mh.data(), 32);
  uint8_t sig[64];
  CHECK(crypto::secp256k1::schnorr_sign_bch(h1.data(), mh.data(), sig));
  CHECK_EQ(hex(sig, 64), std::string(cases[0][2]));
}

TEST(nexa_mainnet_blocks) {
  for (const Block& b : kBlocks) {
    const nexa::Hash hc = hash_of(b.commitment, false);
    const auto nonce = unhex(b.nonce);
    const nexa::Hash mh = nexa::mining_hash(hc, nonce.data(), nonce.size());
    CHECK_EQ(hex(mh.data(), 32), std::string(b.mining_hash));
    const auto h1 = crypto::sha256(mh.data(), 32);
    uint8_t sig[64];
    CHECK(crypto::secp256k1::schnorr_sign_bch(h1.data(), mh.data(), sig));
    CHECK_EQ(hex(sig, 64), std::string(b.sig));
    nexa::Hash pw{};
    CHECK(nexa::pow_hash(hc, nonce.data(), nonce.size(), pw));
    CHECK_EQ(display(pw), std::string(b.pow_display));
  }
}

TEST(nexa_targets) {
  // difficulty 1 = 0xFFFF * 2^208; nBits 0x1b00f8ac
  const nexa::Target t1 = nexa::target_from_difficulty(1);
  CHECK_EQ(int(t1[27]), 0xff);
  CHECK_EQ(int(t1[26]), 0xff);
  CHECK_EQ(int(t1[28]), 0);
  CHECK_EQ(nexa::target_top64(t1), 0x00000000FFFF0000ull);
  const nexa::Target tb = nexa::target_from_nbits(0x1b00f8ac);
  CHECK_EQ(int(tb[26]), 0x00);
  CHECK_EQ(int(tb[25]), 0xf8);
  CHECK_EQ(int(tb[24]), 0xac);
  const nexa::Hash pw = hash_of(kBlocks[2].pow_display, true);
  CHECK(nexa::meets(pw, tb));
  CHECK(!nexa::meets(pw, nexa::target_from_nbits(0x1a00f8ac)));
}
