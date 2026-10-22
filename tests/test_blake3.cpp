// BLAKE3 against the official test vectors (tests/vectors/blake3_test_vectors.json, CC0).
#include <fstream>
#include <string>

#include "crypto/blake3.hpp"
#include "nlohmann/json.hpp"
#include "test.hpp"

using namespace xm;

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

nlohmann::json vectors() {
  std::ifstream f(XM_TEST_VECTORS "/blake3_test_vectors.json");
  return nlohmann::json::parse(f);
}

}  // namespace

TEST(blake3_official_vectors) {
  auto v = vectors();
  std::string key = v["key"];
  std::string ctx = v["context_string"];
  int cases = 0;
  for (const auto& c : v["cases"]) {
    size_t n = c["input_len"];
    std::vector<uint8_t> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = static_cast<uint8_t>(i % 251);
    std::string want_hash = c["hash"], want_keyed = c["keyed_hash"], want_derive = c["derive_key"];
    auto h = blake3::hash_xof(in.data(), n, want_hash.size() / 2);
    CHECK_EQ(hex(h.data(), h.size()), want_hash);
    auto k = blake3::hash_xof(in.data(), n, want_keyed.size() / 2, reinterpret_cast<const uint8_t*>(key.data()));
    CHECK_EQ(hex(k.data(), k.size()), want_keyed);
    auto d = blake3::derive_key(ctx.c_str(), in.data(), n);
    CHECK_EQ(hex(d.data(), 32), want_derive.substr(0, 64));
    auto h32 = blake3::hash(in.data(), n);
    CHECK_EQ(hex(h32.data(), 32), want_hash.substr(0, 64));
    ++cases;
  }
  CHECK_EQ(cases, 35);
}
