// Ergo Autolykos v2 reference. Vector: ergoplatform/ergo AutolykosPowSchemeSpec "test vectors for first increase in N
// value (height 614,400)" (@c1a69f6); N values from the same spec.
#include <string>

#include "algo/autolykos/autolykos.hpp"
#include "test.hpp"

using namespace xm::autolykos;

namespace {
std::string hex(const Bytes32& b) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t c : b) {
    s += d[c >> 4];
    s += d[c & 15];
  }
  return s;
}
Bytes32 unhex(const std::string& h) {
  Bytes32 b{};
  for (int i = 0; i < 32; ++i) b[i] = static_cast<uint8_t>(std::stoul(h.substr(2 * i, 2), nullptr, 16));
  return b;
}
}  // namespace

TEST(autolykos_table_size) {
  CHECK_EQ(table_size(500000), 67108864u);
  CHECK_EQ(table_size(614400), 70464240u);
  CHECK_EQ(table_size(700000), 73987410u);
  CHECK_EQ(table_size(1904505), 238614390u);
  CHECK_EQ(table_size(4198400), 2143944600u);
}

TEST(autolykos_blake2b) {
  CHECK_EQ(hex(blake2b256(nullptr, 0)), std::string("0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"));
}

TEST(autolykos_hit_vector) {
  const Bytes32 msg = unhex("548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f");
  CHECK_EQ(hex(hit(msg, 0x3105, 614400)), std::string("0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a"));
}

TEST(autolykos_decimal_target) {
  // Kryptex b (2026-10-28): ~2^224.0, i.e. about 2^32 hashes per share.
  const Bytes32 b = from_decimal("26959946667150639794667015087019630673637144422540572481103610249215");
  CHECK_EQ(hex(b), std::string("00000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));
  CHECK(work_for(b) > 4.29e9 && work_for(b) < 4.30e9);
}
