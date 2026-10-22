// Parano1d reference against 64 header -> digest vectors generated from the official crates
// (gacanga/NoidMiner tests/vectors.txt, Apache-2.0) and the intermediate values in the research notes.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "algo/noid/noid.hpp"
#include "test.hpp"

using namespace xm::noid;

namespace {

std::string hexd(const Digest& d) {
  static const char* x = "0123456789abcdef";
  std::string s;
  for (uint8_t b : d) {
    s += x[b >> 4];
    s += x[b & 15];
  }
  return s;
}

U128 u(uint64_t hi, uint64_t lo) { return {lo, hi}; }

}  // namespace

TEST(noid_permute_zero_state) {
  State s{};
  permute_flat(s);
  CHECK(s[0] == u(0xaea8cadc2223c257ull, 0x61102e0e8fa2233eull));
  CHECK(s[1] == u(0x45aa2280a4cb84b7ull, 0x2f0dc1ad893c8853ull));
  CHECK(s[2] == u(0x3cb2df4a844bfda6ull, 0x6acb7f4eefd29072ull));
  CHECK(s[3] == u(0x1c65cb85e905b20cull, 0x0e224c196528ada2ull));
}

TEST(noid_basis_maps) {
  CHECK(tower_to_flat(u(0, 2)) == u(0x3d5bd35c94646a24ull, 0x7573da4a5f7710edull));
  CHECK(flat_to_tower(tower_to_flat(u(0x0123456789abcdefull, 0xfedcba9876543210ull))) ==
        u(0x0123456789abcdefull, 0xfedcba9876543210ull));
  CHECK(tower_to_flat(u(0x544b0ec76d00616dull, 0xbcbb9b7e9a319aefull)) ==
        u(0xfe718f94d094941aull, 0x7ff7fbb45c617d95ull));
}

TEST(noid_digest_vectors) {
  std::ifstream f(std::string(XM_TEST_VECTORS) + "/noid_vectors.txt");
  std::string line;
  int n = 0, ok = 0;
  while (std::getline(f, line)) {
    if (line.size() < 10) continue;
    std::istringstream in(line);
    std::string fields_s, want;
    in >> fields_s >> want;
    Fields fs{};
    std::stringstream ss(fields_s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, ',') && i < 16) fs[i++] = parse_u128_be_hex(item);
    CHECK_EQ(i, 16);
    std::string got = hexd(digest(fs));
    if (got == want) ++ok;
    else if (n < 3) CHECK_EQ(got, want);
    ++n;
  }
  CHECK_EQ(n, 64);
  CHECK_EQ(ok, 64);
}

TEST(noid_midstate_path_equals_full) {
  Fields fs{};
  for (int i = 0; i < 16; ++i) fs[i] = u(0x1111111111111111ull * i, 0x0123456789abcdefull ^ i);
  State mid = midstate(fs);
  for (uint64_t n : {0ull, 1ull, 0xdeadbeefull}) {
    Fields g = fs;
    g[kNonceField] = u(0xabcdef0012345678ull, n);
    CHECK(digest(g) == digest_from_midstate(mid, fs, g[kNonceField]));
  }
}
