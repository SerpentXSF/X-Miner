// Parano1d reference against 64 header -> digest vectors generated from the official crates
// (gacanga/NoidMiner tests/vectors.txt, Apache-2.0) and the intermediate values in the research notes.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "algo/noid/noid.hpp"
#include "test.hpp"
#include "pool/noid_pool.hpp"

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

#include "algo/noid/noid_gpu_consts.hpp"

TEST(noid_gpu_permutation_matches_reference) {
  // The ported GPU permutation (kernels/noid, compiled as C++) against the reference, on several states.
  KOp cm[6];
  constant_kops(cm);
  CMulKop mul{cm};
  RcFlat rc;
  for (uint64_t seed : {0ull, 1ull, 0x0123456789abcdefull}) {
    State ref;
    W4 s[4];
    for (int i = 0; i < 4; ++i) {
      ref[i] = u(seed * 0x9e3779b97f4a7c15ull + i, ~seed ^ (i * 0x1111ull));
      s[i] = to_w4(ref[i]);
    }
    permute_flat(ref);
    permute(s, mul, rc);
    for (int i = 0; i < 4; ++i) CHECK(from_w4(s[i]) == ref[i]);
  }
}

TEST(noid_table_multiplier_matches_karatsuba) {
  std::vector<uint32_t> tbl(NOID_TBL_WORDS);
  constant_tables(tbl.data());
  KOp ks[6];
  constant_kops(ks);
  CMulTable t{tbl.data()};
  CMulKop k{ks};
  uint64_t x = 0x243f6a8885a308d3ull;
  for (int i = 0; i < 200; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    W4 a = to_w4(u(x ^ (x >> 7), x * 0x9e3779b97f4a7c15ull));
    for (int c = 0; c < 6; ++c) CHECK(from_w4(t.mul(c, a)) == from_w4(k.mul(c, a)));
  }
}

TEST(noid_gmul32_matches_reference) {
  uint64_t x = 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < 500; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    U128 a = u(x ^ (x >> 11), x * 0xbf58476d1ce4e5b9ull);
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    U128 b = u(x * 0x94d049bb133111ebull, x ^ (x >> 29));
    if (i == 0) a = u(~0ull, ~0ull), b = u(~0ull, ~0ull);  // worst case for the per-class carry bound
    CHECK(from_w4(gmul32(to_w4(a), to_w4(b))) == gf_mul(a, b));
  }
}

// Stratum parsing against a job captured from rplant (na.rplant.xyz:17185, height 241943).
TEST(noid_stratum_job_layout) {
  const std::string fields_hex =
      "23139bb2c638a5f828b30b52399e9f1b0d27027ccf9c6c58855e232dd2cbfe5f28ac3ac46208e0be060ef1044577aa7196fb18a9c4fd829f"
      "4b7f9e09d222e5d28e8354c1f3e8bcc06b881add38926e5662fed38b9ab60df530c4baf761b72600188cd96a000000000000000000000000"
      "17b103000000000000000000000000001c98c496a985fb4e3badf043c470f644cc35fcc72f77f0d78c82bfe3056377090000000000000000"
      "000000000000000034cf3ebff9aecf073b4f4e9a1f414e0175e48d8aa5662edb9d3c08000000000018000000000000000000000000000000"
      "5674010000000000000000000000000050950900000000000000000000000000";
  Fields f{};
  CHECK(xm::noid_parse_fields(fields_hex, f));
  CHECK(f[kNonceField] == U128{});  // the pool zeroes the nonce field
  CHECK_EQ(f[6].lo, 0x6ad98c18ull);  // timestamp field: small little-endian integer
  CHECK_EQ(f[7].lo, 0x3b117ull);     // height 241943
  CHECK(!xm::noid_parse_fields(fields_hex.substr(2), f));
  uint64_t prefix = 0;
  CHECK(xm::noid_parse_prefix("1a33000000002f16", prefix));
  CHECK_EQ(prefix, 0x162f00000000331aull);
  // Submit form: counter bytes (LE) then the raw prefix bytes.
  CHECK_EQ(xm::noid_nonce_hex(U128{0x0102, prefix}), std::string("02010000000000001a33000000002f16"));
}
