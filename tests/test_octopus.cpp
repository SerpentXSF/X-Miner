// Conflux Octopus: sizes from conflux-rust shared.rs (get_cache_size / get_data_size, recomputed independently),
// the evaluation points and the share rule. conflux-rust publishes no hash vectors; the full hash is checked by the
// GPU selftest (GPU against this reference) and by pool acceptance.
#include <cstring>

#include "algo/octopus/octopus.hpp"
#include "pool/cfx_pool.hpp"
#include "test.hpp"

using namespace xm;

TEST(octopus_sizes) {
  CHECK_EQ(octopus::stage_for_height(160197460), 305);
  CHECK_EQ(octopus::cache_items(0), 262139);
  CHECK_EQ(octopus::dataset_pages(0), 16777213);
  CHECK_EQ(octopus::cache_items(1), 263167);
  CHECK_EQ(octopus::dataset_pages(1), 16842751);
  CHECK_EQ(octopus::cache_items(305), 574439);
  CHECK_EQ(octopus::dataset_pages(305), 36765683);
}

TEST(octopus_points) {
  octopus::Hash256 header;
  uint8_t* h = reinterpret_cast<uint8_t*>(header.data());
  for (int i = 0; i < 32; ++i) h[i] = static_cast<uint8_t>(i * 29 + 3);
  octopus::Points pts;
  octopus::make_points(header, pts);
  const uint64_t p = octopus::kMod;
  // x_k = a w^2k + b w^k + c: the second difference over k of a geometric-quadratic sequence is not checkable
  // without a, b, c, but every point and its Shoup constant must be consistent and in range.
  for (int k = 0; k < octopus::kN; ++k) {
    CHECK(pts.x[k] < p);
    CHECK_EQ(pts.xs[k], static_cast<uint32_t>((uint64_t(pts.x[k]) << 32) / p));
  }
  uint64_t v0;
  std::memcpy(&v0, h, 8);
  CHECK_EQ(pts.v[0], v0);
}

TEST(octopus_share_rule) {
  octopus::Hash256 hash{}, bound{};
  reinterpret_cast<uint8_t*>(bound.data())[3] = 0x66;
  reinterpret_cast<uint8_t*>(hash.data())[3] = 0x65;
  CHECK(octopus::below(hash, bound));
  CHECK(!octopus::below(bound, bound));
  // Kryptex's boundary 0x6666...66 (56 hex digits): 2^256 / (0.4 * 2^224) = 2.5 * 2^32 hashes per share.
  const auto b = kawpow::hash_from_hex("0000000066666666666666666666666666666666666666666666666666666666");
  const double d = cfx_difficulty(b);
  CHECK(d > 2.5 * 4294967296.0 * 0.999999 && d < 2.5 * 4294967296.0 * 1.000001);
}
