#include "algo/octopus/octopus.hpp"

#include <cstring>

namespace xm::octopus {

namespace {

constexpr uint64_t kP = kMod;

uint64_t pow_mod(uint64_t a, uint64_t n) {
  uint64_t r = 1;
  for (a %= kP; n; n >>= 1, a = a * a % kP)
    if (n & 1) r = r * a % kP;
  return r;
}

uint64_t gcd(uint64_t a, uint64_t b) {
  while (b) {
    const uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// A header word to a field element 11^e, e coprime to p - 1 (so a generator when 11 is one).
uint64_t remap(uint64_t h) {
  uint64_t e = h % (kP - 2) + 1;
  for (uint64_t g; (g = gcd(e, kP - 1)) != 1;) e /= g;
  return pow_mod(kModB, e);
}

uint64_t rotl(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }

struct Sip {
  uint64_t v0, v1, v2, v3;
  void round() {
    v0 += v1;
    v2 += v3;
    v1 = rotl(v1, 13);
    v3 = rotl(v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rotl(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rotl(v1, 17);
    v3 = rotl(v3, 21);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rotl(v2, 32);
  }
  void hash24(uint64_t nonce) {
    v3 ^= nonce;
    round();
    round();
    v0 ^= nonce;
    v2 ^= 0xff;
    for (int i = 0; i < 4; ++i) round();
  }
};

}  // namespace

int cache_items(int stage) { return kawpow::largest_prime((1 << 18) + 1024 * stage); }
int dataset_pages(int stage) { return kawpow::largest_prime((1 << 24) + 65536 * stage); }

std::shared_ptr<const Light> make_light(int stage) {
  return kawpow::make_light_sized(stage, kDatasetParents, cache_items(stage), 2 * dataset_pages(stage), stage);
}

void make_points(const Hash256& header, Points& out) {
  std::memcpy(out.v, header.data(), 32);
  const uint64_t a = remap(out.v[0]), b = remap(out.v[1]), w = remap(out.v[3]);
  uint64_t c;
  for (uint64_t h = out.v[2];; ++h) {  // c with b^2 != 4ac
    c = remap(h);
    if (b * b % kP != 4 * a * c % kP) break;
  }
  uint64_t wk = 1;
  for (int k = 0; k < kN; ++k) {
    const uint64_t x = (a * (wk * wk % kP) + b * wk + c) % kP;
    out.x[k] = static_cast<uint32_t>(x);
    out.xs[k] = static_cast<uint32_t>((x << 32) / kP);
    wk = wk * w % kP;
  }
}

Hash256 hash(const Light& light, const Points& pts, const Hash256& header, uint64_t nonce) {
  // The polynomial: coefficient j * 32 + i is SipHash round j + 1 of nonce (group base + i), mod p.
  uint32_t d[kN];
  const uint64_t base = nonce / kGroup * kGroup;
  for (int i = 0; i < kGroup; ++i) {
    Sip s{pts.v[0], pts.v[1], pts.v[2], pts.v[3]};
    s.hash24(base + i);
    for (int j = 0; j < kPoints; ++j) {
      s.round();
      d[j * kGroup + i] = static_cast<uint32_t>((s.v0 ^ s.v1 ^ s.v2 ^ s.v3) & 0xffffffffu) % kMod;
    }
  }
  uint32_t res[kPoints];
  uint64_t result = 0;
  for (int t = 0; t < kPoints; ++t) {
    const uint64_t x = pts.x[nonce % kGroup + kGroup * t];
    uint64_t pv = 0;
    for (int j = kN - 1; j >= 0; --j) pv = (pv * x + d[j]) % kP;
    res[t] = static_cast<uint32_t>(pv);
    result = result * kawpow::kFnvPrime ^ pv;
  }

  uint8_t in[40];
  std::memcpy(in, header.data(), 32);
  std::memcpy(in + 32, &result, 8);  // little-endian
  const auto seed = kawpow::keccak512(in, 40);
  uint32_t s[16], mix[64];
  std::memcpy(s, seed.data(), 64);
  for (int i = 0; i < 64; ++i) mix[i] = s[i % 16];
  const uint32_t pages = uint32_t(light.dataset_items / 2);
  uint32_t item[16];
  for (uint32_t i = 0; i < kAccesses; ++i) {
    const uint32_t page = kawpow::fnv1(s[0] ^ i ^ res[i], mix[i]) % pages;
    for (int n = 0; n < 4; ++n) {
      kawpow::dataset_item_512(light, uint64_t(page) * 4 + n, item);
      for (int k = 0; k < 16; ++k) mix[16 * n + k] = kawpow::fnv1(mix[16 * n + k], item[k]);
    }
  }
  auto reduce = [&](int w) { return kawpow::fnv1(kawpow::fnv1(kawpow::fnv1(mix[w], mix[w + 1]), mix[w + 2]), mix[w + 3]); };
  uint32_t compress[8];
  for (int i = 0; i < 8; ++i) compress[i] = kawpow::fnv1(reduce(4 * i), reduce(32 + 4 * i));
  uint8_t fin[96];
  std::memcpy(fin, seed.data(), 64);
  std::memcpy(fin + 64, compress, 32);
  const auto h = kawpow::keccak256(fin, 96);
  Hash256 out;
  std::memcpy(out.data(), h.data(), 32);
  return out;
}

bool below(const Hash256& hash, const Hash256& boundary) {
  return std::memcmp(hash.data(), boundary.data(), 32) < 0;  // byte arrays in big-endian order
}

}  // namespace xm::octopus
