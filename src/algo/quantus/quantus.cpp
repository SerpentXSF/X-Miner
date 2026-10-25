#include "algo/quantus/quantus.hpp"

#include "algo/quantus/quantus_consts.hpp"

namespace xm::qtc {

namespace {

constexpr uint64_t kEps = 0xffffffffull;  // 2^64 mod p

uint64_t add(uint64_t a, uint64_t b) {  // canonical in, canonical out
  uint64_t s = a + b;
  if (s < a) s += kEps;  // wrapped past 2^64
  return s >= kP ? s - kP : s;
}

// 128-bit product reduced with 2^64 = eps and 2^96 = -1 (mod p).
uint64_t mul(uint64_t a, uint64_t b) {
  const uint64_t a0 = a & kEps, a1 = a >> 32, b0 = b & kEps, b1 = b >> 32;
  const uint64_t ll = a0 * b0, lh = a0 * b1, hl = a1 * b0, hh = a1 * b1;
  const uint64_t mid = (ll >> 32) + (lh & kEps) + (hl & kEps);
  const uint64_t lo = (ll & kEps) | (mid << 32);
  const uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
  const uint64_t hi_hi = hi >> 32, hi_lo = hi & kEps;
  uint64_t t0 = lo - hi_hi;
  if (lo < hi_hi) t0 -= kEps;  // borrow: + 2^64 - eps
  const uint64_t t1 = hi_lo * kEps;
  uint64_t r = t0 + t1;
  if (r < t0) r += kEps;
  return r >= kP ? r - kP : r;
}

uint64_t sbox(uint64_t x) {
  const uint64_t x2 = mul(x, x), x3 = mul(x2, x), x4 = mul(x2, x2);
  return mul(x3, x4);
}

void external_linear(uint64_t s[12]) {
  for (int o = 0; o < 12; o += 4) {
    uint64_t* x = s + o;
    const uint64_t t01 = add(x[0], x[1]), t23 = add(x[2], x[3]), t0123 = add(t01, t23);
    const uint64_t t01123 = add(t0123, x[1]), t01233 = add(t0123, x[3]);
    const uint64_t y3 = add(t01233, add(x[0], x[0])), y1 = add(t01123, add(x[2], x[2]));
    x[0] = add(t01123, t01);
    x[2] = add(t01233, t23);
    x[1] = y1;
    x[3] = y3;
  }
  uint64_t sums[4];
  for (int k = 0; k < 4; ++k) sums[k] = add(add(s[k], s[k + 4]), s[k + 8]);
  for (int i = 0; i < 12; ++i) s[i] = add(s[i], sums[i % 4]);
}

void absorb_bytes(uint64_t s[12], const uint8_t* p) {  // 8 elements from 32 bytes
  for (int i = 0; i < 8; ++i) {
    const uint32_t w = uint32_t(p[4 * i]) | uint32_t(p[4 * i + 1]) << 8 | uint32_t(p[4 * i + 2]) << 16 |
                       uint32_t(p[4 * i + 3]) << 24;
    s[i] = add(s[i], w);
  }
}

void squeeze(const uint64_t s[12], uint8_t* out) {  // state[0..4] as little-endian u64s
  for (int i = 0; i < 4; ++i)
    for (int b = 0; b < 8; ++b) out[8 * i + b] = static_cast<uint8_t>(s[i] >> (8 * b));
}

}  // namespace

void permute(uint64_t s[12]) {
  external_linear(s);
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 12; ++i) s[i] = sbox(add(s[i], kInitialRc[r][i]));
    external_linear(s);
  }
  for (int r = 0; r < 22; ++r) {
    s[0] = sbox(add(s[0], kInternalRc[r]));
    uint64_t sum = 0;
    for (int i = 0; i < 12; ++i) sum = add(sum, s[i]);
    for (int i = 0; i < 12; ++i) s[i] = add(sum, mul(s[i], kMatrixDiag[i]));
  }
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 12; ++i) s[i] = sbox(add(s[i], kTerminalRc[r][i]));
    external_linear(s);
  }
}

Hash nonce_hash(const uint8_t header[32], const Nonce& nonce) {
  uint64_t s[12] = {};
  absorb_bytes(s, header);
  permute(s);
  absorb_bytes(s, nonce.data());
  permute(s);
  absorb_bytes(s, nonce.data() + 32);
  permute(s);
  s[0] = add(s[0], 1);  // terminator element
  s[1] = add(s[1], 1);  // sponge padding
  permute(s);
  Hash h;
  squeeze(s, h.data());
  permute(s);
  squeeze(s, h.data() + 32);
  return h;
}

bool meets_target(const Hash& h, const Target& t) {
  for (int i = 0; i < 64; ++i)
    if (h[i] != t[i]) return h[i] < t[i];
  return false;
}

void prestate_low64(const uint8_t header[32], const Nonce& nonce, uint64_t out[12]) {
  uint64_t s[12] = {};
  absorb_bytes(s, header);
  permute(s);
  absorb_bytes(s, nonce.data());
  permute(s);
  // Nonce bytes 32..55 only: elements 6 and 7 (bytes 56..63) are the per-nonce part.
  uint8_t part[32] = {};
  for (int i = 0; i < 24; ++i) part[i] = nonce[32 + i];
  absorb_bytes(s, part);
  external_linear(s);
  for (int i = 0; i < 12; ++i) out[i] = add(s[i], kInitialRc[0][i]);
}

}  // namespace xm::qtc
