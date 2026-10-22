#include "algo/noid/noid.hpp"

#include "algo/noid/noid_consts.hpp"

namespace xm::noid {

namespace {

U128 c(const U128c& v) { return {v.lo, v.hi}; }

// Carry-less 64x64 -> 128.
void clmul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
  lo = hi = 0;
  for (int i = 0; i < 64; ++i)
    if ((b >> i) & 1) {
      lo ^= a << i;
      if (i) hi ^= a >> (64 - i);
    }
}

// Multiply a 128-bit value by q = x^7 + x^2 + x + 1; returns the low 128 bits and the overflow (<= 7 bits).
U128 mul_q(U128 v, uint64_t& over) {
  U128 r = v;
  over = 0;
  for (int s : {1, 2, 7}) {
    r.lo ^= v.lo << s;
    r.hi ^= (v.hi << s) | (v.lo >> (64 - s));
    over ^= v.hi >> (64 - s);
  }
  return r;
}

}  // namespace

U128 gf_mul(U128 a, U128 b) {
  // 256-bit carry-less product (schoolbook), then reduce mod x^128 + x^7 + x^2 + x + 1.
  uint64_t ll, lh, hl, hh, ml, mh, nl, nh;
  clmul64(a.lo, b.lo, ll, lh);
  clmul64(a.hi, b.hi, hl, hh);
  clmul64(a.lo, b.hi, ml, mh);
  clmul64(a.hi, b.lo, nl, nh);
  uint64_t w0 = ll, w1 = lh ^ ml ^ nl, w2 = hl ^ mh ^ nh, w3 = hh;
  U128 low{w0, w1}, high{w2, w3};
  uint64_t over;
  U128 hq = mul_q(high, over);   // high * x^128 == high * q
  U128 r = low ^ hq;
  uint64_t over2;
  U128 oq = mul_q(U128{over, 0}, over2);  // over * x^128 == over * q (fits: deg <= 13)
  return r ^ oq;
}

namespace {

U128 linmap(const U128c* m, U128 v) {
  U128 r;
  for (int i = 0; i < 64; ++i) {
    if ((v.lo >> i) & 1) r ^= c(m[i]);
    if ((v.hi >> i) & 1) r ^= c(m[64 + i]);
  }
  return r;
}

U128 sbox(U128 x) {
  U128 x2 = gf_mul(x, x), x4 = gf_mul(x2, x2);
  return gf_mul(gf_mul(x, x2), x4);
}

void mds(State& s, const uint32_t tower[4][4], const U128c flat[4][4]) {
  State y;
  for (int i = 0; i < 4; ++i) {
    U128 acc;
    for (int j = 0; j < 4; ++j) acc ^= tower[i][j] == 1 ? s[j] : gf_mul(s[j], c(flat[i][j]));
    y[i] = acc;
  }
  s = y;
}

Digest squeeze(const State& s) {
  Digest d;
  U128 a = flat_to_tower(s[0]), b = flat_to_tower(s[1]);
  for (int i = 0; i < 8; ++i) {
    d[i] = static_cast<uint8_t>(a.lo >> (8 * i));
    d[8 + i] = static_cast<uint8_t>(a.hi >> (8 * i));
    d[16 + i] = static_cast<uint8_t>(b.lo >> (8 * i));
    d[24 + i] = static_cast<uint8_t>(b.hi >> (8 * i));
  }
  return d;
}

}  // namespace

U128 tower_to_flat(U128 v) { return linmap(kTowerToFlat, v); }
U128 flat_to_tower(U128 v) { return linmap(kFlatToTower, v); }

void permute_flat(State& s) {
  mds(s, kMdsFullTower, kMdsFullFlat);
  for (int r = 0; r < 66; ++r) {
    if (r < 4 || r >= 62) {
      for (int i = 0; i < 4; ++i) s[i] = sbox(s[i] ^ c(kRoundConstFlat[i][r]));
      mds(s, kMdsFullTower, kMdsFullFlat);
    } else {
      s[0] = sbox(s[0] ^ c(kRoundConstFlat[0][r]));
      mds(s, kMdsPartialTower, kMdsPartialFlat);
    }
  }
}

State midstate(const Fields& f) {
  State s{U128{}, U128{}, c(kIvFlat[0]), c(kIvFlat[1])};
  for (int k = 0; k < 5; ++k) {
    s[0] ^= tower_to_flat(f[2 * k]);
    s[1] ^= tower_to_flat(f[2 * k + 1]);
    permute_flat(s);
  }
  return s;
}

Digest digest_from_midstate(const State& mid, const Fields& f, U128 nonce) {
  State s = mid;
  s[0] ^= tower_to_flat(nonce);
  s[1] ^= tower_to_flat(f[11]);
  permute_flat(s);
  for (int k = 6; k < 8; ++k) {
    s[0] ^= tower_to_flat(f[2 * k]);
    s[1] ^= tower_to_flat(f[2 * k + 1]);
    permute_flat(s);
  }
  return squeeze(s);
}

Digest digest(const Fields& f) { return digest_from_midstate(midstate(f), f, f[kNonceField]); }

bool less_le256(const Digest& d, const uint8_t t[32]) {
  for (int i = 31; i >= 0; --i)
    if (d[i] != t[i]) return d[i] < t[i];
  return false;
}

U128 parse_u128_be_hex(const std::string& h) {
  std::string s = std::string(32 - h.size(), '0') + h;
  return {std::stoull(s.substr(16, 16), nullptr, 16), std::stoull(s.substr(0, 16), nullptr, 16)};
}

U128 u128_from_le_bytes(const uint8_t* p) {
  U128 v;
  for (int i = 0; i < 8; ++i) {
    v.lo |= uint64_t(p[i]) << (8 * i);
    v.hi |= uint64_t(p[8 + i]) << (8 * i);
  }
  return v;
}

}  // namespace xm::noid
