// Host helpers shared by the Parano1d GPU engine and its tests: conversions between the reference U128 and the
// kernel's W4 word layout, and the pre-split constant-multiplier operands (order: g, h, d0+1, d1+1, d2+1, d3+1).
#pragma once

#include <utility>

#include "algo/noid/noid.hpp"
#include "algo/noid/noid_consts.hpp"
#include "noid/noid_comp.h"  // kernels/noid (plain C++ when not compiled as GPU code)
#include "noid/noid_perm.h"

namespace xm::noid {

inline W4 to_w4(const U128& v) {
  return w4(static_cast<uint32_t>(v.lo), static_cast<uint32_t>(v.lo >> 32), static_cast<uint32_t>(v.hi),
            static_cast<uint32_t>(v.hi >> 32));
}
inline U128 from_w4(const W4& w) {
  return {uint64_t(w.w0) | uint64_t(w.w1) << 32, uint64_t(w.w2) | uint64_t(w.w3) << 32};
}

// The six MDS constants in the flat basis: g, h (full MDS) and d_i + 1 (partial MDS: y_i = S ^ (d_i + 1) * x_i,
// where d_i is the diagonal entry and "+" is XOR in the tower basis).
inline void constants_flat(U128 out[6]) {
  out[0] = tower_to_flat({2, 0});
  out[1] = tower_to_flat({4, 0});
  for (int i = 0; i < 4; ++i) out[2 + i] = tower_to_flat({uint64_t(kMdsPartialTower[i][i]) ^ 1u, 0});
}

// Pre-split operands for the Karatsuba constant multiplier.
inline void constant_kops(KOp out[6]) {
  U128 c[6];
  constants_flat(c);
  for (int i = 0; i < 6; ++i) out[i] = kop(to_w4(c[i]));
}

// Nibble tables for CMulTable: tbl[(((c*2 + half)*32 + nibble)*16 + value)*2 + {0,1}] = words (w0,w1) or (w2,w3)
// of (value * x^(4*nibble)) * const_c. NOID_TBL_WORDS words (48 KB).
inline void constant_tables(uint32_t* tbl) {
  U128 c[6];
  constants_flat(c);
  for (int ci = 0; ci < 6; ++ci)
    for (int k = 0; k < 32; ++k)
      for (uint32_t v = 0; v < 16; ++v) {
        U128 x;
        int bit = 4 * k;
        if (bit < 64) x.lo = uint64_t(v) << bit; else x.hi = uint64_t(v) << (bit - 64);
        W4 p = to_w4(gf_mul(x, c[ci]));
        uint32_t* e0 = tbl + (((ci * 2 + 0) * 32 + k) * 16 + v) * 2;
        uint32_t* e1 = tbl + (((ci * 2 + 1) * 32 + k) * 16 + v) * 2;
        e0[0] = p.w0;
        e0[1] = p.w1;
        e1[0] = p.w2;
        e1[1] = p.w3;
      }
}

// Composite basis (kernels/noid/noid_comp.h): word j bit i of a W4 is Y^i Z^j, with Y a flat-basis root of
// y^32 + y^7 + y^3 + y^2 + 1 and Z a root of z^4 + z^3 + z^2 + Y z + Y (found by noidcomp/find.py, scratch tooling).
// Composite elements travel as U128 in the to_w4 layout (bit 32 j + i).
inline constexpr U128 kCompY{0x31ee2172561b7483ull, 0xb30bd434e3cd3098ull};
inline constexpr U128 kCompZ{0x0ae9b93755c26f77ull, 0x3054b334c3f18e7cull};

struct CompBasis {
  U128 c2f[128];  // flat image of composite bit n
  U128 f2c[128];  // composite image of flat bit n
};

inline bool u128_bit(const U128& v, int n) { return ((n < 64 ? v.lo >> n : v.hi >> (n - 64)) & 1u) != 0; }
inline U128 u128_unit(int n) { return n < 64 ? U128{1ull << n, 0} : U128{0, 1ull << (n - 64)}; }

inline const CompBasis& comp_basis() {
  static const CompBasis b = [] {
    CompBasis r;
    U128 zj{1, 0};
    for (int j = 0; j < 4; ++j) {
      U128 yi{1, 0};
      for (int i = 0; i < 32; ++i) {
        r.c2f[32 * j + i] = gf_mul(yi, zj);
        yi = gf_mul(yi, kCompY);
      }
      zj = gf_mul(zj, kCompZ);
    }
    U128 v[128], m[128];  // Gauss-Jordan: v[k] (flat) = image of the composite combination m[k]
    for (int n = 0; n < 128; ++n) v[n] = r.c2f[n], m[n] = u128_unit(n);
    for (int p = 0; p < 128; ++p) {
      int q = p;
      while (!u128_bit(v[q], p)) ++q;  // the basis is invertible: a pivot always exists
      std::swap(v[p], v[q]);
      std::swap(m[p], m[q]);
      for (int k = 0; k < 128; ++k)
        if (k != p && u128_bit(v[k], p)) v[k] ^= v[p], m[k] ^= m[p];
    }
    for (int n = 0; n < 128; ++n) r.f2c[n] = m[n];
    return r;
  }();
  return b;
}

inline U128 flat_to_comp(const U128& f) {
  U128 r;
  for (int n = 0; n < 128; ++n)
    if (u128_bit(f, n)) r ^= comp_basis().f2c[n];
  return r;
}
inline U128 comp_to_flat(const U128& c) {
  U128 r;
  for (int n = 0; n < 128; ++n)
    if (u128_bit(c, n)) r ^= comp_basis().c2f[n];
  return r;
}

// Byte tables for CMulByte: tbl[(c * 4 + p) * 256 + v] = (v y^(8p)) * const_c in K. Every constant lies in GF(2^16)
// inside K = span(Y^i), so only word 0 of its composite image is set (checked by the tests).
inline void constant_tables_comp(uint32_t* tbl) {
  U128 c[6];
  constants_flat(c);
  for (int ci = 0; ci < 6; ++ci) {
    const uint32_t k = to_w4(flat_to_comp(c[ci])).w0;
    for (int p = 0; p < 4; ++p)
      for (uint32_t v = 0; v < 256; ++v) tbl[(ci * 4 + p) * 256 + v] = kred(cl32(v << (8 * p), k));
  }
}

struct RcComp {  // round constants in the composite basis (host side)
  NOID_HDM W4 get(int lane, int r) {
    const U128c& c = kRoundConstFlat[lane][r];
    return to_w4(flat_to_comp({c.lo, c.hi}));
  }
};

struct RcFlat {  // round-constant accessor for the shared permutation code (host side)
  NOID_HDM W4 get(int lane, int r) {
    const U128c& c = kRoundConstFlat[lane][r];
    return to_w4({c.lo, c.hi});
  }
};

}  // namespace xm::noid
