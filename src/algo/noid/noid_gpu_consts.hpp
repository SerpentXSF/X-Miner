// Host helpers shared by the Parano1d GPU engine and its tests: conversions between the reference U128 and the
// kernel's W4 word layout, and the pre-split constant-multiplier operands (order: g, h, d0+1, d1+1, d2+1, d3+1).
#pragma once

#include "algo/noid/noid.hpp"
#include "algo/noid/noid_consts.hpp"
#include "noid/noid_perm.h"  // kernels/noid (plain C++ when not compiled as GPU code)

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

struct RcFlat {  // round-constant accessor for the shared permutation code (host side)
  NOID_HDM W4 get(int lane, int r) {
    const U128c& c = kRoundConstFlat[lane][r];
    return to_w4({c.lo, c.hi});
  }
};

}  // namespace xm::noid
