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

// Constant multipliers for the MDS layers, in the flat basis, pre-split for the Karatsuba core.
inline void constant_kops(KOp out[6]) {
  U128 g = tower_to_flat({2, 0}), h = tower_to_flat({4, 0});
  // Partial MDS: y_i = S ^ (d_i + 1) * x_i, where d_i is the diagonal entry and "+" is XOR in the tower basis.
  U128 d[4];
  for (int i = 0; i < 4; ++i) d[i] = tower_to_flat({uint64_t(kMdsPartialTower[i][i]) ^ 1u, 0});
  out[0] = kop(to_w4(g));
  out[1] = kop(to_w4(h));
  for (int i = 0; i < 4; ++i) out[2 + i] = kop(to_w4(d[i]));
}

struct RcFlat {  // round-constant accessor for the shared permutation code (host side)
  NOID_HDM W4 get(int lane, int r) {
    const U128c& c = kRoundConstFlat[lane][r];
    return to_w4({c.lo, c.hi});
  }
};

}  // namespace xm::noid
