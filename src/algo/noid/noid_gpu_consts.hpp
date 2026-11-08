// Host helpers shared by the Parano1d GPU engine and its tests: conversions between the reference U128 and the
// kernel's W4 word layout, and the pre-split constant-multiplier operands (order: g, h, d0+1, d1+1, d2+1, d3+1).
#pragma once

#include <utility>

#include "algo/noid/noid.hpp"
#include "algo/noid/noid_consts.hpp"
#include "noid/noid_comp.h"  // kernels/noid (plain C++ when not compiled as GPU code)
#include "noid/noid_perm.h"
#include "noid/noid_t64.h"

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
  U128 c2f[128];  // flat image of kernel-basis bit n
  U128 f2c[128];  // kernel-basis image of flat bit n
};

inline bool u128_bit(const U128& v, int n) { return ((n < 64 ? v.lo >> n : v.hi >> (n - 64)) & 1u) != 0; }
inline U128 u128_unit(int n) { return n < 64 ? U128{1ull << n, 0} : U128{0, 1ull << (n - 64)}; }

// Basis b^i c^j (i < ni, j < 128 / ni), bit ni * j + i, and its inverse.
inline CompBasis make_basis(const U128& b, const U128& c, int ni) {
  CompBasis r;
  U128 cj{1, 0};
  for (int j = 0; j < 128 / ni; ++j) {
    U128 bi{1, 0};
    for (int i = 0; i < ni; ++i) {
      r.c2f[ni * j + i] = gf_mul(bi, cj);
      bi = gf_mul(bi, b);
    }
    cj = gf_mul(cj, c);
  }
  U128 v[128], m[128];  // Gauss-Jordan: v[k] (flat) = image of the kernel-basis combination m[k]
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
}

inline U128 basis_apply(const U128 (&map)[128], const U128& v) {
  U128 r;
  for (int n = 0; n < 128; ++n)
    if (u128_bit(v, n)) r ^= map[n];
  return r;
}

inline const CompBasis& comp_basis() {
  static const CompBasis b = make_basis(kCompY, kCompZ, 32);
  return b;
}

// Tower64 basis (kernels/noid/noid_t64.h): bit i of e_j is X^i Z^j, with X a flat-basis root of
// x^64 + x^4 + x^3 + x + 1 and Z a root of z^2 + z + X^-1 (found by t64/find.py, scratch tooling).
inline constexpr U128 kT64X{0x579bda638e9568a1ull, 0xa69a4f40a1109d7eull};
inline constexpr U128 kT64Z{0xbb0375c6266df84aull, 0xd3bf9e6d2bceba15ull};

inline const CompBasis& t64_basis() {
  static const CompBasis b = make_basis(kT64X, kT64Z, 64);
  return b;
}
inline U128 flat_to_t64(const U128& f) { return basis_apply(t64_basis().f2c, f); }
inline U128 t64_to_flat(const U128& c) { return basis_apply(t64_basis().c2f, c); }

// Nibble tables for CMulNib64: entry (c, k, v) = (v x^(4k)) * const_c in L. Every constant lies in GF(2^16) inside
// L = span(X^i), so only e0 of its tower64 image is set (checked by the tests).
inline void constant_tables_t64(uint32_t* tbl) {
  U128 c[6];
  constants_flat(c);
  for (int ci = 0; ci < 6; ++ci) {
    const uint64_t k = flat_to_t64(c[ci]).lo;
    for (int n = 0; n < 16; ++n)
      for (uint32_t v = 0; v < 16; ++v) {
        uint64_t lo, hi;
        tcl_ref(uint64_t(v) << (4 * n), k, lo, hi);
        const uint64_t e = tred(lo, hi);
        uint32_t* p = tbl + 2 * ((ci * 16 + n) * 16 + v);
        p[0] = static_cast<uint32_t>(e);
        p[1] = static_cast<uint32_t>(e >> 32);
      }
  }
}

struct RcT64 {  // round constants in the tower64 basis (host side)
  NOID_HDM W4 get(int lane, int r) {
    const U128c& c = kRoundConstFlat[lane][r];
    return to_w4(flat_to_t64({c.lo, c.hi}));
  }
};

inline U128 flat_to_comp(const U128& f) { return basis_apply(comp_basis().f2c, f); }
inline U128 comp_to_flat(const U128& c) { return basis_apply(comp_basis().c2f, c); }

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
