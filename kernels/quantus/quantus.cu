// X-Miner Quantus (QPoW) search kernel, compiled for AMD (HIP) and NVIDIA (CUDA) from this one file.
//
// Per nonce the sponge needs permutations 3 to 5 of five; the host absorbs everything that does not depend on the
// low 64 nonce bits (the header, nonce bytes 0..55, the initial linear layer and first round constants of
// permutation 3), as in the official miner's prestate split (quantus-miner pow-core, Apache-2.0). The kernel adds
// the low bits' linear contribution, finishes permutation 3, runs permutation 4 and compares the top 64 bits of
// the first squeeze with the share target; permutation 5 only matters on an exact tie and is left to the host,
// which re-checks every candidate with the CPU reference.
//
// Arithmetic is lazy: elements are any u64 congruent mod p (p = 2^64 - 2^32 + 1), sums are carried in 96 bits
// and reduced once, and the rare final carry of a fold is not corrected (about 2^-32 per operation on random
// data). A slipped value makes that nonce's hash wrong, which can only lose a share, never send a bad one.
// Neither compiler keeps a carry flag alive between 32-bit adds (both rebuild it with a 64-bit compare), so the
// carry chains are written out: PTX .cc instructions on NVIDIA, VCC chains in inline assembly on AMD.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

#define XM_QTC_CONST __constant__
#include "../../src/algo/quantus/quantus_consts.hpp"

typedef uint64_t u64;
typedef uint32_t u32;

struct QtcJob {
  u64 pre[12];    // prestate (see above)
  u64 target_hi;  // top 64 bits of the 512-bit share target
};

#define c_initial xm::qtc::kInitialRc
#define c_internal xm::qtc::kInternalRc
#define c_terminal xm::qtc::kTerminalRc
#define c_diag xm::qtc::kMatrixDiag

// A value below 2^96 as three words; a 96-bit sum of field elements, reduced once.
struct Acc {
  u32 l0, l1, h;
};

__device__ __forceinline__ Acc acc(u64 x) { return {(u32)x, (u32)(x >> 32), 0u}; }

#if defined(__HIP__)

// dst = a + x (96-bit + 64-bit), three-address so no copies are needed.
__device__ __forceinline__ Acc acc_sum(const Acc& a, u64 x) {
  Acc r;
  asm("v_add_co_u32 %0, vcc_lo, %3, %6\n\t"
      "v_add_co_ci_u32_e64 %1, vcc_lo, %4, %7, vcc_lo\n\t"
      "v_add_co_ci_u32_e64 %2, null, %5, 0, vcc_lo"
      : "=&v"(r.l0), "=&v"(r.l1), "=&v"(r.h)
      : "v"(a.l0), "v"(a.l1), "v"(a.h), "v"((u32)x), "v"((u32)(x >> 32))
      : "vcc");
  return r;
}

__device__ __forceinline__ Acc acc_sum(const Acc& a, const Acc& x) {
  Acc r;
  asm("v_add_co_u32 %0, vcc_lo, %3, %6\n\t"
      "v_add_co_ci_u32_e64 %1, vcc_lo, %4, %7, vcc_lo\n\t"
      "v_add_co_ci_u32_e64 %2, null, %5, %8, vcc_lo"
      : "=&v"(r.l0), "=&v"(r.l1), "=&v"(r.h)
      : "v"(a.l0), "v"(a.l1), "v"(a.h), "v"(x.l0), "v"(x.l1), "v"(x.h)
      : "vcc");
  return r;
}

// (r1:r0) + r2 * 2^64 + r3 * 2^96  ->  (r1:r0) - r2 - r3 + r2 * 2^32  (2^64 = 2^32 - 1, 2^96 = -1). A carry out of
// the high word is worth 2^64 = 2^32 - 1: added as +1 to the high word and -1 to the low word. The borrows out of
// the high word in the two subtractions, and out of the low word in the correction, are not corrected: each needs
// a zero 32-bit word (about 2^-32).
__device__ __forceinline__ u64 fold(u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 o0, o1;
  asm("v_sub_co_u32 %0, vcc_lo, %2, %4\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %3, 0, vcc_lo\n\t"
      "v_sub_co_u32 %0, vcc_lo, %0, %5\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %1, 0, vcc_lo\n\t"
      "v_add_co_u32 %1, vcc_lo, %1, %4\n\t"
      "v_add_co_ci_u32_e64 %1, null, %1, 0, vcc_lo\n\t"
      "v_sub_co_ci_u32_e64 %0, null, %0, 0, vcc_lo"
      : "=&v"(o0), "=&v"(o1)
      : "v"(r0), "v"(r1), "v"(r2), "v"(r3)
      : "vcc");
  return ((u64)o1 << 32) | o0;
}

// A 96-bit sum: (l1:l0) - h + h * 2^32, same carry handling.
__device__ __forceinline__ u64 fold96(const Acc& a) {
  u32 o0, o1;
  asm("v_sub_co_u32 %0, vcc_lo, %2, %4\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %3, 0, vcc_lo\n\t"
      "v_add_co_u32 %1, vcc_lo, %1, %4\n\t"
      "v_add_co_ci_u32_e64 %1, null, %1, 0, vcc_lo\n\t"
      "v_sub_co_ci_u32_e64 %0, null, %0, 0, vcc_lo"
      : "=&v"(o0), "=&v"(o1)
      : "v"(a.l0), "v"(a.l1), "v"(a.h)
      : "vcc");
  return ((u64)o1 << 32) | o0;
}

__device__ __forceinline__ u64 gadd(u64 a, u64 b) {
  u32 o0, o1;
  asm("v_add_co_u32 %0, vcc_lo, %2, %4\n\t"
      "v_add_co_ci_u32_e64 %1, vcc_lo, %3, %5, vcc_lo\n\t"
      "v_add_co_ci_u32_e64 %1, null, %1, 0, vcc_lo\n\t"
      "v_sub_co_ci_u32_e64 %0, null, %0, 0, vcc_lo"
      : "=&v"(o0), "=&v"(o1)
      : "v"((u32)a), "v"((u32)(a >> 32)), "v"((u32)b), "v"((u32)(b >> 32))
      : "vcc");
  return ((u64)o1 << 32) | o0;
}

// a * b + w (w < 2^96) as four words: four 32x32 -> 64 multiply-adds, the addend riding in their accumulators.
// Needs b < 2^64 - 2^36 or a small w so nothing overflows (true for the diagonal and for the w = 0 callers).
__device__ __forceinline__ void mul_wide(u64 a, u64 b, const Acc& w, u32& r0, u32& r1, u32& r2, u32& r3) {
  const u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
  const u64 p00 = (u64)a0 * b0 + w.l0;
  const u64 p01 = (u64)a0 * b1 + ((p00 >> 32) + w.l1);
  const u64 p10 = (u64)a1 * b0 + (u32)p01;
  const u64 p11 = (u64)a1 * b1 + ((p01 >> 32) + (p10 >> 32) + w.h);
  r0 = (u32)p00;
  r1 = (u32)p10;
  r2 = (u32)p11;
  r3 = (u32)(p11 >> 32);
}

// AMD reductions: XM_QTC_AFOLD 0 = add/sub chains only, 1 = fold_mad for products, 2 = also fold96_mad for the
// linear layers (RX 9070 XT stock: 183.1 / 192.6 / 196.0 MH/s). XM_QTC_ASQR 1 squares with four products (slower:
// the extra zero-extended addends cost more moves than the shifts they replace).
#ifndef XM_QTC_AFOLD
#define XM_QTC_AFOLD 2
#endif
#ifndef XM_QTC_ASQR
#define XM_QTC_ASQR 0
#endif

// fold() with r2 * (2^32 - 1) + (r1:r0) on the multiplier: its carry c is worth 2^64 = 2^32 - 1 (+c high, -c low).
__device__ __forceinline__ u64 fold_mad(u32 r0, u32 r1, u32 r2, u32 r3) {
  u64 t;
  u32 c, o0, o1;
  asm("v_mad_co_u64_u32 %0, %1, %2, %3, %4" : "=&v"(t), "=&s"(c) : "v"(r2), "s"(0xffffffffu), "v"(((u64)r1 << 32) | r0));
  asm("v_sub_co_u32 %0, vcc_lo, %2, %4\n\t"
      "v_sub_co_ci_u32_e64 %1, null, %3, 0, vcc_lo\n\t"
      "v_add_co_ci_u32_e64 %1, null, %1, 0, %5\n\t"
      "v_sub_co_ci_u32_e64 %0, null, %0, 0, %5"
      : "=&v"(o0), "=&v"(o1)
      : "v"((u32)t), "v"((u32)(t >> 32)), "v"(r3), "s"(c)
      : "vcc");
  return ((u64)o1 << 32) | o0;
}

// fold96() the same way: h * (2^32 - 1) + (l1:l0), carry c corrected as above.
__device__ __forceinline__ u64 fold96_mad(const Acc& a) {
  u64 t;
  u32 c, o0, o1;
  asm("v_mad_co_u64_u32 %0, %1, %2, %3, %4" : "=&v"(t), "=&s"(c) : "v"(a.h), "s"(0xffffffffu), "v"(((u64)a.l1 << 32) | a.l0));
  asm("v_add_co_ci_u32_e64 %1, null, %3, 0, %4\n\t"
      "v_sub_co_ci_u32_e64 %0, null, %2, 0, %4"
      : "=&v"(o0), "=&v"(o1)
      : "v"((u32)t), "v"((u32)(t >> 32)), "s"(c));
  return ((u64)o1 << 32) | o0;
}

#if XM_QTC_AFOLD >= 2
#define FOLD96 fold96_mad
#else
#define FOLD96 fold96
#endif
#if XM_QTC_AFOLD
#define FOLD_SQ fold_mad
#define FOLD_MUL fold_mad
#define FOLD_DIAG fold_mad
#else
#define FOLD_SQ fold
#define FOLD_MUL fold
#define FOLD_DIAG fold
#endif

// a^2 with three products: a0^2 + 2 a0 a1 2^32 + a1^2 2^64.
__device__ __forceinline__ u64 gsqr(u64 a) {
#if XM_QTC_ASQR
  u32 r0, r1, r2, r3;
  mul_wide(a, a, Acc{0u, 0u, 0u}, r0, r1, r2, r3);
  return FOLD_SQ(r0, r1, r2, r3);
#else
  const u32 a0 = (u32)a, a1 = (u32)(a >> 32);
  const u64 ll = (u64)a0 * a0, lh = (u64)a0 * a1, hh = (u64)a1 * a1;
  // (hh:ll) + (lh << 33) as a 128-bit sum: lh << 1 is 65 bits.
  const u64 mid = (ll >> 32) + ((lh << 1) & 0xffffffffull);
  const u64 top = hh + (lh >> 31) + (mid >> 32);
  return FOLD_SQ((u32)ll, (u32)mid, (u32)top, (u32)(top >> 32));
#endif
}

#else  // NVIDIA

__device__ __forceinline__ Acc acc_sum(const Acc& a, u64 x) {
  Acc r;
  asm("add.cc.u32 %0, %3, %6;\n\t"
      "addc.cc.u32 %1, %4, %7;\n\t"
      "addc.u32 %2, %5, 0;"
      : "=r"(r.l0), "=r"(r.l1), "=r"(r.h)
      : "r"(a.l0), "r"(a.l1), "r"(a.h), "r"((u32)x), "r"((u32)(x >> 32)));
  return r;
}

__device__ __forceinline__ Acc acc_sum(const Acc& a, const Acc& x) {
  Acc r;
  asm("add.cc.u32 %0, %3, %6;\n\t"
      "addc.cc.u32 %1, %4, %7;\n\t"
      "addc.u32 %2, %5, %8;"
      : "=r"(r.l0), "=r"(r.l1), "=r"(r.h)
      : "r"(a.l0), "r"(a.l1), "r"(a.h), "r"(x.l0), "r"(x.l1), "r"(x.h));
  return r;
}

// Same reduction as the AMD path, on the integer ALU pipe so the multiply pipe only multiplies: (r1:r0) - r2 - r3 +
// r2 * 2^32, a carry out of the high word (2^64 = 2^32 - 1) added as +1 high / -1 low, rare borrows uncorrected.
// The carry is taken as a value: PTX subc after add.cc consumes CF as an inverted borrow.
__device__ __forceinline__ u64 fold(u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 o0, o1, c;
  asm("sub.cc.u32 %0, %3, %5;\n\t"
      "subc.u32 %1, %4, 0;\n\t"
      "sub.cc.u32 %0, %0, %6;\n\t"
      "subc.u32 %1, %1, 0;\n\t"
      "add.cc.u32 %1, %1, %5;\n\t"
      "addc.u32 %2, 0, 0;\n\t"
      "add.u32 %1, %1, %2;\n\t"
      "sub.u32 %0, %0, %2;"
      : "=&r"(o0), "=&r"(o1), "=&r"(c)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return ((u64)o1 << 32) | o0;
}

// The same value with r2 * eps formed on the multiply pipe (quantus-miner engine-cuda reduce128, Apache-2.0): the
// carry c out of (r1:r0) + r2 * eps is worth 2^32 - 1, so it rides into the high word and the subtrahend r3 + c.
// The kernel mixes the two folds to keep the ALU and FMA pipes evenly loaded.
__device__ __forceinline__ u64 fold_fma(u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 o0, o1;
  asm("{\n\t.reg .u32 c;\n\t"
      "mad.lo.cc.u32 %0, %4, 0xffffffff, %2;\n\t"
      "madc.hi.cc.u32 %1, %4, 0xffffffff, %3;\n\t"
      "addc.u32 c, %5, 0;\n\t"
      "addc.u32 %1, %1, 0;\n\t"
      "sub.cc.u32 %0, %0, c;\n\t"
      "subc.u32 %1, %1, 0;\n\t}"
      : "=&r"(o0), "=&r"(o1)
      : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return ((u64)o1 << 32) | o0;
}

__device__ __forceinline__ u64 fold96(const Acc& a) {
  u32 o0, o1, c;
  asm("sub.cc.u32 %0, %3, %5;\n\t"
      "subc.u32 %1, %4, 0;\n\t"
      "add.cc.u32 %1, %1, %5;\n\t"
      "addc.u32 %2, 0, 0;\n\t"
      "add.u32 %1, %1, %2;\n\t"
      "sub.u32 %0, %0, %2;"
      : "=&r"(o0), "=&r"(o1), "=&r"(c)
      : "r"(a.l0), "r"(a.l1), "r"(a.h));
  return ((u64)o1 << 32) | o0;
}

__device__ __forceinline__ u64 gadd(u64 a, u64 b) {
  u32 o0, o1, c;
  asm("add.cc.u32 %0, %3, %5;\n\t"
      "addc.cc.u32 %1, %4, %6;\n\t"
      "addc.u32 %2, 0, 0;\n\t"
      "add.u32 %1, %1, %2;\n\t"
      "sub.u32 %0, %0, %2;"
      : "=&r"(o0), "=&r"(o1), "=&r"(c)
      : "r"((u32)a), "r"((u32)(a >> 32)), "r"((u32)b), "r"((u32)(b >> 32)));
  return ((u64)o1 << 32) | o0;
}

// Which reduction each call site uses (XM_QTC_MIX, bit 0: squares, bit 1: multiplies, bit 2: internal-layer
// products; a set bit selects the multiply-pipe fold). 2 measured best on an RTX 3060 (148.9 vs 144.5 MH/s for 0).
#ifndef XM_QTC_MIX
#define XM_QTC_MIX 2
#endif
#define FOLD96 fold96
#define FOLD_SQ ((XM_QTC_MIX & 1) ? fold_fma : fold)
#define FOLD_MUL ((XM_QTC_MIX & 2) ? fold_fma : fold)
#define FOLD_DIAG ((XM_QTC_MIX & 4) ? fold_fma : fold)

// a * b + w (w < 2^96) as four words: schoolbook 32-bit lo/hi multiply-adds with the carries in the chain. IMAD.WIDE
// costs two issue slots and needs a zeroed register pair per addend, so plain lo/hi pairs come out cheaper.
// Needs b < 2^64 - 2^36 or a small w so nothing overflows (true for the diagonal and for the w = 0 callers).
__device__ __forceinline__ void mul_wide(u64 a, u64 b, const Acc& w, u32& r0, u32& r1, u32& r2, u32& r3) {
  const u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
  asm("mad.lo.cc.u32 %0, %4, %6, %8;\n\t"
      "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
      "madc.lo.cc.u32 %2, %5, %7, %10;\n\t"
      "madc.hi.u32 %3, %5, %7, 0;\n\t"
      "mad.lo.cc.u32 %1, %4, %7, %1;\n\t"
      "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
      "addc.u32 %3, %3, 0;\n\t"
      "mad.lo.cc.u32 %1, %5, %6, %1;\n\t"
      "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
      "addc.u32 %3, %3, 0;"
      : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(w.l0), "r"(w.l1), "r"(w.h));
}

// a^2 with three products: a0^2 + 2 a0 a1 2^32 + a1^2 2^64.
__device__ __forceinline__ u64 gsqr(u64 a) {
  const u32 a0 = (u32)a, a1 = (u32)(a >> 32);
  u32 r0, r1, r2, r3;
  asm("{\n\t.reg .u32 m0, m1, m2;\n\t"
      "mul.lo.u32 m0, %4, %5;\n\t"
      "mul.hi.u32 m1, %4, %5;\n\t"
      "add.cc.u32 m0, m0, m0;\n\t"
      "addc.cc.u32 m1, m1, m1;\n\t"
      "addc.u32 m2, 0, 0;\n\t"
      "mul.lo.u32 %0, %4, %4;\n\t"
      "mul.hi.u32 %1, %4, %4;\n\t"
      "mul.lo.u32 %2, %5, %5;\n\t"
      "mul.hi.u32 %3, %5, %5;\n\t"
      "add.cc.u32 %1, %1, m0;\n\t"
      "addc.cc.u32 %2, %2, m1;\n\t"
      "addc.u32 %3, %3, m2;\n\t}"
      : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
      : "r"(a0), "r"(a1));
  return FOLD_SQ(r0, r1, r2, r3);
}

#endif

__device__ __forceinline__ u64 gmul(u64 a, u64 b) {
  u32 r0, r1, r2, r3;
  mul_wide(a, b, Acc{0u, 0u, 0u}, r0, r1, r2, r3);
  return FOLD_MUL(r0, r1, r2, r3);
}

__device__ __forceinline__ u64 sbox(u64 x) {
  const u64 x2 = gsqr(x);
  const u64 x3 = gmul(x2, x);
  const u64 x4 = gsqr(x2);
  return gmul(x3, x4);
}

// Poseidon2 external linear layer with the next round's constants folded in: M4 on each group of 4, column sums
// added, + rc, one reduction per element.
__device__ __forceinline__ void external(u64 s[12], const u64* rc) {
  Acc y[12];
#pragma unroll
  for (int o = 0; o < 12; o += 4) {
    const u64 x0 = s[o], x1 = s[o + 1], x2 = s[o + 2], x3 = s[o + 3];
    const Acc t01 = acc_sum(acc(x0), x1), t23 = acc_sum(acc(x2), x3);
    const Acc t0123 = acc_sum(t01, t23);
    const Acc t01123 = acc_sum(t0123, x1), t01233 = acc_sum(t0123, x3);
    y[o + 3] = acc_sum(acc_sum(t01233, x0), x0);
    y[o + 1] = acc_sum(acc_sum(t01123, x2), x2);
    y[o] = acc_sum(t01123, t01);
    y[o + 2] = acc_sum(t01233, t23);
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const Acc sum = acc_sum(acc_sum(y[k], y[k + 4]), y[k + 8]);
#pragma unroll
    for (int j = k; j < 12; j += 4) {
      const Acc v = acc_sum(y[j], sum);
      s[j] = FOLD96(rc ? acc_sum(v, rc[j]) : v);
    }
  }
}

// One internal round on an already S-boxed s0: s_i = sum + diag_i * s_i, the 96-bit sum riding in the product.
__device__ __forceinline__ void internal(u64 s[12]) {
  Acc sum = acc(s[0]);
#pragma unroll
  for (int i = 1; i < 12; ++i) sum = acc_sum(sum, s[i]);
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    u32 r0, r1, r2, r3;
    mul_wide(s[i], c_diag[i], sum, r0, r1, r2, r3);
    s[i] = FOLD_DIAG(r0, r1, r2, r3);
  }
}

// The rest of a permutation from just before the first S-box (initial linear layer and rc[0] already applied), for
// N independent states side by side so the multiply-heavy and add-heavy stretches of different nonces can overlap.
template <int N>
__device__ __forceinline__ void permute_tail(u64 (&s)[N][12]) {
#ifndef XM_QTC_FUNROLL
#define XM_QTC_FUNROLL 1
#endif
#pragma unroll XM_QTC_FUNROLL
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < 12; ++i)
#pragma unroll
      for (int n = 0; n < N; ++n) s[n][i] = sbox(s[n][i]);
#pragma unroll
    for (int n = 0; n < N; ++n) external(s[n], r < 3 ? c_initial[r + 1] : nullptr);
  }
#ifndef XM_QTC_IUNROLL
#define XM_QTC_IUNROLL 1
#endif
#pragma unroll XM_QTC_IUNROLL
  for (int r = 0; r < 22; ++r) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
      s[n][0] = sbox(gadd(s[n][0], c_internal[r]));
      internal(s[n]);
    }
  }
#pragma unroll XM_QTC_FUNROLL
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < 12; ++i)
#pragma unroll
      for (int n = 0; n < N; ++n) s[n][i] = sbox(gadd(s[n][i], c_terminal[r][i]));
#pragma unroll
    for (int n = 0; n < N; ++n) external(s[n], nullptr);
  }
}

__device__ __forceinline__ u32 bswap32(u32 v) {
#if defined(__HIP__)
  return __builtin_bswap32(v);
#else
  return __byte_perm(v, 0, 0x0123);
#endif
}

// Nonces per thread (XM_QTC_NPT; the host's launch must match, see quantus_gpu.cpp).
#ifndef XM_QTC_NPT
#define XM_QTC_NPT 1
#endif

// Grid of 1-D blocks, XM_QTC_NPT nonces per thread: low 64 nonce bits = nonce_base + global thread index * NPT + k
// (the host keeps a batch from carrying into bit 64). found[0] counts candidates; found[1 + 2k], found[2 + 2k] hold
// candidate k's low 64 nonce bits, low word first.
extern "C" __global__ void xm_quantus_search(QtcJob job, u64 nonce_base, u32* __restrict__ found, u32 max_found) {
  constexpr int N = XM_QTC_NPT;
  const u64 n0 = nonce_base + ((u64)blockIdx.x * blockDim.x + threadIdx.x) * N;
  u64 s[N][12];
#pragma unroll
  for (int k = 0; k < N; ++k) {
    // Nonce bytes 56..63 big-endian are elements 6 and 7 (little-endian u32s); through M4 on the second group and
    // the column sums they add [a+b, 3a+b, 2a+3b, a+2b] to groups 0 and 2 and twice that to group 1.
    const u64 n = n0 + k;
    const u64 a = bswap32((u32)(n >> 32)), b = bswap32((u32)n);
    const u64 c[4] = {a + b, 3 * a + b, 2 * a + 3 * b, a + 2 * b};
#pragma unroll
    for (int i = 0; i < 12; ++i) s[k][i] = gadd(job.pre[i], (i >= 4 && i < 8) ? 2 * c[i & 3] : c[i & 3]);
  }
  permute_tail<N>(s);
  // Permutation 4: absorb the terminator and padding elements, then a full permutation.
#pragma unroll
  for (int k = 0; k < N; ++k) {
    s[k][0] = gadd(s[k][0], 1);
    s[k][1] = gadd(s[k][1], 1);
    external(s[k], c_initial[0]);
  }
  permute_tail<N>(s);
#pragma unroll
  for (int k = 0; k < N; ++k) {
    u64 h = s[k][0];
    if (h >= 0xffffffff00000001ull) h -= 0xffffffff00000001ull;  // canonical (lazy values can sit in [p, 2^64))
    // The first 8 output bytes are s0 little-endian; read big-endian that is bswap64(s0).
    const u64 top = ((u64)bswap32((u32)h) << 32) | bswap32((u32)(h >> 32));
    if (top <= job.target_hi) {
      const u32 slot = atomicAdd(found, 1u);
      if (slot < max_found) {
        found[1 + 2 * slot] = (u32)(n0 + k);
        found[2 + 2 * slot] = (u32)((n0 + k) >> 32);
      }
    }
  }
}
