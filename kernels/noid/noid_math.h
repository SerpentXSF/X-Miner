// Ported for X-Miner from gacanga/NoidMiner gpu/noid_math.h (commit 31fa4a8), Apache License 2.0,
// Copyright 2026 gacanga. Modifications (X-Miner): HIP/AMD support (portable 16-bit multiplies and a
// shift-and-mask bit spread replace the PTX mul.wide.u16 and prmt paths on non-NVIDIA targets).
// See NOTICE and THIRD_PARTY.md.
// noid_math.h - GF(2^128) / Poseidon2b arithmetic for the NOID GPU engine.
//
// Compiles as CUDA device code (nvcc) AND as plain C++ (g++/MSVC) so that the
// exact same arithmetic can be validated on the CPU against noid_ref.cpp.
//
// Element layout: W4 {w0,w1,w2,w3}, w0 = coefficients x^0..x^31 (flat basis,
// polynomial basis modulo x^128 + x^7 + x^2 + x + 1).
//
// Carry-less multiplication without a CLMUL instruction (Pascal sm_61):
//   * 16x16 -> 32 carry-less products with ordinary 16-bit integer multiplies
//     (XMAD on Pascal) on operands split into 3 bit classes (bit i mod 3).
//     Each class product has at most 6 terms per position, so the integer
//     carries (value <= 6, 3 bits) never reach the next bit of the same class.
//   * Karatsuba 16 -> 32 -> 64 -> 128 (27 base products).
//   * The class split is linear, so the 9 Karatsuba operand words of a
//     128-bit operand are XORs of the split of its 4 words.
//   * "E/O" accumulation: the only Karatsuba terms that are not 32-bit aligned
//     are the 9 middle products of the 32-bit level (offset 16). They are
//     collected in a separate "odd" vector and shifted once at the end
//     (8 funnel shifts instead of 18 shifts).
//
// Squaring: bit spreading with PRMT byte lookups (4 nibbles -> 4 bytes per
// PRMT, see spread16).
//
// Multiplication by the fixed MDS constants uses shared-memory nibble tables
// (noid_perm.h, CMulTable) or the generic multiply with a pre-split operand.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIP__)
#define NOID_HD __host__ __device__ __forceinline__
#define NOID_HDM __host__ __device__ __forceinline__
#else
#define NOID_HD static inline
#define NOID_HDM inline
#endif

struct W4 { uint32_t w0, w1, w2, w3; };

NOID_HD W4 w4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { W4 r; r.w0 = a; r.w1 = b; r.w2 = c; r.w3 = d; return r; }
NOID_HD W4 w4x(W4 a, W4 b) { return w4(a.w0 ^ b.w0, a.w1 ^ b.w1, a.w2 ^ b.w2, a.w3 ^ b.w3); }
NOID_HD W4 w4x3(W4 a, W4 b, W4 c) { return w4(a.w0 ^ b.w0 ^ c.w0, a.w1 ^ b.w1 ^ c.w1, a.w2 ^ b.w2 ^ c.w2, a.w3 ^ b.w3 ^ c.w3); }

// funnel shift: high 32 bits of ((hi:lo) << s), 0 < s < 32
NOID_HD uint32_t fshl(uint32_t lo, uint32_t hi, int s)
{
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
    return __funnelshift_l(lo, hi, s);
#else
    return (hi << s) | (lo >> (32 - s));
#endif
}

// PTX prmt.b32 (default mode): result byte n is selected by nibble n of sel:
// bits 0..2 pick a byte of {b:a} (0..3 = a, 4..7 = b); if bit 3 is set the
// byte is replaced by the replicated sign bit of the picked byte.
NOID_HD uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel)
{
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
#else
    uint32_t r = 0;
    for (int n = 0; n < 4; n++) {
        uint32_t s = (sel >> (4 * n)) & 15u;
        uint32_t idx = s & 7u;
        uint32_t byte = idx < 4 ? (a >> (8 * idx)) & 0xFFu : (b >> (8 * (idx - 4))) & 0xFFu;
        if (s & 8u) byte = (byte & 0x80u) ? 0xFFu : 0u;
        r |= byte << (8 * n);
    }
    return r;
#endif
}

// 16x16 -> 32 integer multiply of half HA of a by half HB of b (0 = low, 1 = high).
// On the GPU this is a single XMAD with half selectors (PTX mul.wide.u16 on
// the 16-bit halves of the registers).
template <int HA, int HB>
NOID_HD uint32_t imul16(uint32_t a, uint32_t b)
{
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
    uint32_t r;
    if (HA == 0 && HB == 0)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, al, bl;}" : "=r"(r) : "r"(a), "r"(b));
    else if (HA == 1 && HB == 1)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, ah, bh;}" : "=r"(r) : "r"(a), "r"(b));
    else if (HA == 0)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, al, bh;}" : "=r"(r) : "r"(a), "r"(b));
    else
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, ah, bl;}" : "=r"(r) : "r"(a), "r"(b));
    return r;
#else
    uint32_t x = HA ? (a >> 16) : (a & 0xFFFFu);
    uint32_t y = HB ? (b >> 16) : (b & 0xFFFFu);
    return x * y;
#endif
}

#define NOID_SPLIT0 0x92499249u   // bits = 0 mod 3 inside each 16-bit half
#define NOID_SPLIT1 0x24922492u   // bits = 1 mod 3
#define NOID_SPLIT2 0x49244924u   // bits = 2 mod 3
#define NOID_PCLS0  0x49249249u   // product bits = 0 mod 3 (32-bit)
#define NOID_PCLS1  0x92492492u
#define NOID_PCLS2  0x24924924u

// a 32-bit word split in 3 classes (both 16-bit halves at once)
struct S3 { uint32_t c0, c1, c2; };

NOID_HD S3 s3(uint32_t w) { S3 r; r.c0 = w & NOID_SPLIT0; r.c1 = w & NOID_SPLIT1; r.c2 = w & NOID_SPLIT2; return r; }
NOID_HD S3 s3x(S3 a, S3 b) { S3 r; r.c0 = a.c0 ^ b.c0; r.c1 = a.c1 ^ b.c1; r.c2 = a.c2 ^ b.c2; return r; }

// carry-less 16x16 -> 32 of half HA of a by half HB of b (split words).
// The 3 class sums are merged with two bit-selects (LOP3 with an immediate):
// positions = 0 mod 3 take t0, = 1 mod 3 take t1, = 2 mod 3 take t2.
template <int HA, int HB>
NOID_HD uint32_t cm16(S3 a, S3 b)
{
    uint32_t p00 = imul16<HA, HB>(a.c0, b.c0), p01 = imul16<HA, HB>(a.c0, b.c1), p02 = imul16<HA, HB>(a.c0, b.c2);
    uint32_t p10 = imul16<HA, HB>(a.c1, b.c0), p11 = imul16<HA, HB>(a.c1, b.c1), p12 = imul16<HA, HB>(a.c1, b.c2);
    uint32_t p20 = imul16<HA, HB>(a.c2, b.c0), p21 = imul16<HA, HB>(a.c2, b.c1), p22 = imul16<HA, HB>(a.c2, b.c2);
    uint32_t t0 = p00 ^ p12 ^ p21;
    uint32_t t1 = p01 ^ p10 ^ p22;
    uint32_t t2 = p02 ^ p11 ^ p20;
    uint32_t sel = (t1 & NOID_PCLS1) | (t2 & ~NOID_PCLS1);
    return (t0 & NOID_PCLS0) | (sel & ~NOID_PCLS0);
}

// Karatsuba operand word: split of the word (both halves) and split of
// (low half ^ high half) in the low half.
struct KW { S3 s, m; };

NOID_HD KW kw(uint32_t w)
{
    KW r;
    r.s = s3(w);
    r.m = s3(w ^ (w >> 16));
    return r;
}
NOID_HD KW kwx(KW a, KW b) { KW r; r.s = s3x(a.s, b.s); r.m = s3x(a.m, b.m); return r; }

// carry-less 32x32 -> 64, E/O form:
//   product = e0 + e1 * x^32 + o * x^16
NOID_HD void cm32(KW a, KW b, uint32_t& e0, uint32_t& e1, uint32_t& o)
{
    uint32_t L = cm16<0, 0>(a.s, b.s);
    uint32_t H = cm16<1, 1>(a.s, b.s);
    uint32_t M = cm16<0, 0>(a.m, b.m);
    e0 = L;
    e1 = H;
    o = M ^ L ^ H;
}

// carry-less 64x64 -> 128 of operand pairs (u0,u1) x (v0,v1), E/O form:
//   product = sum e[j] x^(32 j) + sum o[j] x^(32 j + 16)
NOID_HD void cm64(KW u0, KW u1, KW v0, KW v1, uint32_t e[4], uint32_t o[3])
{
    uint32_t l0, l1, lo, h0, h1, ho, m0, m1, mo;
    cm32(u0, v0, l0, l1, lo);
    cm32(u1, v1, h0, h1, ho);
    cm32(kwx(u0, u1), kwx(v0, v1), m0, m1, mo);
    m0 ^= l0 ^ h0;
    m1 ^= l1 ^ h1;
    mo ^= lo ^ ho;
    e[0] = l0;
    e[1] = l1 ^ m0;
    e[2] = h0 ^ m1;
    e[3] = h1;
    o[0] = lo;
    o[1] = mo;
    o[2] = ho;
}

// reduce a 256-bit carry-less product modulo x^128 + x^7 + x^2 + x + 1
NOID_HD W4 reduce256(const uint32_t r[8])
{
    // H = r4..r7 ; H*x^128 = H*(x^7+x^2+x+1)
    uint32_t h0 = r[4], h1 = r[5], h2 = r[6], h3 = r[7];
    uint32_t t0 = r[0] ^ h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    uint32_t t1 = r[1] ^ h1 ^ fshl(h0, h1, 1) ^ fshl(h0, h1, 2) ^ fshl(h0, h1, 7);
    uint32_t t2 = r[2] ^ h2 ^ fshl(h1, h2, 1) ^ fshl(h1, h2, 2) ^ fshl(h1, h2, 7);
    uint32_t t3 = r[3] ^ h3 ^ fshl(h2, h3, 1) ^ fshl(h2, h3, 2) ^ fshl(h2, h3, 7);
    uint32_t ov = (h3 >> 31) ^ (h3 >> 30) ^ (h3 >> 25);      // x^128..x^134
    t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    return w4(t0, t1, t2, t3);
}

// Karatsuba operand: the 4 words of a 128-bit element, split
struct KOp { KW a0, a1, a2, a3; };

NOID_HD KOp kop(W4 a)
{
    KOp k; k.a0 = kw(a.w0); k.a1 = kw(a.w1); k.a2 = kw(a.w2); k.a3 = kw(a.w3);
    return k;
}

// full product of two split operands, reduced
NOID_HD W4 kmul(const KOp& a, const KOp& b)
{
    uint32_t Le[4], Lo[3], He[4], Ho[3], Me[4], Mo[3], e[8], o[7], r[8];
    cm64(a.a0, a.a1, b.a0, b.a1, Le, Lo);
    cm64(a.a2, a.a3, b.a2, b.a3, He, Ho);
    cm64(kwx(a.a0, a.a2), kwx(a.a1, a.a3), kwx(b.a0, b.a2), kwx(b.a1, b.a3), Me, Mo);
    for (int i = 0; i < 4; i++) Me[i] ^= Le[i] ^ He[i];
    for (int i = 0; i < 3; i++) Mo[i] ^= Lo[i] ^ Ho[i];
    // even part (32-bit aligned): L + M x^64 + H x^128
    e[0] = Le[0];
    e[1] = Le[1];
    e[2] = Le[2] ^ Me[0];
    e[3] = Le[3] ^ Me[1];
    e[4] = He[0] ^ Me[2];
    e[5] = He[1] ^ Me[3];
    e[6] = He[2];
    e[7] = He[3];
    // odd part (offset 16): same placement
    o[0] = Lo[0];
    o[1] = Lo[1];
    o[2] = Lo[2] ^ Mo[0];
    o[3] = Mo[1];
    o[4] = Mo[2] ^ Ho[0];
    o[5] = Ho[1];
    o[6] = Ho[2];
    // r = e + o * x^16
    r[0] = e[0] ^ (o[0] << 16);
    for (int i = 1; i < 7; i++) r[i] = e[i] ^ fshl(o[i - 1], o[i], 16);
    r[7] = e[7] ^ (o[6] >> 16);
    return reduce256(r);
}

NOID_HD W4 gmul(W4 a, W4 b)
{
    KOp ka = kop(a), kb = kop(b);
    return kmul(ka, kb);
}

// Bit spread of the low 16 bits of x into the even bits of a 32-bit word
// (nibble n -> byte n), with two PRMT byte lookups:
//   * nibble < 8 : PRMT picks spread(n) from the table {SPA, SPB};
//   * nibble >= 8: the sign flag of the selector makes that byte 0, while the
//     second PRMT (selector with bit 3 flipped) picks spread(n & 7) | 0x40.
// xf must be x ^ 0x88888888 (computed once per word by the caller).
NOID_HD uint32_t spread16p(uint32_t x, uint32_t xf)
{
    // table bytes: T0[j] = spread(j), T1[j] = spread(j) | 0x40, j = 0..7
    //   spread(0..7) = 00 01 04 05 10 11 14 15
    const uint32_t t0lo = 0x05040100u, t0hi = 0x15141110u;
    const uint32_t t1lo = 0x45444140u, t1hi = 0x55545150u;
    return prmt(t0lo, t0hi, x) | prmt(t1lo, t1hi, xf);
}

// Portable bit spread (X-Miner): low 16 bits of x into the even bits of a 32-bit word.
NOID_HD uint32_t spread16s(uint32_t x)
{
    x &= 0xFFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

NOID_HD W4 gsqr(W4 a)
{
    uint32_t r[8];
    uint32_t w[4] = { a.w0, a.w1, a.w2, a.w3 };
    for (int i = 0; i < 4; i++) {
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
        uint32_t x = w[i], xf = x ^ 0x88888888u;
        r[2 * i] = spread16p(x, xf);
        r[2 * i + 1] = spread16p(x >> 16, xf >> 16);
#else
        r[2 * i] = spread16s(w[i]);
        r[2 * i + 1] = spread16s(w[i] >> 16);
#endif
    }
    return reduce256(r);
}

// x^7 = (x * x^2) * x^4 with one squaring and one multiply instance.
// (Reusing the split of x in both multiplies saves ~60 instructions but keeps
// 24 more registers alive: at 512 threads/block (64 registers) that spills.)
#if defined(__CUDACC__) || defined(__HIP__)
#define NOID_MATH_UNROLL1 _Pragma("unroll 1")
#else
#define NOID_MATH_UNROLL1
#endif
NOID_HD W4 pow7(W4 x)
{
    W4 sq = x, x2 = x;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        sq = gsqr(sq);
        if (k == 0) x2 = sq;
    }
    // sq = x^4
    W4 a = x, b = x2;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        a = gmul(a, b);
        b = sq;
    }
    return a;
}

// ---------------------------------------------------------------------------
// MDS constants (flat basis), derived from the tower-basis matrices:
//   MDS_FULL = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]  (tower integers)
//     with g = tower(2), h = tower(4):  5 = h+1, 7 = h+g+1, 3 = g+1, 6 = h+g
//     A = h(x0+x1)  B = h(x2+x3)  C = g x1  D = g x3  S = x0+x1+x2+x3
//     y0 = A+C+D+S  y1 = A+C+x2+x3  y2 = B+C+D+S  y3 = B+D+x0+x1
//   MDS_PARTIAL = diag(0x20,0x2000,0x200,0x800) + (ones - I):
//     y_i = S + (d_i + 1) x_i
// Constant indices: 0 = g, 1 = h, 2..5 = d_0+1 .. d_3+1
// ---------------------------------------------------------------------------
#define NOID_NCONST 6
