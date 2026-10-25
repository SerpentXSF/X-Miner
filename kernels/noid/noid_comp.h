// noid_comp.h - X-Miner: Parano1d arithmetic in a composite basis of GF(2^128).
//
// Element layout: W4 {w0,w1,w2,w3} = coefficients of 1, z, z^2, z^3 in K[z]/(q(z)), where
//   K = GF(2^32) = GF(2)[y]/(y^32 + y^7 + y^3 + y^2 + 1), polynomial basis (bit i = y^i),
//   q(z) = z^4 + z^3 + z^2 + y z + y (irreducible over K).
// The map to the flat basis is word j bit i -> Y^i Z^j, with Y, Z the flat-basis roots of the two polynomials
// (noid_gpu_consts.hpp). Why: every MDS constant (g, h, d_i + 1) lies in GF(2^16), a subfield of K, so a constant
// multiply is four independent K-products by a constant: 16 byte-table lookups (24 KB of tables for the six
// constants) instead of 64 LDS.64 into 48 KB of nibble tables in the flat basis. The general multiply stays at 9
// carry-less 32 x 32 products (Karatsuba over the z coefficients); the z^4..z^6 terms fold into the low four
// coefficients with shifts by y (one bit), then each coefficient is reduced once from 64 to 32 bits.
#pragma once
#include "noid_math.h"

// 64-bit carry-less product of two K elements (unreduced, degree <= 62). On NVIDIA: cl32 (IMAD.WIDE; RTX 3060
// 8.65 -> 9.63 MH/s in the flat basis). Elsewhere the 16-bit class-split products of NoidMiner (cm32, E/O form) on
// pre-split operands (KW): 32-bit multiplies are quarter rate on AMD (RX 9070 XT 17.7 -> 14.9 MH/s with cl32), the
// 16-bit products use the full-rate 24-bit multiplier.
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
typedef uint32_t NoidKOp;
NOID_HD NoidKOp kop1(uint32_t w) { return w; }
NOID_HD NoidKOp kopx(NoidKOp a, NoidKOp b) { return a ^ b; }
NOID_HD uint64_t kclmul(NoidKOp a, NoidKOp b) { return cl32(a, b); }
#else
typedef KW NoidKOp;
NOID_HD NoidKOp kop1(uint32_t w) { return kw(w); }
NOID_HD NoidKOp kopx(NoidKOp a, NoidKOp b) { return kwx(a, b); }
NOID_HD uint64_t kclmul(NoidKOp a, NoidKOp b)
{
    uint32_t e0, e1, o;
    cm32(a, b, e0, e1, o);
    return ((uint64_t)e1 << 32 | e0) ^ ((uint64_t)o << 16);
}
#endif

// reduce a 64-bit product modulo y^32 + y^7 + y^3 + y^2 + 1
NOID_HD uint32_t kred(uint64_t d)
{
    const uint32_t hi = (uint32_t)(d >> 32), lo = (uint32_t)d;
    const uint32_t ov = (hi >> 25) ^ (hi >> 29) ^ (hi >> 30);  // y^32..y^38 of hi * (y^7 + y^3 + y^2 + 1)
    const uint32_t f = hi ^ ov;
    return lo ^ f ^ (f << 2) ^ (f << 3) ^ (f << 7);
}

// c[0..6] (coefficients of z^0..z^6, unreduced) -> reduced element. z^4 = z^3 + z^2 + y z + y,
// z^5 = (1 + y) z^2 + y, z^6 = (1 + y) z^3 + y z.
NOID_HD W4 zfold(const uint64_t c[7])
{
    const uint64_t y4 = c[4] << 1, y5 = c[5] << 1, y6 = c[6] << 1;
    return w4(kred(c[0] ^ y4 ^ y5), kred(c[1] ^ y4 ^ y6), kred(c[2] ^ c[4] ^ c[5] ^ y5), kred(c[3] ^ c[4] ^ c[6] ^ y6));
}

NOID_HD W4 gmulc(W4 a, W4 b)
{
    const NoidKOp a0 = kop1(a.w0), a1 = kop1(a.w1), a2 = kop1(a.w2), a3 = kop1(a.w3);
    const NoidKOp b0 = kop1(b.w0), b1 = kop1(b.w1), b2 = kop1(b.w2), b3 = kop1(b.w3);
    // Karatsuba over pairs of coefficients: (u0 + u1 z)(v0 + v1 z) = L + (M + L + H) z + H z^2
    const uint64_t L0 = kclmul(a0, b0), L2 = kclmul(a1, b1), L1 = kclmul(kopx(a0, a1), kopx(b0, b1)) ^ L0 ^ L2;
    const uint64_t H0 = kclmul(a2, b2), H2 = kclmul(a3, b3), H1 = kclmul(kopx(a2, a3), kopx(b2, b3)) ^ H0 ^ H2;
    const NoidKOp s0 = kopx(a0, a2), s1 = kopx(a1, a3), t0 = kopx(b0, b2), t1 = kopx(b1, b3);
    const uint64_t m0 = kclmul(s0, t0), m2 = kclmul(s1, t1);
    const uint64_t M0 = m0 ^ L0 ^ H0, M2 = m2 ^ L2 ^ H2;
    const uint64_t M1 = kclmul(kopx(s0, s1), kopx(t0, t1)) ^ m0 ^ m2 ^ L1 ^ H1;
    uint64_t c[7] = { L0, L1, L2 ^ M0, M1, H0 ^ M2, H1, H2 };
    return zfold(c);
}

// spread the 32 bits of w into the even bits of a 64-bit word (the square of w as a polynomial)
NOID_HD uint64_t spread32(uint32_t w)
{
#if defined(__CUDA_ARCH__) && !defined(__HIP__)
    const uint32_t wf = w ^ 0x88888888u;
    return (uint64_t)spread16p(w >> 16, wf >> 16) << 32 | spread16p(w, wf);
#else
    return (uint64_t)spread16s(w >> 16) << 32 | spread16s(w);
#endif
}

NOID_HD W4 gsqrc(W4 a)
{
    uint64_t c[7] = { spread32(a.w0), 0, spread32(a.w1), 0, spread32(a.w2), 0, spread32(a.w3) };
    return zfold(c);
}

NOID_HD W4 pow7c(W4 x)
{
    W4 sq = x, x2 = x;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        sq = gsqrc(sq);
        if (k == 0) x2 = sq;
    }
    W4 a = x, b = x2;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        a = gmulc(a, b);
        b = sq;
    }
    return a;
}

// Byte tables for the constant multiplies: tbl[(c * 4 + p) * 256 + v] = (v * y^(8p)) * const_c in K (the constants
// lie in K, so the same table serves all four coefficients). NOID_CTBL_WORDS words = 24 KB.
#define NOID_CTBL_WORDS (NOID_NCONST * 4 * 256)
struct CMulByte {
    const uint32_t* tbl;
    NOID_HDM uint32_t k1(uint32_t base, uint32_t w) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // byte addresses: (byte p << 2) | (base + p * 1024); the tables of one constant are 4 KB aligned
        const char* t = reinterpret_cast<const char*>(tbl);
        const uint32_t e0 = *reinterpret_cast<const uint32_t*>(t + (((w << 2) & 0x3FCu) | base));
        const uint32_t e1 = *reinterpret_cast<const uint32_t*>(t + (((w >> 6) & 0x3FCu) | base) + 1024);
        const uint32_t e2 = *reinterpret_cast<const uint32_t*>(t + (((w >> 14) & 0x3FCu) | base) + 2048);
        const uint32_t e3 = *reinterpret_cast<const uint32_t*>(t + (((w >> 22) & 0x3FCu) | base) + 3072);
        return e0 ^ e1 ^ e2 ^ e3;
#else
        const uint32_t* tc = tbl + base / 4;
        return tc[w & 255u] ^ tc[256 + ((w >> 8) & 255u)] ^ tc[512 + ((w >> 16) & 255u)] ^ tc[768 + (w >> 24)];
#endif
    }
    NOID_HDM W4 mul(int c, W4 a) const
    {
        const uint32_t base = (uint32_t)c * 4096u;
        return w4(k1(base, a.w0), k1(base, a.w1), k1(base, a.w2), k1(base, a.w3));
    }
    NOID_HDM static W4 sbox(W4 x) { return pow7c(x); }
};
