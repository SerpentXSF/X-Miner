// noid_t64.h - X-Miner: Parano1d arithmetic in a 64-bit tower basis of GF(2^128), for GPUs with a 64 x 64-bit
// carry-less multiply (NVIDIA CLMAD, CUDA 13.3+).
//
// Element layout: W4 {w0,w1,w2,w3} = (e0, e1) with e0 = w0 | w1 << 32, e1 = w2 | w3 << 32, the coefficients of 1
// and z in L[z]/(z^2 + z + w), where
//   L = GF(2^64) = GF(2)[x]/(x^64 + x^4 + x^3 + x + 1), polynomial basis (bit i = x^i),
//   w = x^-1 = x^63 + x^3 + x^2 + 1 (trace 1, so z^2 + z + w is irreducible over L).
// The map to the flat basis is bit i of e_j -> X^i Z^j, with X, Z flat-basis roots of the two polynomials
// (noid_gpu_consts.hpp). Why: the general multiply is three 64 x 64 carry-less products (Karatsuba over z) =
// 6 CLMAD, against 9 32 x 32 products in the composite basis of noid_comp.h, each using a quarter of the CLMAD.
// The MDS constants lie in GF(2^16), inside L, so a constant multiply is two independent L-products by a
// constant: 16 nibble-table lookups (LDS.64, 128-byte tables, no bank conflicts) per coefficient.
#pragma once
#include "noid_comp.h"

#if defined(__CUDA_ARCH__) && !defined(__HIP__)
#if !defined(NOID_CLMAD)
#error "noid_t64.h needs NOID_CLMAD (CUDA 13.3+)"
#endif
NOID_HD uint64_t tclo(uint64_t a, uint64_t b, uint64_t c)
{
    uint64_t d;
    asm("clmad.lo.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
}
NOID_HD uint64_t tchi(uint64_t a, uint64_t b, uint64_t c)
{
    uint64_t d;
    asm("clmad.hi.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
}
#else
NOID_HD void tcl_ref(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
    lo = hi = 0;
    for (int i = 0; i < 64; i++)
        if ((b >> i) & 1) {
            lo ^= a << i;
            if (i) hi ^= a >> (64 - i);
        }
}
NOID_HD uint64_t tclo(uint64_t a, uint64_t b, uint64_t c) { uint64_t l, h; tcl_ref(a, b, l, h); return l ^ c; }
NOID_HD uint64_t tchi(uint64_t a, uint64_t b, uint64_t c) { uint64_t l, h; tcl_ref(a, b, l, h); return h ^ c; }
#endif

NOID_HD uint64_t te0(W4 a) { return (uint64_t)a.w1 << 32 | a.w0; }
NOID_HD uint64_t te1(W4 a) { return (uint64_t)a.w3 << 32 | a.w2; }
NOID_HD W4 tw4(uint64_t e0, uint64_t e1) { return w4((uint32_t)e0, (uint32_t)(e0 >> 32), (uint32_t)e1, (uint32_t)(e1 >> 32)); }

// (lo + hi x^64) mod x^64 + x^4 + x^3 + x + 1
NOID_HD uint64_t tred(uint64_t lo, uint64_t hi)
{
    const uint64_t f = hi ^ (hi >> 63) ^ (hi >> 61) ^ (hi >> 60);
    return lo ^ f ^ (f << 1) ^ (f << 3) ^ (f << 4);
}
#define NOID_T64_XINV 0x800000000000000Dull  // x^-1

// (a0 + a1 z)(b0 + b1 z) = (P0 + w P1) + (Pm + P0) z with P0 = a0 b0, P1 = a1 b1, Pm = (a0 + a1)(b0 + b1)
NOID_HD W4 gmult(W4 a, W4 b)
{
    const uint64_t a0 = te0(a), a1 = te1(a), b0 = te0(b), b1 = te1(b);
    const uint64_t p0l = tclo(a0, b0, 0), p0h = tchi(a0, b0, 0);
    const uint64_t p1l = tclo(a1, b1, 0), p1h = tchi(a1, b1, 0);
    const uint64_t c1l = tclo(a0 ^ a1, b0 ^ b1, p0l), c1h = tchi(a0 ^ a1, b0 ^ b1, p0h);
    // w P1 = (P1 >> 1) + P1_0 x^-1 (P1 has degree <= 126, so the result fits in 128 bits)
    const uint64_t m = 0 - (p1l & 1);
    const uint64_t c0l = p0l ^ (p1l >> 1 | p1h << 63) ^ (m & NOID_T64_XINV), c0h = p0h ^ (p1h >> 1);
    return tw4(tred(c0l, c0h), tred(c1l, c1h));
}

// (a0 + a1 z)^2 = (a0^2 + w a1^2) + a1^2 z
NOID_HD W4 gsqrt(W4 a)
{
    const uint64_t s0l = spread32(a.w0), s0h = spread32(a.w1), s1l = spread32(a.w2), s1h = spread32(a.w3);
    const uint64_t m = 0 - (uint64_t)(a.w2 & 1u);
    const uint64_t c0l = s0l ^ (s1l >> 1 | s1h << 63) ^ (m & NOID_T64_XINV), c0h = s0h ^ (s1h >> 1);
    return tw4(tred(c0l, c0h), tred(s1l, s1h));
}

NOID_HD W4 pow7t(W4 x)
{
    W4 sq = x, x2 = x;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        sq = gsqrt(sq);
        if (k == 0) x2 = sq;
    }
    W4 a = x, b = x2;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        a = gmult(a, b);
        b = sq;
    }
    return a;
}

// Nibble tables: 64-bit entry (c, k, v) at word 2 * ((c * 16 + k) * 16 + v) = (v x^(4k)) * const_c in L.
// NOID_T64_TBL_WORDS words = 12 KB; one constant's tables are 2 KB (16 x 128-byte tables).
#define NOID_T64_TBL_WORDS (NOID_NCONST * 16 * 16 * 2)
#ifndef NOID_T64_CMCL
#define NOID_T64_CMCL 0x04
#endif
struct CMulNib64 {
    const uint32_t* tbl;
    NOID_HDM uint64_t m1(uint32_t base, uint64_t e) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // byte offset of entry (c, k, v) = c * 2048 + k * 128 + v * 8: v * 8 of the 8 nibbles of a word comes from
        // two masked copies (even nibbles << 3, odd nibbles >> 1); a PRMT merges the selected byte with the
        // constant's base (bytes 1-3 of `base`), so a lookup is 1 PRMT + 1 LDS.64 with an immediate offset.
        const char* t = reinterpret_cast<const char*>(tbl);
        uint64_t acc = 0;
        const uint32_t w[2] = { (uint32_t)e, (uint32_t)(e >> 32) };
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint32_t mlo = (w[i] << 3) & 0x78787878u;
            const uint32_t mhi = (w[i] >> 1) & 0x78787878u;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t alo = prmt(mlo, base, 0x7650u | (uint32_t)j);
                const uint32_t ahi = prmt(mhi, base, 0x7650u | (uint32_t)j);
                const int k = 8 * i + 2 * j;
                acc ^= *reinterpret_cast<const uint64_t*>(t + alo + k * 128);
                acc ^= *reinterpret_cast<const uint64_t*>(t + ahi + (k + 1) * 128);
            }
        }
        return acc;
#else
        uint64_t acc = 0;
        for (int k = 0; k < 16; k++) {
            const uint32_t* p = tbl + base / 4 + 2 * (k * 16 + ((e >> (4 * k)) & 15u));
            acc ^= (uint64_t)p[1] << 32 | p[0];
        }
        return acc;
#endif
    }
    NOID_HDM W4 mul(int c, W4 a) const
    {
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 1
        return w4(a.w1 ^ (uint32_t)c, a.w2, a.w3 ^ a.w0, a.w0 + 3u);
#endif
        const uint32_t base = (uint32_t)c * 2048u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Constants whose bit is set in NOID_T64_CMCL multiply on the CLMAD pipe instead of the tables (entry
        // (c, 0, 1) is the constant itself): it balances the ALU (table addresses and XORs) against CLMAD. RTX 3060
        // at 99 W: none 18.34, d0+1 (bit 2) 18.82, g and h 18.75, d0+1 and d1+1 17.27 MH/s.
        if ((NOID_T64_CMCL >> c) & 1) {
            const uint64_t k = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const char*>(tbl) + base + 8);
            const uint64_t e0 = te0(a), e1 = te1(a);
            return tw4(tred(tclo(e0, k, 0), tchi(e0, k, 0)), tred(tclo(e1, k, 0), tchi(e1, k, 0)));
        }
#endif
        return tw4(m1(base, te0(a)), m1(base, te1(a)));
    }
    NOID_HDM static W4 sbox(W4 x) { return pow7t(x); }
};
