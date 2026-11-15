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
//
// On the RTX 3060 three pipes are close to full at once: CLMAD (FP64 pipe, 2 per SM per clock, 3240 per nonce for
// the 540 general multiplies), shared-memory loads (804 constant multiplies x 32 LDS.64) and, the limit, the integer
// ALU (table addresses and XORs, reductions, squaring spreads). permute_t64 keeps the ALU work low: see there.
#pragma once
#include "noid_comp.h"
#include "noid_perm.h"  // NOID_UNROLL1

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

// (a0 + a1 z)(b0 + b1 z) = (P0 + w P1) + (Pm + P0) z with P0 = a0 b0, P1 = a1 b1, Pm = (a0 + a1)(b0 + b1).
// w P1 is computed as (w a1) b1, with w a1 = (a1 >> 1) + a1_0 x^-1 = (a1 rotated right by 1) + a1_0 (x^3 + x^2 + 1),
// so both products take P0 as the CLMAD addend (RTX 3060: +1% against shifting the 128-bit P1).
NOID_HD W4 gmult(W4 a, W4 b)
{
    const uint64_t a0 = te0(a), a1 = te1(a), b0 = te0(b), b1 = te1(b);
    const uint64_t p0l = tclo(a0, b0, 0), p0h = tchi(a0, b0, 0);
    const uint64_t wa1 = (a1 >> 1 | a1 << 63) ^ ((0 - (a1 & 1)) & 0xDull);
    const uint64_t c0l = tclo(wa1, b1, p0l), c0h = tchi(wa1, b1, p0h);
    const uint64_t c1l = tclo(a0 ^ a1, b0 ^ b1, p0l), c1h = tchi(a0 ^ a1, b0 ^ b1, p0h);
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

// x^7 = (x x^2) x^4. The two squarings are unrolled (RTX 3060 +1.3%); the two multiplies share one instance (an
// unrolled pair is no faster, and the instruction cache is a concern with permute_t64 inlined three times).
NOID_HD W4 pow7t(W4 x)
{
    const W4 x2 = gsqrt(x), x4 = gsqrt(x2);
    W4 a = x, b = x2;
    NOID_MATH_UNROLL1
    for (int k = 0; k < 2; k++) {
        a = gmult(a, b);
        b = x4;
    }
    return a;
}

// Nibble tables: 64-bit entry (c, k, v) at word 2 * ((c * 16 + k) * 16 + v) = (v x^(4k)) * const_c in L.
// NOID_T64_TBL_WORDS words = 12 KB; one constant's tables are 2 KB (16 x 128-byte tables).
#define NOID_T64_TBL_WORDS (NOID_NCONST * 16 * 16 * 2)
// Constants whose bit is set multiply on the CLMAD pipe instead of the tables (entry (c, 0, 1) is the constant
// itself). CLMAD has ~25% headroom left by the general multiplies, but a CLMAD constant multiply needs two
// reductions on the ALU, so only h (the full rounds' x0 + x1 and x2 + x3) uses it. RTX 3060 at 99 W with permute_t64:
// none 20.53, h 20.61, g and h 20.1, d0+1 19.4 MH/s.
#ifndef NOID_T64_CMCL
#define NOID_T64_CMCL 0x02
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
    // acc + e * const (one coefficient) with the table base known at compile time: the selected byte only needs
    // zero-extending (PRMT with zero; byte 3 is a shift the compiler merges with the address), and BASE goes into
    // the LDS immediate. Starting from acc saves the XOR that would add the product to it.
    template <uint32_t BASE>
    NOID_HDM uint64_t m1c(uint64_t e, uint64_t acc) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const char* t = reinterpret_cast<const char*>(tbl) + BASE;
        const uint32_t w[2] = { (uint32_t)e, (uint32_t)(e >> 32) };
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint32_t mlo = (w[i] << 3) & 0x78787878u;
            const uint32_t mhi = (w[i] >> 1) & 0x78787878u;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t alo = j == 3 ? mlo >> 24 : prmt(mlo, 0u, 0x4440u | (uint32_t)j);
                const uint32_t ahi = j == 3 ? mhi >> 24 : prmt(mhi, 0u, 0x4440u | (uint32_t)j);
                const int k = 8 * i + 2 * j;
                acc ^= *reinterpret_cast<const uint64_t*>(t + alo + k * 128);
                acc ^= *reinterpret_cast<const uint64_t*>(t + ahi + (k + 1) * 128);
            }
        }
        return acc;
#else
        return acc ^ m1(BASE, e);
#endif
    }
    NOID_HDM W4 mul(int c, W4 a) const
    {
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 1
        return w4(a.w1 ^ (uint32_t)c, a.w2, a.w3 ^ a.w0, a.w0 + 3u);
#endif
        const uint32_t base = (uint32_t)c * 2048u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if ((NOID_T64_CMCL >> c) & 1) {
            const uint64_t k = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const char*>(tbl) + base + 8);
            const uint64_t e0 = te0(a), e1 = te1(a);
            return tw4(tred(tclo(e0, k, 0), tchi(e0, k, 0)), tred(tclo(e1, k, 0), tchi(e1, k, 0)));
        }
#endif
        return tw4(m1(base, te0(a)), m1(base, te1(a)));
    }
    // s + a * const[C] with C known at compile time (permute_t64): no branch, the table base is an immediate
    template <int C>
    NOID_HDM W4 mulcx(W4 a, W4 s) const
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if ((NOID_T64_CMCL >> C) & 1) {
            const uint64_t k = *reinterpret_cast<const uint64_t*>(reinterpret_cast<const char*>(tbl) + C * 2048 + 8);
            const uint64_t e0 = te0(a), e1 = te1(a);
            return tw4(tred(tclo(e0, k, te0(s)), tchi(e0, k, 0)), tred(tclo(e1, k, te1(s)), tchi(e1, k, 0)));
        }
#endif
        return tw4(m1c<C * 2048u>(te0(a), te0(s)), m1c<C * 2048u>(te1(a), te1(s)));
    }
    NOID_HDM static W4 sbox(W4 x) { return pow7t(x); }
};

// full MDS layer: A = h(x0+x1), B = h(x2+x3), C = g x1, D = g x3 (constants 1, 1, 0, 0)
NOID_HD void mds_full_t64(W4 s[4], const CMulNib64& cm)
{
    const W4 z = w4(0u, 0u, 0u, 0u);
    const W4 x01 = w4x(s[0], s[1]), x23 = w4x(s[2], s[3]), S = w4x(x01, x23);
    const W4 A = cm.mulcx<1>(x01, z), B = cm.mulcx<1>(x23, z), C = cm.mulcx<0>(s[1], z), D = cm.mulcx<0>(s[3], z);
    const W4 CD = w4x(C, D);
    s[0] = w4x3(A, CD, S);
    s[1] = w4x3(A, C, x23);
    s[2] = w4x3(B, CD, S);
    s[3] = w4x3(B, D, x01);
}

// The permutation (same rounds as permute() of noid_perm.h): round -1 is the initial full MDS layer, rounds 0..3
// and 62..65 are full (the four S-boxes in a loop with one instance), rounds 4..61 are partial:
//   y = (s0 + rc)^7, S = y + s1 + s2 + s3, s0 = S + (d0+1) y, s_i = S + (d_i+1) s_i (constants 2..5).
// Every MDS layer is unrolled with compile-time constants, and each partial-round product starts from S: no lane
// rotation, no per-constant branch, table bases as LDS immediates. RTX 3060 at 99 W: 18.8 -> 20.9 MH/s (with
// gmult's CLMAD addend and the unrolled squarings).
template <class RC>
NOID_HD void permute_t64(W4 s[4], const CMulNib64& cm, RC& rc)
{
    mds_full_t64(s, cm);
    NOID_UNROLL1
    for (int q = 0; q < 8; q++) {
        const int r = q < 4 ? q : q + 58;
        NOID_UNROLL1
        for (int i = 0; i < 4; i++) {
            W4 y = pow7t(w4x(s[0], rc.get(i, r)));
            s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = y;
        }
        mds_full_t64(s, cm);
        if (q == 3) {
            NOID_UNROLL1
            for (int rp = 4; rp < 62; rp++) {
                const W4 y = pow7t(w4x(s[0], rc.get(0, rp)));
                const W4 S = w4x(w4x(y, s[1]), w4x(s[2], s[3]));
                s[0] = cm.mulcx<2>(y, S); s[1] = cm.mulcx<3>(s[1], S); s[2] = cm.mulcx<4>(s[2], S);
                s[3] = cm.mulcx<5>(s[3], S);
            }
        }
    }
}
