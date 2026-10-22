// Ported for X-Miner from gacanga/NoidMiner gpu/noid_perm.h (commit 31fa4a8), Apache License 2.0,
// Copyright 2026 gacanga. X-Miner keeps the permutation core and the pre-split constant-multiplier policy; the
// shared-table and tower engines are not included. See NOTICE and THIRD_PARTY.md.
// noid_perm.h - Poseidon2b permutation (flat basis) built on noid_math.h.
//
// Templated on a constant-multiplier policy CM providing
//     W4 CM::mul(int c, W4 a)     // a * const[c], c in 0..NOID_NCONST-1
// and on a round-constant accessor RC providing
//     W4 RC::get(int lane, int round)
// so that the same code runs on the GPU (shared tables / constant memory)
// and on the CPU (validation).
//
// Code size matters on the GPU (instruction cache): the permutation is written
// so that there is a single instance of the S-box, of the 128-bit multiply and
// of the squaring; full rounds loop over the 4 lanes with a register rotation.
#pragma once
#include "noid_math.h"

#if defined(__CUDACC__) || defined(__HIP__)
#define NOID_PRAGMA(x) _Pragma(#x)
#define NOID_UNROLL1 NOID_PRAGMA(unroll 1)
#define NOID_UNROLL NOID_PRAGMA(unroll)
#else
#define NOID_UNROLL1
#define NOID_UNROLL
#endif

// x^7 (pow7 in noid_math.h)
// NOID_EXPERIMENT (profiling builds only, results are WRONG):
//   1 = constant multiplications skipped (arithmetic only)
//   2 = S-box arithmetic replaced by XORs (table lookups only)
NOID_HD W4 sbox7(W4 x)
{
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 2
    return w4(x.w1 ^ 0x9e3779b9u, x.w2 ^ x.w0, x.w3 + 7u, x.w0 ^ x.w3);
#endif
    return pow7(x);
}

// MDS layer with a single constant-multiplier instance (4 iterations, rotation)
//   full   : A = h(x0+x1)  B = h(x2+x3)  C = g x1  D = g x3   (consts 1,1,0,0)
//   partial: P_i = (d_i + 1) x_i                              (consts 2,3,4,5)
template <class CM>
NOID_HD void mds(W4 s[4], bool full, CM& cm)
{
    W4 x01 = w4x(s[0], s[1]);
    W4 x23 = w4x(s[2], s[3]);
    W4 S = w4x(x01, x23);
    W4 u0, u1, u2, u3;
    if (full) { u0 = x01; u1 = x23; u2 = s[1]; u3 = s[3]; }
    else      { u0 = s[0]; u1 = s[1]; u2 = s[2]; u3 = s[3]; }
    NOID_UNROLL1
    for (int i = 0; i < 4; i++) {
        int c = full ? (i < 2 ? 1 : 0) : 2 + i;
        W4 p = cm.mul(c, u0);
        u0 = u1; u1 = u2; u2 = u3; u3 = p;
    }
    if (full) {
        // u0 = A, u1 = B, u2 = C, u3 = D
        W4 CD = w4x(u2, u3);
        s[0] = w4x3(u0, CD, S);
        s[1] = w4x3(u0, u2, x23);
        s[2] = w4x3(u1, CD, S);
        s[3] = w4x3(u1, u3, x01);
    } else {
        s[0] = w4x(S, u0);
        s[1] = w4x(S, u1);
        s[2] = w4x(S, u2);
        s[3] = w4x(S, u3);
    }
}

// round -1 is the initial MDS_FULL layer; rounds 0..3 and 62..65 are full
template <class CM, class RC>
NOID_HD void permute(W4 s[4], CM& cm, RC& rc)
{
    NOID_UNROLL1
    for (int r = -1; r < 66; r++) {
        const bool full = (r < 4) || (r >= 62);
        const int lanes = r < 0 ? 0 : (full ? 4 : 1);
        NOID_UNROLL1
        for (int i = 0; i < lanes; i++) {
            W4 y = sbox7(w4x(s[0], rc.get(i, r)));
            if (full) {               // rotate: after 4 steps lanes are back in place
                s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = y;
            } else {
                s[0] = y;
            }
        }
        mds(s, full, cm);
    }
}

// ---------------------------------------------------------------------------
// N independent permutations interleaved (instruction-level parallelism).
// Same structure as permute(): one S-box / multiply / squaring / constant
// multiply instance, each doing the N lanes back to back so the GPU always has
// independent arithmetic to issue while table loads are in flight.
// ---------------------------------------------------------------------------
template <int N>
NOID_HD void sbox7_n(W4 x[N])
{
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 2
    NOID_UNROLL
    for (int n = 0; n < N; n++) x[n] = w4(x[n].w1 ^ 0x9e3779b9u, x[n].w2 ^ x[n].w0, x[n].w3 + 7u, x[n].w0 ^ x[n].w3);
    return;
#endif
    W4 sq[N], x2[N], a[N], b[N];
    NOID_UNROLL
    for (int n = 0; n < N; n++) { sq[n] = x[n]; x2[n] = x[n]; }
    NOID_UNROLL1
    for (int k = 0; k < 2; k++) {
        NOID_UNROLL
        for (int n = 0; n < N; n++) {
            sq[n] = gsqr(sq[n]);
            if (k == 0) x2[n] = sq[n];
        }
    }
    NOID_UNROLL
    for (int n = 0; n < N; n++) { a[n] = x[n]; b[n] = x2[n]; }
    NOID_UNROLL1
    for (int k = 0; k < 2; k++) {
        NOID_UNROLL
        for (int n = 0; n < N; n++) {
            a[n] = gmul(a[n], b[n]);
            b[n] = sq[n];
        }
    }
    NOID_UNROLL
    for (int n = 0; n < N; n++) x[n] = a[n];
}

template <int N, class CM>
NOID_HD void mds_n(W4 s[N][4], bool full, CM& cm)
{
    W4 x01[N], x23[N], S[N], u0[N], u1[N], u2[N], u3[N];
    NOID_UNROLL
    for (int n = 0; n < N; n++) {
        x01[n] = w4x(s[n][0], s[n][1]);
        x23[n] = w4x(s[n][2], s[n][3]);
        S[n] = w4x(x01[n], x23[n]);
        if (full) { u0[n] = x01[n]; u1[n] = x23[n]; u2[n] = s[n][1]; u3[n] = s[n][3]; }
        else      { u0[n] = s[n][0]; u1[n] = s[n][1]; u2[n] = s[n][2]; u3[n] = s[n][3]; }
    }
    NOID_UNROLL1
    for (int i = 0; i < 4; i++) {
        int c = full ? (i < 2 ? 1 : 0) : 2 + i;
        W4 p[N];
        NOID_UNROLL
        for (int n = 0; n < N; n++) p[n] = cm.mul(c, u0[n]);
        NOID_UNROLL
        for (int n = 0; n < N; n++) { u0[n] = u1[n]; u1[n] = u2[n]; u2[n] = u3[n]; u3[n] = p[n]; }
    }
    NOID_UNROLL
    for (int n = 0; n < N; n++) {
        if (full) {
            W4 CD = w4x(u2[n], u3[n]);
            s[n][0] = w4x3(u0[n], CD, S[n]);
            s[n][1] = w4x3(u0[n], u2[n], x23[n]);
            s[n][2] = w4x3(u1[n], CD, S[n]);
            s[n][3] = w4x3(u1[n], u3[n], x01[n]);
        } else {
            s[n][0] = w4x(S[n], u0[n]);
            s[n][1] = w4x(S[n], u1[n]);
            s[n][2] = w4x(S[n], u2[n]);
            s[n][3] = w4x(S[n], u3[n]);
        }
    }
}

template <int N, class CM, class RC>
NOID_HD void permute_n(W4 s[N][4], CM& cm, RC& rc)
{
    NOID_UNROLL1
    for (int r = -1; r < 66; r++) {
        const bool full = (r < 4) || (r >= 62);
        const int lanes = r < 0 ? 0 : (full ? 4 : 1);
        NOID_UNROLL1
        for (int i = 0; i < lanes; i++) {
            W4 c = rc.get(i, r);
            W4 y[N];
            NOID_UNROLL
            for (int n = 0; n < N; n++) y[n] = w4x(s[n][0], c);
            sbox7_n<N>(y);
            NOID_UNROLL
            for (int n = 0; n < N; n++) {
                if (full) { s[n][0] = s[n][1]; s[n][1] = s[n][2]; s[n][2] = s[n][3]; s[n][3] = y[n]; }
                else      { s[n][0] = y[n]; }
            }
        }
        mds_n<N>(s, full, cm);
    }
}

// ---------------------------------------------------------------------------
// constant multiplier policies
// ---------------------------------------------------------------------------

// pre-split constant operands, multiplication with the generic Karatsuba core
struct CMulKop {
    const KOp* k;   // NOID_NCONST entries
    NOID_HDM W4 mul(int c, W4 a) { KOp ka = kop(a); return kmul(ka, k[c]); }
};

