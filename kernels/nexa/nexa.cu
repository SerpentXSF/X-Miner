// NexaPow search kernel (compiled at run time with NVRTC / hipRTC; the host prepends TPB, W, NW and the constants).
// One thread per nonce:
//   miningHash = SHA256d(commitment | 0x0c | extranonce1 (4) | nonce (8)), h1 = SHA256(miningHash),
//   k = RFC6979(miningHash, h1) (HMAC-SHA256 DRBG, first output), P = x G and R = k G by a signed fixed-base comb
//   (affine table, Jacobian accumulator, no doublings), one inversion per block for the affine x of P and R
//   (Montgomery's trick over the block's 2 * TPB z values), k -> n - k unless R.y is a quadratic residue (Legendre
//   symbol of Y Z: y = Y / Z^3), e = SHA256(R.x | 02/03 P.x | h1) mod n, s = k + e x mod n,
//   powHash = SHA256(R.x | s); candidates with top 64 bits <= the target's are re-checked on the CPU.
// Field elements are 9 limbs of 29 bits (see the field section); scalars mod n are 8 x 32-bit limbs.
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 ror(u32 x, u32 n) { return (x >> n) | (x << (32u - n)); }

__constant__ u32 K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

#define S0(x) (ror(x, 2) ^ ror(x, 13) ^ ror(x, 22))
#define S1(x) (ror(x, 6) ^ ror(x, 11) ^ ror(x, 25))
#define s0(x) (ror(x, 7) ^ ror(x, 18) ^ ((x) >> 3))
#define s1(x) (ror(x, 17) ^ ror(x, 19) ^ ((x) >> 10))
#define CH(e, f, g) ((g) ^ ((e) & ((f) ^ (g))))
#define MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) | (b))))

// One block into st (w is consumed). Rounds in four unrolled groups of 16 (constant indices into w) to keep the
// code size of the ~20 call sites down.
__device__ __forceinline__ void sha_blk(u32 st[8], u32 w[16]) {
  u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int r = 0; r < 64; r += 16) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (r) w[j] += s1(w[(j + 14) & 15]) + w[(j + 9) & 15] + s0(w[(j + 1) & 15]);
      const u32 t1 = h + S1(e) + CH(e, f, g) + K256[r + j] + w[j];
      const u32 t2 = S0(a) + MAJ(a, b, c);
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

__device__ __forceinline__ void sha_iv(u32 st[8]) {
  st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;
  st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;
}

// SHA256 of a 32-byte message (8 words) into out.
__device__ __forceinline__ void sha_32(u32 out[8], const u32 m[8]) {
  u32 w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = m[i];
  w[8] = 0x80000000u;
#pragma unroll
  for (int i = 9; i < 15; ++i) w[i] = 0u;
  w[15] = 256u;
  sha_iv(out);
  sha_blk(out, w);
}

// HMAC-SHA256 key states: inner / outer after the key's ipad / opad block (32-byte key).
__device__ __forceinline__ void hmac_keys(u32 is[8], u32 os[8], const u32 k[8]) {
  u32 w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (i < 8 ? k[i] : 0u) ^ 0x36363636u;
  sha_iv(is);
  sha_blk(is, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (i < 8 ? k[i] : 0u) ^ 0x5c5c5c5cu;
  sha_iv(os);
  sha_blk(os, w);
}

// Outer hash: out = SHA256 from os over the 32-byte inner digest (total length 96 bytes).
__device__ __forceinline__ void hmac_outer(u32 out[8], const u32 os[8], const u32 in[8]) {
  u32 w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { out[i] = os[i]; w[i] = in[i]; }
  w[8] = 0x80000000u;
#pragma unroll
  for (int i = 9; i < 15; ++i) w[i] = 0u;
  w[15] = 768u;
  sha_blk(out, w);
}

// HMAC(key states, 32-byte message).
__device__ __forceinline__ void hmac_32(u32 out[8], const u32 is[8], const u32 os[8], const u32 m[8]) {
  u32 in[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { in[i] = is[i]; w[i] = m[i]; }
  w[8] = 0x80000000u;
#pragma unroll
  for (int i = 9; i < 15; ++i) w[i] = 0u;
  w[15] = 768u;
  sha_blk(in, w);
  hmac_outer(out, os, in);
}

// HMAC(key states, V | sep | seed), seed = miningHash | h1 | "Schnorr+SHA256  " (113 bytes): sh[] is the seed
// stream shifted by one byte (sh[t] = seed[t - 1] << 24 | seed[t] >> 8 for t >= 1, sh[0] = seed[0] >> 8).
__device__ __forceinline__ void hmac_seed(u32 out[8], const u32 is[8], const u32 os[8], const u32 v[8], u32 sep,
                                          const u32 sh[21]) {
  u32 in[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { in[i] = is[i]; w[i] = v[i]; }
  w[8] = sep << 24 | sh[0];
#pragma unroll
  for (int i = 9; i < 16; ++i) w[i] = sh[i - 8];
  sha_blk(in, w);
#pragma unroll
  for (int i = 0; i < 13; ++i) w[i] = sh[8 + i];  // sh[20] carries the 0x80 pad byte
#pragma unroll
  for (int i = 13; i < 15; ++i) w[i] = 0u;
  w[15] = (64u + 113u) * 8u;
  sha_blk(in, w);
  hmac_outer(out, os, in);
}

// ---------------- field mod p = 2^256 - 2^32 - 977: 9 limbs of 29 bits ----------------
// value = sum a_i 2^(29 i). Normalized (m1): limbs < 2^29 + 2^14, limb 8 < 2^24 + 1. A sum of two m1 values (m2,
// limbs < 2^30 + 2^15) may go into fe_mul / fe_sqr: each 64-bit column of 29 x 29-bit products then stays below
// 2^64, so a product is one multiply-add per limb pair and the carries are resolved once. fe_sub(a, b) (a, b <= m2)
// adds 4p, whose limbs exceed b's, and normalizes.
#define M29 0x1FFFFFFFu
__constant__ u32 FP[8] = {0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                          0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
__constant__ u32 P29[9] = {0x1ffffc2fu, 0x1ffffff7u, 0x1fffffffu, 0x1fffffffu, 0x1fffffffu,
                           0x1fffffffu, 0x1fffffffu, 0x1fffffffu, 0x00ffffffu};

// u64 limbs (< 2^63.5) -> m1: carry pass, then the part above 2^256 folded back with 2^256 = 2^32 + 977.
__device__ __forceinline__ void fe_norm64(u32 r[9], u64 c[9]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c[i + 1] += c[i] >> 29;
    c[i] &= M29;
  }
  const u64 t = c[8] >> 24;
  c[8] &= 0xFFFFFFu;
  c[0] += t * 977u;
  c[1] += t << 3;  // 2^32 = 2^29 2^3
  c[1] += c[0] >> 29;
  c[0] &= M29;
  c[2] += c[1] >> 29;
  c[1] &= M29;
#pragma unroll
  for (int i = 0; i < 9; ++i) r[i] = (u32)c[i];
}

// 17 product columns -> m1. High columns become 29-bit digits; 2^261 = 2^37 + 31264 (mod p), so digit 9 + k adds
// 31264 d to column k and d << 8 to column k + 1.
#if defined(__HIPCC__) && !defined(RED_INLINE)
// AMD: inlined at the squaring sites, the reduction is miscompiled at -O1 and above (gfx1201: results depend on other
// lanes' data); kept out of line there.
__device__ __attribute__((noinline)) void fe_red(u32 r[9], u64 c[17]) {
#else
__device__ __forceinline__ void fe_red(u32 r[9], u64 c[17]) {
#endif
#pragma unroll
  for (int k = 9; k < 16; ++k) {
    c[k + 1] += c[k] >> 29;
    c[k] &= M29;
  }
  const u64 d17 = c[16] >> 29;
  c[16] &= M29;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    c[k] += c[9 + k] * 31264u;
    c[k + 1] += c[9 + k] << 8;
  }
  c[8] += d17 * 31264u;
  c[0] += (d17 << 8) * 31264u;  // d17 << 8 sits at 2^261 again
  c[1] += d17 << 16;
  fe_norm64(r, c);
}

__device__ __forceinline__ void fe_mul(u32 r[9], const u32 a[9], const u32 b[9]) {
  u64 c[17];
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
#pragma unroll
    for (int j = 0; j < 9; ++j) c[i + j] += (u64)a[i] * b[j];
  }
  fe_red(r, c);
}

__device__ __forceinline__ void fe_sqr(u32 r[9], const u32 a[9]) {
  // cross products a_i a_j (i < j) per column, doubled, plus the squares
  u64 c[17];
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] = 0;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
#pragma unroll
    for (int j = i + 1; j < 9; ++j) c[i + j] += (u64)a[i] * a[j];
  }
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] <<= 1;
#pragma unroll
  for (int i = 0; i < 9; ++i) c[2 * i] += (u64)a[i] * a[i];
  fe_red(r, c);
}

__device__ __forceinline__ void fe_add(u32 r[9], const u32 a[9], const u32 b[9]) {
#pragma unroll
  for (int i = 0; i < 9; ++i) r[i] = a[i] + b[i];
}

// u32 limbs (< 2^32) -> m1
__device__ __forceinline__ void fe_norm(u32 r[9]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    r[i + 1] += r[i] >> 29;
    r[i] &= M29;
  }
  const u32 t = r[8] >> 24;
  r[8] &= 0xFFFFFFu;
  r[0] += t * 977u;
  r[1] += t << 3;
  r[1] += r[0] >> 29;
  r[0] &= M29;
  r[2] += r[1] >> 29;
  r[1] &= M29;
}

__device__ __forceinline__ void fe_sub(u32 r[9], const u32 a[9], const u32 b[9]) {
#pragma unroll
  for (int i = 0; i < 9; ++i) r[i] = a[i] + (P29[i] << 2) - b[i];
  fe_norm(r);
}

// canonical (< p, limbs < 2^29)
__device__ __forceinline__ void fe_canon(u32 r[9]) {
  fe_norm(r);
  fe_norm(r);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    r[i + 1] += r[i] >> 29;
    r[i] &= M29;
  }
  bool ge = true;
#pragma unroll
  for (int i = 8; i >= 0; --i) {
    if (r[i] != P29[i]) { ge = r[i] > P29[i]; break; }
  }
  if (ge) {
    int borrow = 0;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const int v = (int)r[i] - (int)P29[i] - borrow;
      borrow = v < 0;
      r[i] = (u32)v & M29;
    }
  }
}

__device__ __forceinline__ void fe_copy(u32 r[9], const u32 a[9]) {
#pragma unroll
  for (int i = 0; i < 9; ++i) r[i] = a[i];
}

// 8 x 32-bit little-endian limbs <-> 9 x 29 (the latter canonical)
__device__ __forceinline__ void fe_from32(u32 r[9], const u32 w[8]) {
  r[0] = w[0] & M29;
  r[1] = (w[0] >> 29 | w[1] << 3) & M29;
  r[2] = (w[1] >> 26 | w[2] << 6) & M29;
  r[3] = (w[2] >> 23 | w[3] << 9) & M29;
  r[4] = (w[3] >> 20 | w[4] << 12) & M29;
  r[5] = (w[4] >> 17 | w[5] << 15) & M29;
  r[6] = (w[5] >> 14 | w[6] << 18) & M29;
  r[7] = (w[6] >> 11 | w[7] << 21) & M29;
  r[8] = w[7] >> 8;
}
__device__ __forceinline__ void fe_to32(u32 w[8], const u32 r[9]) {
  w[0] = r[0] | r[1] << 29;
  w[1] = r[1] >> 3 | r[2] << 26;
  w[2] = r[2] >> 6 | r[3] << 23;
  w[3] = r[3] >> 9 | r[4] << 20;
  w[4] = r[4] >> 12 | r[5] << 17;
  w[5] = r[5] >> 15 | r[6] << 14;
  w[6] = r[6] >> 18 | r[7] << 11;
  w[7] = r[7] >> 21 | r[8] << 8;
}

__device__ __forceinline__ void fe_sqr_n(u32 r[9], int n) {
#pragma unroll 1
  for (int i = 0; i < n; ++i) fe_sqr(r, r);
}

// a^(p - 2) (mode 0, inverse) or a^((p - 1) / 2) (mode 1, Legendre symbol): x_k = a^(2^k - 1) blocks up to x223,
// then the tails (checked against the exponents with Python).
__device__ __forceinline__ void fe_pow_chain(u32 r[9], const u32 a_in[9], int mode) {
  u32 a[9], x2[9], x3[9], x22[9], t[9], u[9];
  fe_copy(a, a_in);  // r may alias a_in
  fe_sqr(x2, a); fe_mul(x2, x2, a);                    // x2
  fe_sqr(x3, x2); fe_mul(x3, x3, a);                   // x3
  fe_copy(t, x3); fe_sqr_n(t, 3); fe_mul(t, t, x3);    // x6
  fe_sqr_n(t, 3); fe_mul(t, t, x3);                    // x9
  fe_sqr_n(t, 2); fe_mul(t, t, x2);                    // x11
  fe_copy(u, t); fe_sqr_n(u, 11); fe_mul(x22, u, t);   // x22
  fe_copy(t, x22); fe_sqr_n(t, 22); fe_mul(t, t, x22);  // x44
  fe_copy(u, t); fe_sqr_n(u, 44); fe_mul(u, u, t);     // x88
  fe_copy(r, u); fe_sqr_n(r, 88); fe_mul(r, r, u);     // x176
  fe_sqr_n(r, 44); fe_mul(r, r, t);                    // x220
  fe_sqr_n(r, 3); fe_mul(r, r, x3);                    // x223
  fe_sqr_n(r, 23); fe_mul(r, r, x22);
  if (mode == 0) {
    fe_sqr_n(r, 5); fe_mul(r, r, a);
    fe_sqr_n(r, 3); fe_mul(r, r, x2);
    fe_sqr_n(r, 2); fe_mul(r, r, a);
  } else {
    fe_sqr_n(r, 5); fe_mul(r, r, a);
    fe_sqr_n(r, 1);
    fe_sqr_n(r, 3); fe_mul(r, r, x3);
  }
}

// ---------------- scalars mod n ----------------
__constant__ u32 FN[8] = {0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u,
                          0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
__constant__ u32 FNC[5] = {0x2FC9BEBFu, 0x402DA173u, 0x50B75FC4u, 0x45512319u, 1u};  // 2^256 - n

__device__ __forceinline__ bool sc_ge_n(const u32 a[8]) {
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    if (a[i] != FN[i]) return a[i] > FN[i];
  }
  return true;
}

__device__ __forceinline__ void sc_sub_n(u32 a[8]) {
  long long c = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c += (long long)a[i] - (long long)FN[i];
    a[i] = (u32)c;
    c >>= 32;
  }
}

// r = n - a (a < n, a != 0)
__device__ __forceinline__ void sc_neg(u32 r[8], const u32 a[8]) {
  long long c = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c += (long long)FN[i] - (long long)a[i];
    r[i] = (u32)c;
    c >>= 32;
  }
}

// r = (a + b * c) mod n for a, b, c < n (once per nonce: plain loops).
__device__ __forceinline__ void sc_muladd(u32 r[8], const u32 a[8], const u32 b[8], const u32 c[8]) {
  u32 t[17];
#pragma unroll
  for (int i = 0; i < 17; ++i) t[i] = i < 8 ? a[i] : 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    u64 carry = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      carry += (u64)b[i] * c[j] + t[i + j];
      t[i + j] = (u32)carry;
      carry >>= 32;
    }
#pragma unroll
    for (int k = i + 8; carry && k < 17; ++k) {
      carry += t[k];
      t[k] = (u32)carry;
      carry >>= 32;
    }
  }
  // fold the part above 2^256 with 2^256 = FNC (mod n) until it is gone
#pragma unroll
  for (int round = 0; round < 4; ++round) {
    u32 h[9];
    bool any = false;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      h[i] = t[8 + i];
      any |= h[i] != 0u;
      t[8 + i] = 0u;
    }
    if (!any) break;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      if (!h[i]) continue;
      u64 carry = 0;
#pragma unroll
      for (int j = 0; j < 5; ++j) {
        carry += (u64)h[i] * FNC[j] + t[i + j];
        t[i + j] = (u32)carry;
        carry >>= 32;
      }
#pragma unroll
      for (int k = i + 5; carry && k < 17; ++k) {
        carry += t[k];
        t[k] = (u32)carry;
        carry >>= 32;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = t[i];
  if (sc_ge_n(r)) sc_sub_n(r);
  if (sc_ge_n(r)) sc_sub_n(r);
}

// ---------------- points ----------------
// (X, Y, Z) += (x2, y2) affine (madd-2007-bl, a = 0); X, Y m1, Z m2 in and out. The equal / opposite point cases
// are not handled (a random scalar meets them with probability ~2^-250; a wrong result only loses that nonce: the
// CPU re-checks candidates).
__device__ __forceinline__ void madd(u32 X[9], u32 Y[9], u32 Z[9], const u32 x2[9], const u32 y2[9]) {
  u32 t0[9], t1[9], t2[9], t3[9], t4[9];  // five temporaries (register pressure)
  fe_sqr(t0, Z);         // Z1Z1
  fe_mul(t1, x2, t0);    // U2
  fe_mul(t2, y2, Z);
  fe_mul(t2, t2, t0);    // S2
  fe_sub(t1, t1, X);     // H
  fe_sub(t2, t2, Y);
  fe_add(t2, t2, t2);    // r = 2 (S2 - Y1), m2
  fe_sqr(t0, t1);        // HH
  fe_add(t0, t0, t0);    // 2 HH, m2
  fe_mul(t3, t1, t0);
  fe_add(t3, t3, t3);    // J = 4 H HH, m2
  fe_mul(t4, X, t0);
  fe_add(t4, t4, t4);    // V = 4 X1 HH, m2
  fe_mul(Z, Z, t1);
  fe_add(Z, Z, Z);       // Z3 = 2 Z1 H, m2
  fe_sqr(t0, t2);
  fe_sub(t0, t0, t3);
  fe_sub(t0, t0, t4);
  fe_sub(X, t0, t4);     // X3 = r^2 - J - 2 V
  fe_sub(t1, t4, X);
  fe_mul(t1, t2, t1);    // r (V - X3)
  fe_mul(t3, Y, t3);
  fe_add(t3, t3, t3);    // 2 Y1 J, m2
  fe_sub(Y, t1, t3);
}

// k G with the signed W-bit comb: digit i (window i plus the carry from below) in [-2^(W-1) + 1, 2^(W-1)] picks
// table entry |d| of window i (8 x 32-bit affine x, y), negated for d < 0. NW windows cover 256 bits plus the last
// carry.
__device__ __forceinline__ void comb_mul(u32 X[9], u32 Y[9], u32 Z[9], const u32 k[8], const uint4* __restrict__ tab) {
  int carry = 0;
  bool first = true;
  u32 ks[8];  // the scalar, shifted down by W bits per window (no dynamically indexed arrays)
#pragma unroll
  for (int m = 0; m < 9; ++m) { X[m] = 0u; Y[m] = 0u; Z[m] = m ? 0u : 1u; }
#pragma unroll
  for (int m = 0; m < 8; ++m) ks[m] = k[m];
#pragma unroll 1
  for (int i = 0; i < NW; ++i) {
    const u32 win = ks[0] & ((1u << W) - 1u);
#pragma unroll
    for (int m = 0; m < 7; ++m) ks[m] = (ks[m] >> W) | (ks[m + 1] << (32 - W));
    ks[7] >>= W;
    int d = (int)win + carry;
    carry = 0;
    if (d > (1 << (W - 1))) {
      d -= 1 << W;
      carry = 1;
    }
    const int mag = d < 0 ? -d : (d == 0 ? 1 : d);
    const uint4* q = tab + 4 * (i * (1 << (W - 1)) + mag - 1);
    const uint4 q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    const u32 px[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const u32 py0[8] = {q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
    u32 py[8];
    {
      long long c = 0;
      const bool neg = d < 0;
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        c += (long long)FP[m] - (long long)py0[m];
        py[m] = neg ? (u32)c : py0[m];
        c >>= 32;
      }
    }
    u32 x2[9], y2[9];
    fe_from32(x2, px);
    fe_from32(y2, py);
    if (d != 0) {
      if (first) {
        fe_copy(X, x2);
        fe_copy(Y, y2);
        first = false;
      } else {
        madd(X, Y, Z, x2, y2);
      }
    }
  }
}

// Legendre symbol (a / p) = 1 for a canonical, nonzero a: the binary Jacobi algorithm, one bit per step, branch
// free. (a, b) = (a, p); a odd: if a < b swap to (b - a, a) (sign flips when a = b = 3 mod 4), else a -= b; then
// a /= 2 (sign flips when b = 3 or 5 mod 8). Ends at a = 0 with b = gcd = 1.
__device__ __forceinline__ bool is_qr(const u32 a_in[8]) {
  u32 a[8], b[8], s = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) { a[i] = a_in[i]; b[i] = FP[i]; }
#pragma unroll 1
  for (;;) {
    u32 any = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) any |= a[i];
    if (!any) break;
#pragma unroll 1
    for (int rep = 0; rep < 4; ++rep) {  // four steps per zero check (extra steps on a = 0 change nothing)
      const u32 odd = a[0] & 1u;
      u32 d[8], nd[8];
      long long c = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c += (long long)a[i] - (long long)b[i];
        d[i] = (u32)c;
        c >>= 32;
      }
      const u32 br = (u32)c & 1u;  // a < b
      c = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c -= (long long)d[i];
        nd[i] = (u32)c;
        c >>= 32;
      }
      const u32 sw = odd & br;
      s ^= sw & (a[0] & b[0]) >> 1;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const u32 na = odd ? (br ? nd[i] : d[i]) : a[i];
        b[i] = sw ? a[i] : b[i];
        a[i] = na;
      }
      s ^= ((b[0] >> 1) ^ (b[0] >> 2)) & 1u;
#pragma unroll
      for (int i = 0; i < 7; ++i) a[i] = (a[i] >> 1) | (a[i + 1] << 31);
      a[7] >>= 1;
    }
  }
  return (s & 1u) == 0u;
}

// big-endian SHA words <-> little-endian 32-bit limbs
__device__ __forceinline__ void be_to_limbs(u32 r[8], const u32 w[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = w[7 - i];
}
__device__ __forceinline__ void limbs_to_be(u32 w[8], const u32 r[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = r[7 - i];
}
// canonical field element -> big-endian words
__device__ __forceinline__ void fe_to_be(u32 w[8], u32 a[9]) {
  u32 l[8];
  fe_canon(a);
  fe_to32(l, a);
  limbs_to_be(w, l);
}

struct Params {
  u32 cw[8];        // commitment words (big-endian), message words 0..7
  u32 r8[8];        // SHA-256 working state after rounds 0..7 of the first block
  u32 is0[8], os0[8];  // HMAC key states for the all-zero first DRBG key
  u32 w8;           // 0x0c << 24 | extranonce1[0..2]
  u32 en3;          // extranonce1[3]
  u64 base;         // first nonce of the launch
  u64 target_hi;    // top 64 bits of the share target
  u32 dump;         // debug: write powHash and the signature of every nonce
  u32 pad;
};

#ifndef MINB
#define MINB 1
#endif
extern "C" __global__ __launch_bounds__(TPB, MINB) void xm_nexa(Params p, const uint4* __restrict__ tab,
                                                                u32* __restrict__ found, u32 maxf,
                                                                u32* __restrict__ dbg) {
  __shared__ u32 sh_pre[TPB][9];
  __shared__ u32 sh_suf[TPB][9];
  __shared__ u32 sh_inv[9];
  const u32 tid = threadIdx.x;
  const u64 nonce = p.base + (u64)blockIdx.x * TPB + tid;

  // ---- miningHash = SHA256d(commitment | 0x0c | en1 | nonce): first block from the round-8 state ----
  u32 mh[8], h1[8];
  {
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = p.cw[i];
    const u32 b0 = (u32)nonce, b1 = (u32)(nonce >> 32);  // nonce bytes little-endian
    w[8] = p.w8;
    w[9] = p.en3 << 24 | (b0 & 0xffu) << 16 | ((b0 >> 8) & 0xffu) << 8 | ((b0 >> 16) & 0xffu);
    w[10] = (b0 >> 24) << 24 | (b1 & 0xffu) << 16 | ((b1 >> 8) & 0xffu) << 8 | ((b1 >> 16) & 0xffu);
    w[11] = (b1 >> 24) << 24 | 0x00800000u;
    w[12] = 0u; w[13] = 0u; w[14] = 0u;
    w[15] = 45u * 8u;
    // rounds 8..63 from the precomputed state (rounds 0..7 only read w0..w7)
    u32 a = p.r8[0], b = p.r8[1], c = p.r8[2], d = p.r8[3], e = p.r8[4], f = p.r8[5], g = p.r8[6], h = p.r8[7];
#pragma unroll
    for (int j = 8; j < 16; ++j) {
      const u32 t1 = h + S1(e) + CH(e, f, g) + K256[j] + w[j];
      const u32 t2 = S0(a) + MAJ(a, b, c);
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
#pragma unroll 1
    for (int r = 16; r < 64; r += 16) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        w[j] += s1(w[(j + 14) & 15]) + w[(j + 9) & 15] + s0(w[(j + 1) & 15]);
        const u32 t1 = h + S1(e) + CH(e, f, g) + K256[r + j] + w[j];
        const u32 t2 = S0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
      }
    }
    u32 s[8];
    sha_iv(s);
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
    sha_32(mh, s);
  }
  sha_32(h1, mh);

  // ---- k = RFC6979(miningHash, h1): K1 = HMAC(0, V0|00|seed), V1 = HMAC(K1, V0), K2 = HMAC(K1, V1|01|seed),
  //      V2 = HMAC(K2, V1), k = V3 = HMAC(K2, V2) ----
  u32 kk[8];
  {
    u32 sh[21];
    // seed = mh | h1 | "Schnorr+SHA256  " as one byte-shifted stream (then the 0x80 pad byte)
    u32 seed[20];
#pragma unroll
    for (int i = 0; i < 8; ++i) { seed[i] = mh[i]; seed[8 + i] = h1[i]; }
    seed[16] = 0x5363686eu; seed[17] = 0x6f72722bu; seed[18] = 0x53484132u; seed[19] = 0x35362020u;
    sh[0] = seed[0] >> 8;
#pragma unroll
    for (int t = 1; t < 20; ++t) sh[t] = seed[t - 1] << 24 | seed[t] >> 8;
    sh[20] = seed[19] << 24 | 0x00800000u;
    u32 v[8], k1[8], is[8], os[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = 0x01010101u;
    hmac_seed(k1, p.is0, p.os0, v, 0u, sh);
    hmac_keys(is, os, k1);
    hmac_32(v, is, os, v);
    hmac_seed(k1, is, os, v, 1u, sh);  // K2
    hmac_keys(is, os, k1);
    hmac_32(v, is, os, v);             // V2
    hmac_32(v, is, os, v);             // V3 = k
    be_to_limbs(kk, v);
  }

  u32 x[8];
  be_to_limbs(x, mh);
  bool valid = !sc_ge_n(x) && !sc_ge_n(kk);
  {
    u32 z = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) z |= x[i];
    valid = valid && z != 0u;
  }

  // ---- P = x G, R = k G ----
  u32 PX[9], PY[9], PZ[9], RX[9], RY[9], RZ[9];
  comb_mul(PX, PY, PZ, x, tab);
  comb_mul(RX, RY, RZ, kk, tab);

  // ---- 1 / (PZ RZ) for the block: Montgomery's trick with block-wide prefix / suffix products ----
  u32 q[9];
  fe_mul(q, PZ, RZ);
#pragma unroll
  for (int i = 0; i < 9; ++i) { sh_pre[tid][i] = q[i]; sh_suf[tid][i] = q[i]; }
  __syncthreads();
#pragma unroll 1
  for (u32 off = 1; off < TPB; off <<= 1) {
    u32 a[9], b[9];
    const bool hp = tid >= off, hs = tid + off < TPB;
    if (hp) {
#pragma unroll
      for (int i = 0; i < 9; ++i) a[i] = sh_pre[tid - off][i];
    }
    if (hs) {
#pragma unroll
      for (int i = 0; i < 9; ++i) b[i] = sh_suf[tid + off][i];
    }
    __syncthreads();
    if (hp) {
      u32 c[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) c[i] = sh_pre[tid][i];
      fe_mul(c, c, a);
#pragma unroll
      for (int i = 0; i < 9; ++i) sh_pre[tid][i] = c[i];
    }
    if (hs) {
      u32 c[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) c[i] = sh_suf[tid][i];
      fe_mul(c, c, b);
#pragma unroll
      for (int i = 0; i < 9; ++i) sh_suf[tid][i] = c[i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    u32 tot[9], inv[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) tot[i] = sh_pre[TPB - 1][i];
    fe_pow_chain(inv, tot, 0);
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_inv[i] = inv[i];
  }
  __syncthreads();
  u32 iq[9];  // 1 / q = inv(total) * prefix(tid - 1) * suffix(tid + 1)
#pragma unroll
  for (int i = 0; i < 9; ++i) iq[i] = sh_inv[i];
  if (tid > 0) {
    u32 a[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) a[i] = sh_pre[tid - 1][i];
    fe_mul(iq, iq, a);
  }
  if (tid + 1 < TPB) {
    u32 a[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) a[i] = sh_suf[tid + 1][i];
    fe_mul(iq, iq, a);
  }

  // ---- affine P.x, parity of P.y, R.x ----
  u32 zi[9], zi2[9], t[9], px9[9], rx9[9];
  fe_mul(zi, iq, RZ);  // 1 / PZ
  fe_sqr(zi2, zi);
  fe_mul(px9, PX, zi2);
  fe_mul(t, zi2, zi);
  fe_mul(t, PY, t);
  fe_canon(t);
  const u32 pyodd = t[0] & 1u;
  fe_mul(zi, iq, PZ);  // 1 / RZ
  fe_sqr(zi2, zi);
  fe_mul(rx9, RX, zi2);

  // ---- k -> n - k unless R.y = RY / RZ^3 is a quadratic residue (same Legendre symbol as RY RZ) ----
  fe_mul(t, RY, RZ);
  fe_canon(t);
  {
    u32 a[8];
    fe_to32(a, t);
    if (!is_qr(a)) sc_neg(kk, kk);
  }

  // ---- e = SHA256(R.x | 02/03 | P.x | h1) mod n (97 bytes), s = k + e x ----
  u32 rxw[8], pxw[8], e[8], sw[8];
  fe_to_be(rxw, rx9);
  fe_to_be(pxw, px9);
  {
    u32 st[8], w[16], strm[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { strm[i] = pxw[i]; strm[8 + i] = h1[i]; }
    sha_iv(st);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = rxw[i];
    w[8] = (2u + pyodd) << 24 | strm[0] >> 8;
#pragma unroll
    for (int i = 9; i < 16; ++i) w[i] = strm[i - 9] << 24 | strm[i - 8] >> 8;
    sha_blk(st, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = strm[7 + i] << 24 | strm[8 + i] >> 8;
    w[8] = strm[15] << 24 | 0x00800000u;
#pragma unroll
    for (int i = 9; i < 15; ++i) w[i] = 0u;
    w[15] = 97u * 8u;
    sha_blk(st, w);
    be_to_limbs(e, st);
    if (sc_ge_n(e)) sc_sub_n(e);
  }
  u32 s[8];
  sc_muladd(s, kk, e, x);
  limbs_to_be(sw, s);

  // ---- powHash = SHA256(R.x | s) ----
  u32 ph[8];
  {
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { w[i] = rxw[i]; w[8 + i] = sw[i]; }
    sha_iv(ph);
    sha_blk(ph, w);
    w[0] = 0x80000000u;
#pragma unroll
    for (int i = 1; i < 15; ++i) w[i] = 0u;
    w[15] = 512u;
    sha_blk(ph, w);
  }
  if (p.dump) {
    u32* o = dbg + (size_t)(blockIdx.x * TPB + tid) * 24;
#pragma unroll
    for (int i = 0; i < 8; ++i) { o[i] = ph[i]; o[8 + i] = rxw[i]; o[16 + i] = sw[i]; }
    if (!valid) o[0] = ~o[0];
    return;
  }
  // powHash as a little-endian number: top 64 bits = bytes 31..24 = bswap(ph7), bswap(ph6)
  const u32 h7 = ph[7], h6 = ph[6];
  const u64 top = (u64)((h7 >> 24) | ((h7 >> 8) & 0xff00u) | ((h7 << 8) & 0xff0000u) | (h7 << 24)) << 32 |
                  ((h6 >> 24) | ((h6 >> 8) & 0xff00u) | ((h6 << 8) & 0xff0000u) | (h6 << 24));
  if (valid && top <= p.target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < maxf) found[1 + slot] = blockIdx.x * TPB + tid;
  }
}
