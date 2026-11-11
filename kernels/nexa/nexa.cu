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

// One block into st (w is consumed). Rounds in unrolled groups of G (constant indices into w) to keep the code size
// of the ~20 call sites down. Stage 1 (kernel a: ~18 blocks per nonce) uses G = 32: with groups of 16 the AMD compiler
// copies the state at each group's back edge (38% of kernel a's instructions were moves; RX 9070 XT a 3.6 -> 2.7 ms
// per 1M nonces). Fully unrolled is barely faster and triples the run-time compile time.
template <int G>
__device__ __forceinline__ void sha_blk_g(u32 st[8], u32 w[16]) {
  u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int r = 0; r < 64; r += G) {
#pragma unroll
    for (int j = 0; j < G; ++j) {
      if (r || j >= 16) w[j & 15] += s1(w[(j + 14) & 15]) + w[(j + 9) & 15] + s0(w[(j + 1) & 15]);
      const u32 t1 = h + S1(e) + CH(e, f, g) + K256[r + j] + w[j & 15];
      const u32 t2 = S0(a) + MAJ(a, b, c);
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
__device__ __forceinline__ void sha_blk(u32 st[8], u32 w[16]) { sha_blk_g<16>(st, w); }
__device__ __forceinline__ void sha_blk1(u32 st[8], u32 w[16]) { sha_blk_g<32>(st, w); }  // stage 1

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
  sha_blk1(out, w);
}

// HMAC-SHA256 key states: inner / outer after the key's ipad / opad block (32-byte key).
__device__ __forceinline__ void hmac_keys(u32 is[8], u32 os[8], const u32 k[8]) {
  u32 w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (i < 8 ? k[i] : 0u) ^ 0x36363636u;
  sha_iv(is);
  sha_blk1(is, w);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = (i < 8 ? k[i] : 0u) ^ 0x5c5c5c5cu;
  sha_iv(os);
  sha_blk1(os, w);
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
  sha_blk1(out, w);
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
  sha_blk1(in, w);
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
  sha_blk1(in, w);
#pragma unroll
  for (int i = 0; i < 13; ++i) w[i] = sh[8 + i];  // sh[20] carries the 0x80 pad byte
#pragma unroll
  for (int i = 13; i < 15; ++i) w[i] = 0u;
  w[15] = (64u + 113u) * 8u;
  sha_blk1(in, w);
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
__device__ __forceinline__ void fe_red(u32 r[9], u64 c[17]) {
#ifdef __HIPCC__
  // AMD gfx12: without this (empty) fence between the products and the reduction, the compiler miscompiles the
  // inlined reduction at -O1 and above (results depend on other lanes' data). It costs nothing; out of line instead
  // halves the RX 9070 XT's speed.
#pragma unroll
  for (int k = 0; k < 17; ++k) asm volatile("" : "+v"(c[k]));
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
  // cross products a_i a_j (i < j) per column, doubled, plus the squares. AMD doubles one factor instead of the
  // 64-bit columns (+0.5% on the RX 9070 XT, -0.5% on the RTX 3060); m2 input: a column then holds at most four
  // cross products < 2^61.01 and one square, < 2^63.2.
  u64 c[17];
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] = 0;
#ifdef __HIPCC__
  u32 d[9];
#pragma unroll
  for (int j = 0; j < 9; ++j) d[j] = a[j] << 1;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
#pragma unroll
    for (int j = i + 1; j < 9; ++j) c[i + j] += (u64)a[i] * d[j];
  }
#else
#pragma unroll
  for (int i = 0; i < 9; ++i) {
#pragma unroll
    for (int j = i + 1; j < 9; ++j) c[i + j] += (u64)a[i] * a[j];
  }
#pragma unroll
  for (int k = 0; k < 17; ++k) c[k] <<= 1;
#endif
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

// r = (a - b) mod n for a, b < n
__device__ __forceinline__ void sc_submod(u32 r[8], const u32 a[8], const u32 b[8]) {
  long long c = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    c += (long long)a[i] - (long long)b[i];
    r[i] = (u32)c;
    c >>= 32;
  }
  if (c < 0) {
    long long d = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      d += (long long)r[i] + (long long)FN[i];
      r[i] = (u32)d;
      d >>= 32;
    }
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

// (X, Y, Z) = 2 (X, Y, Z) (dbl-2009-l, a = 0); X, Y m1, Z m2 in and out.
__device__ __forceinline__ void jdouble(u32 X[9], u32 Y[9], u32 Z[9]) {
  u32 A[9], B[9], C[9], D[9], E[9], t[9];
  fe_sqr(A, X);
  fe_sqr(B, Y);
  fe_sqr(C, B);
  fe_add(t, X, B);
  fe_sqr(t, t);
  fe_sub(t, t, A);
  fe_sub(D, t, C);
  fe_add(D, D, D);      // D = 2 ((X + B)^2 - A - C), m2
  fe_add(E, A, A);
  fe_add(E, E, A);
  fe_norm(E);           // E = 3 A
  fe_mul(Z, Y, Z);
  fe_add(Z, Z, Z);      // Z3 = 2 Y Z, m2
  fe_sqr(t, E);         // F = E^2
  fe_sub(t, t, D);
  fe_sub(X, t, D);      // X3 = F - 2 D
  fe_sub(t, D, X);
  fe_mul(t, E, t);      // E (D - X3)
  fe_add(C, C, C);
  fe_add(C, C, C);
  fe_norm(C);
  fe_add(C, C, C);      // 8 C, m2
  fe_sub(Y, t, C);      // Y3
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

// Legendre symbol (a / p) = 1 for a canonical, nonzero a, by Bernstein-Yang "posdivsteps" with Jacobi tracking
// (the method of libsecp256k1's secp256k1_jacobi32_maybe_var, MIT, written here for the GPU): batches of 30 steps
// are decided on the low 32 bits of f and g alone and collected in a 2x2 matrix, which is then applied to the full
// numbers (9 signed 30-bit limbs). The sign flips when g is divided by an odd power of 2 while f = 3, 5 mod 8, and
// when f and g are swapped while both are 3 mod 4. Converges (f = 1) after ~27 batches; 80 is the cap, after which
// the nonce is counted as a residue (a wrong guess there only loses that nonce: the CPU re-checks candidates).
#define M30 0x3FFFFFFF
__device__ __forceinline__ void to30(int r[9], const u32 a[8]) {
  r[0] = (int)(a[0] & M30);
  r[1] = (int)((a[0] >> 30 | a[1] << 2) & M30);
  r[2] = (int)((a[1] >> 28 | a[2] << 4) & M30);
  r[3] = (int)((a[2] >> 26 | a[3] << 6) & M30);
  r[4] = (int)((a[3] >> 24 | a[4] << 8) & M30);
  r[5] = (int)((a[4] >> 22 | a[5] << 10) & M30);
  r[6] = (int)((a[5] >> 20 | a[6] << 12) & M30);
  r[7] = (int)((a[6] >> 18 | a[7] << 14) & M30);
  r[8] = (int)(a[7] >> 16);
}

// 30 posdivsteps decided on the low 32 bits of f and g: the transition matrix [u v; q r] (entries as two's complement
// 32-bit values), eta and the Jacobi sign bit are updated.
__device__ __forceinline__ void posdivsteps30(int& eta, int& jac, u32 ff, u32 gg, u32& u, u32& v, u32& q, u32& r) {
  u = 1; v = 0; q = 0; r = 1;
  int i = 30;
#pragma unroll 1
  for (;;) {
    const int zeros = __ffs(gg | (0xFFFFFFFFu << i)) - 1;  // the sentinel bit stops at i
    gg >>= zeros;
    u <<= zeros;
    v <<= zeros;
    eta -= zeros;
    i -= zeros;
    jac ^= zeros & (int)((ff >> 1) ^ (ff >> 2));
    if (i == 0) break;
    if (eta < 0) {
      eta = -eta;
      jac ^= (int)((ff & gg) >> 1);
      u32 tmp = ff; ff = gg; gg = tmp;
      tmp = u; u = q; q = tmp;
      tmp = v; v = r; r = tmp;
    }
    // at most 5 bits per elimination: (3 f) xor 2 is 1 / f mod 2^5, against two Newton steps for 8 bits
    // (a simulation gives 0.2% more warp iterations, for a much shorter chain)
    const int limit = eta + 1 > i ? i : eta + 1;
    const u32 m = (0xFFFFFFFFu >> (32 - limit)) & 31u;
    const u32 x = (ff * 3u) ^ 2u;
    const u32 w = (gg * (0u - x)) & m;  // cancels the bottom bits of g: g + f w = 0 mod 2^limit
    gg += ff * w;
    q += u * w;
    r += v * w;
  }
}

// The Legendre test's batch: QR_K warp-uniform iterations instead of exactly 30 steps (in the loop above lanes need
// different iteration counts, and a warp runs the largest). A lane that reaches 30 steps idles; one that does fewer
// (s) has its matrix scaled by 2^(30 - s), so the update still divides by 2^30. Invariant: f 2^s = u f0 + v g0,
// g 2^s = q f0 + r g0. RTX 3060: kernel c -9% at QR_K 8 (7..10 within 2%).
// NEXA_QR_GPU 1: the R.y quadratic-residue test on the GPU for every nonce; 0 (default): only in the debug dump
// (selftest), otherwise both candidate signatures are hashed (nexa_stage2_sig).
#ifndef NEXA_QR_GPU
#define NEXA_QR_GPU 0
#endif
#ifndef QR_K
#define QR_K 8
#endif
// (jac2: the Jacobi sign in bit 1)
__device__ __forceinline__ void posdivsteps_k(int& eta, u32& jac2, u32 ff, u32 gg, u32& u, u32& v, u32& q, u32& r) {
  u = 1; v = 0; q = 0; r = 1;
  int i = 30;
#pragma unroll
  for (int it = 0; it < QR_K; ++it) {
    const int zeros = __ffs(gg | (0xFFFFFFFFu << i)) - 1;  // the sentinel bit stops at i
    gg >>= zeros;
    u <<= zeros;
    v <<= zeros;
    eta -= zeros;
    i -= zeros;
    jac2 ^= ((u32)zeros << 1) & (ff ^ (ff >> 1));  // bit 1: f = 3, 5 mod 8 and an odd shift
    // swap when eta < 0 and i > 0 (g is odd then), without a branch: c = all ones then (eta & -i < 0)
    const u32 c = (u32)((eta & -i) >> 31);
    eta = (eta ^ (int)c) - (int)c;
    jac2 ^= ff & gg & c;  // bit 1: both 3 mod 4
    u32 t = (ff ^ gg) & c; ff ^= t; gg ^= t;
    t = (u ^ q) & c; u ^= t; q ^= t;
    t = (v ^ r) & c; v ^= t; r ^= t;
    // m = 2^limit - 1, limit = min(eta + 1, i, 5) (0 when i = 0): 31 >> (5 - limit), the shift clamped at 32
    const int sh = max(max(4 - eta, 5 - i), 0);
#if defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    const u32 m = __funnelshift_rc(31u, 0u, (u32)sh);
#else
    const u32 m = sh > 5 ? 0u : 31u >> sh;
#endif
    const u32 x = (ff * 3u) ^ 2u;  // 1 / f mod 2^5
    const u32 w = (gg * (0u - x)) & m;
    gg += ff * w;
    q += u * w;
    r += v * w;
  }
  {  // strip the bits the last elimination cleared: the matrix bound then holds as at the end of the loop above
     // (entries < 2^30 after scaling; without it they reach 2^31)
    const int zeros = __ffs(gg | (0xFFFFFFFFu << i)) - 1;
    u <<= zeros;
    v <<= zeros;
    eta -= zeros;
    i -= zeros;
    jac2 ^= ((u32)zeros << 1) & (ff ^ (ff >> 1));
  }
  u <<= i;
  v <<= i;
  q <<= i;
  r <<= i;
}

// (f, g) = [u v; q r] (f, g) / 2^30, 9 signed 30-bit limbs
__device__ __forceinline__ void update_fg(int f[9], int g[9], u32 u, u32 v, u32 q, u32 r) {
  const long long tu = (int)u, tv = (int)v, tq = (int)q, tr = (int)r;
  long long cf = tu * f[0] + tv * g[0], cg = tq * f[0] + tr * g[0];
  cf >>= 30;
  cg >>= 30;
#pragma unroll
  for (int k = 1; k < 9; ++k) {
    cf += tu * f[k] + tv * g[k];
    cg += tq * f[k] + tr * g[k];
    f[k - 1] = (int)cf & M30;
    cf >>= 30;
    g[k - 1] = (int)cg & M30;
    cg >>= 30;
  }
  f[8] = (int)cf;
  g[8] = (int)cg;
}

__device__ __forceinline__ bool is_one30(const int f[9]) {
  int rest = f[0] ^ 1;
#pragma unroll
  for (int k = 1; k < 9; ++k) rest |= f[k];
  return rest == 0;
}

// update_fg on the low N limbs only (both f and g < 2^(30 N); max(f, g) never grows), then N shrinks when the top limb
// is zero in every lane of the warp (a vote over the lanes still running; each lane's own top limb is zero then).
template <int N>
__device__ __forceinline__ void update_fg_n(int f[9], int g[9], u32 u, u32 v, u32 q, u32 r, int& n) {
  const long long tu = (int)u, tv = (int)v, tq = (int)q, tr = (int)r;
  long long cf = tu * f[0] + tv * g[0], cg = tq * f[0] + tr * g[0];
  cf >>= 30;
  cg >>= 30;
#pragma unroll
  for (int k = 1; k < N; ++k) {
    cf += tu * f[k] + tv * g[k];
    cg += tq * f[k] + tr * g[k];
    f[k - 1] = (int)cf & M30;
    cf >>= 30;
    g[k - 1] = (int)cg & M30;
    cg >>= 30;
  }
  f[N - 1] = (int)cf;
  g[N - 1] = (int)cg;
  if (N > 1) {
#if defined(__HIPCC__)
    const bool top0 = __all((f[N - 1] | g[N - 1]) == 0);
#else
    const bool top0 = __all_sync(__activemask(), (f[N - 1] | g[N - 1]) == 0);
#endif
    if (top0) n = N - 1;
  }
}

__device__ __forceinline__ bool is_qr(const u32 a_in[8]) {
  int f[9], g[9];
  to30(f, FP);
  to30(g, a_in);
  int eta = -1, n = 9;
  u32 jac2 = 0;
#pragma unroll 1
  for (int it = 0; it < 80; ++it) {
    u32 u, v, q, r;
    posdivsteps_k(eta, jac2, (u32)f[0] | (u32)f[1] << 30, (u32)g[0] | (u32)g[1] << 30, u, v, q, r);
    switch (n) {  // warp-uniform in practice
      case 9: update_fg_n<9>(f, g, u, v, q, r, n); break;
      case 8: update_fg_n<8>(f, g, u, v, q, r, n); break;
      case 7: update_fg_n<7>(f, g, u, v, q, r, n); break;
      case 6: update_fg_n<6>(f, g, u, v, q, r, n); break;
      case 5: update_fg_n<5>(f, g, u, v, q, r, n); break;
      case 4: update_fg_n<4>(f, g, u, v, q, r, n); break;
      case 3: update_fg_n<3>(f, g, u, v, q, r, n); break;
      case 2: update_fg_n<2>(f, g, u, v, q, r, n); break;
      default: update_fg_n<1>(f, g, u, v, q, r, n); break;
    }
    if (is_one30(f)) return (jac2 & 2u) == 0;
  }
  return true;
}

// 1 / a mod p (a canonical, nonzero) with the same posdivsteps (variable time; thread 0 inverts the block's product
// with it): f = d a, g = e a (mod p) throughout, so d = 1 / a when f reaches 1. (d, e) take the same matrix plus a
// multiple of p that makes the division by 2^30 exact, and are kept in [0, p).
__constant__ int P30[9] = {0x3ffffc2f, 0x3ffffffb, 0x3fffffff, 0x3fffffff, 0x3fffffff,
                           0x3fffffff, 0x3fffffff, 0x3fffffff, 0xffff};
#define PINV30 0x2ddacacfu  // p^-1 mod 2^30

// x (limbs in [0, 2^30) except a signed top limb, value in (-p, 2p)) -> [0, p)
__device__ __forceinline__ void reduce30(int x[9]) {
  if (x[8] < 0) {  // + p
    long long c = 0;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      c += (long long)x[k] + P30[k];
      x[k] = k < 8 ? (int)c & M30 : (int)c;
      c >>= 30;
    }
  } else {  // - p unless that goes negative
    int y[9];
    long long c = 0;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      c += (long long)x[k] - P30[k];
      y[k] = k < 8 ? (int)c & M30 : (int)c;
      c >>= 30;
    }
    if (y[8] >= 0) {
#pragma unroll
      for (int k = 0; k < 9; ++k) x[k] = y[k];
    }
  }
}

__device__ __forceinline__ void update_de(int d[9], int e[9], u32 u, u32 v, u32 q, u32 r) {
  const long long tu = (int)u, tv = (int)v, tq = (int)q, tr = (int)r;
  long long cd = tu * d[0] + tv * e[0], ce = tq * d[0] + tr * e[0];
  const long long md = (long long)((0u - (u32)cd * PINV30) & M30), me = (long long)((0u - (u32)ce * PINV30) & M30);
  cd += md * P30[0];
  ce += me * P30[0];
  cd >>= 30;
  ce >>= 30;
#pragma unroll
  for (int k = 1; k < 9; ++k) {
    cd += tu * d[k] + tv * e[k] + md * P30[k];
    ce += tq * d[k] + tr * e[k] + me * P30[k];
    d[k - 1] = (int)cd & M30;
    cd >>= 30;
    e[k - 1] = (int)ce & M30;
    ce >>= 30;
  }
  d[8] = (int)cd;
  e[8] = (int)ce;
  reduce30(d);
  reduce30(e);
}

__device__ __forceinline__ void fe_inv_var(u32 r9[9], const u32 a9[9]) {
  u32 a[9], w[8];
  fe_copy(a, a9);
  fe_canon(a);
  fe_to32(w, a);
  int f[9], g[9], d[9], e[9];
  to30(f, FP);
  to30(g, w);
#pragma unroll
  for (int k = 0; k < 9; ++k) { d[k] = 0; e[k] = k ? 0 : 1; }
  int eta = -1, jac = 0;
#pragma unroll 1
  for (int it = 0; it < 60; ++it) {
    u32 u, v, q, r;
    posdivsteps30(eta, jac, (u32)f[0] | (u32)f[1] << 30, (u32)g[0] | (u32)g[1] << 30, u, v, q, r);
    update_fg(f, g, u, v, q, r);
    update_de(d, e, u, v, q, r);
    if (is_one30(f)) break;
  }
  // d (signed 30-bit limbs, in [0, p)) -> 32-bit limbs -> 9 x 29
  u32 o[8];
  o[0] = (u32)d[0] | (u32)d[1] << 30;
  o[1] = (u32)d[1] >> 2 | (u32)d[2] << 28;
  o[2] = (u32)d[2] >> 4 | (u32)d[3] << 26;
  o[3] = (u32)d[3] >> 6 | (u32)d[4] << 24;
  o[4] = (u32)d[4] >> 8 | (u32)d[5] << 22;
  o[5] = (u32)d[5] >> 10 | (u32)d[6] << 20;
  o[6] = (u32)d[6] >> 12 | (u32)d[7] << 18;
  o[7] = (u32)d[7] >> 14 | (u32)d[8] << 16;
  fe_from32(r9, o);
}

// R.y = RY / RZ^3 is a quadratic residue (same Legendre symbol as RY RZ)
__device__ __forceinline__ bool ry_is_qr(const u32 RY[9], const u32 RZ[9]) {
  u32 t[9], a[8];
  fe_mul(t, RY, RZ);
  fe_canon(t);
  fe_to32(a, t);
  return is_qr(a);
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
// Stage 1 of a nonce: miningHash, h1 = SHA256(miningHash), the RFC6979 nonce k and the private key x (big-endian
// words -> little-endian limbs); valid = both scalars in [1, n).
__device__ __forceinline__ void nexa_stage1(const Params& p, u64 nonce, u32 x[8], u32 kk[8], u32 h1[8], bool& valid) {
  // ---- miningHash = SHA256d(commitment | 0x0c | en1 | nonce): first block from the round-8 state ----
  u32 mh[8];
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
#pragma unroll
    for (int r = 16; r < 64; r += 16) {  // unrolled: one call site (see sha_blk_g)
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

  be_to_limbs(x, mh);
  valid = !sc_ge_n(x) && !sc_ge_n(kk);
  {
    u32 z = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) z |= x[i];
    valid = valid && z != 0u;
  }
}

// Stage 2: from P = x G and R = k G (Jacobian): the block's batched inversion (every thread of the block takes
// part), the signature, powHash and the target check. Two parts, so that kernel c loads P.X, P.Y, R.X and the
// scalars only after the inversion and the quadratic-residue test (fewer live registers there, more warps per SM).
// The inversion is Montgomery's trick on a product tree (tree_up / tree_down below).
// Montgomery's trick on a product tree in shared memory (sh_t[2 TPB], the heap layout: node i has children 2i, 2i + 1,
// leaves TPB..2 TPB - 1, any TPB): tree_up() multiplies the leaves q up to the root sh_t[1]; after the root has been
// replaced by its inverse (and a barrier), tree_down() gives each thread 1 / q (1 / child = 1 / parent * sibling).
// About 3 field multiplies per value, against 2 log2(TPB) + 2 for a block-wide prefix / suffix scan.
__device__ __forceinline__ void tree_up(u32 tid, const u32 q[9], u32 (*sh_t)[9]) {
#pragma unroll
  for (int i = 0; i < 9; ++i) sh_t[TPB + tid][i] = q[i];
  u32 lo = 1;  // the largest power of two below TPB: nodes [lo, 2 lo) are the lowest internal level
  while (2 * lo < TPB) lo <<= 1;
  __syncthreads();
#pragma unroll 1
  for (u32 m = lo; m >= 1; m >>= 1) {  // nodes [m, 2m) from their children
    const u32 i = m + tid;
    if (tid < m && i < TPB) {
      u32 a[9], b[9];
#pragma unroll
      for (int k = 0; k < 9; ++k) { a[k] = sh_t[2 * i][k]; b[k] = sh_t[2 * i + 1][k]; }
      fe_mul(a, a, b);
#pragma unroll
      for (int k = 0; k < 9; ++k) sh_t[i][k] = a[k];
    }
    __syncthreads();
  }
}
// inverses down, in place: every node of a level reads its parent's inverse and its sibling's product, then all
// write
__device__ __forceinline__ void tree_down(u32 tid, u32 (*sh_t)[9], u32 iq[9]) {
#pragma unroll 1
  for (u32 m = 2; m < 2 * TPB; m <<= 1) {  // nodes [m, 2m)
    const u32 c = m + tid;
    const bool on = tid < m && c < 2 * TPB;
    u32 a[9], b[9];
    if (on) {
#pragma unroll
      for (int k = 0; k < 9; ++k) { a[k] = sh_t[c >> 1][k]; b[k] = sh_t[c ^ 1u][k]; }
    }
    __syncthreads();
    if (on) {
      fe_mul(a, a, b);
#pragma unroll
      for (int k = 0; k < 9; ++k) sh_t[c][k] = a[k];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 9; ++i) iq[i] = sh_t[TPB + tid][i];
}

// Part 1: izp = 1 / PZ, izr = 1 / RZ, qr = R.y is a quadratic residue.
__device__ __forceinline__ void nexa_stage2_inv(u32 tid, const u32 PZ[9], const u32 RZ[9], const u32 RY[9],
                                                u32 (*sh_t)[9], u32 izp[9], u32 izr[9], bool& qr,
                                                bool doqr) {
  // ---- 1 / (PZ RZ) for the block ----
  u32 q[9];
  fe_mul(q, PZ, RZ);
  tree_up(tid, q, sh_t);
  // The root's inversion is a long dependent chain on thread 0: warp 0 runs it while the other warps take their
  // quadratic-residue test (R.y = RY / RZ^3 has the Legendre symbol of RY RZ), instead of all waiting at the barrier.
  // Warp 0 takes its own test at the end.
  qr = true;
  if (tid < 32) {
    if (tid == 0) {
      u32 r[9];
#pragma unroll
      for (int i = 0; i < 9; ++i) r[i] = sh_t[1][i];
      fe_inv_var(r, r);
#pragma unroll
      for (int i = 0; i < 9; ++i) sh_t[1][i] = r[i];
    }
  } else if (doqr) {
    qr = ry_is_qr(RY, RZ);
  }
  __syncthreads();
  u32 iq[9];  // 1 / q
  tree_down(tid, sh_t, iq);
  fe_mul(izp, iq, RZ);
  fe_mul(izr, iq, PZ);
  if (doqr && tid < 32) qr = ry_is_qr(RY, RZ);  // after the last barrier: the block does not wait for it
}

// Part 2: the signature, powHash and the target check.
__device__ __forceinline__ void nexa_stage2_sig(const Params& p, u32 tid, u32 x[8], u32 kk[8], const u32 h1[8],
                                                bool valid, const u32 PX[9], const u32 PY[9], const u32 RX[9],
                                                const u32 izp[9], const u32 izr[9], bool qr, bool doqr,
                                                u32* __restrict__ found,
                                                u32 maxf, u32* __restrict__ dbg) {
  // ---- affine P.x, parity of P.y, R.x ----
  u32 zi2[9], t[9], px9[9], rx9[9];
  fe_sqr(zi2, izp);
  fe_mul(px9, PX, zi2);
  fe_mul(t, zi2, izp);
  fe_mul(t, PY, t);
  fe_canon(t);
  const u32 pyodd = t[0] & 1u;
  fe_sqr(zi2, izr);
  fe_mul(rx9, RX, zi2);

  // ---- k -> n - k unless R.y is a quadratic residue ----
  if (!qr) sc_neg(kk, kk);

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
  auto pow_hash = [&](u32 ph[8], const u32 sw[8]) {
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
  };
  u32 ph[8];
  pow_hash(ph, sw);
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
  bool hit = top <= p.target_hi;
  if (!doqr) {
    // The R.y quadratic-residue test only picks k or n - k: R.x and e are the same, s = k + e x or e x - k. With
    // the test skipped (kk is k), the other signature's powHash is checked too; the CPU re-check of each candidate
    // finds which one is real (about as many false candidates as real ones, all dropped there).
    u32 s2[8], ph2[8];
    sc_submod(s2, s, kk);
    sc_submod(s2, s2, kk);
    limbs_to_be(sw, s2);
    pow_hash(ph2, sw);
    const u32 g7 = ph2[7], g6 = ph2[6];
    const u64 top2 = (u64)((g7 >> 24) | ((g7 >> 8) & 0xff00u) | ((g7 << 8) & 0xff0000u) | (g7 << 24)) << 32 |
                     ((g6 >> 24) | ((g6 >> 8) & 0xff00u) | ((g6 << 8) & 0xff0000u) | (g6 << 24));
    hit = hit || top2 <= p.target_hi;
  }
  if (valid && hit) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < maxf) found[1 + slot] = blockIdx.x * TPB + tid;
  }
}

#if !defined(XM_NEXA_GEN_SHELL) && !defined(XM_NEXA_TABLE_ONLY)  // (the generated build's cubin holds xm_nexa_b only)
extern "C" __global__ __launch_bounds__(TPB, MINB) void xm_nexa(Params p, const uint4* __restrict__ tab,
                                                                u32* __restrict__ found, u32 maxf,
                                                                u32* __restrict__ dbg) {
  __shared__ u32 sh_t[2 * TPB][9];
  const u32 tid = threadIdx.x;
  const u64 nonce = p.base + (u64)blockIdx.x * TPB + tid;
  u32 x[8], kk[8], h1[8];
  bool valid;
  nexa_stage1(p, nonce, x, kk, h1, valid);
  // ---- P = x G, R = k G ----
  u32 PX[9], PY[9], PZ[9], RX[9], RY[9], RZ[9];
  comb_mul(PX, PY, PZ, x, tab);
  comb_mul(RX, RY, RZ, kk, tab);
  u32 izp[9], izr[9];
  bool qr;
  const bool doqr = NEXA_QR_GPU || p.dump;
  nexa_stage2_inv(tid, PZ, RZ, RY, sh_t, izp, izr, qr, doqr);
  nexa_stage2_sig(p, tid, x, kk, h1, valid, PX, PY, RX, izp, izr, qr, doqr, found, maxf, dbg);
}
#endif

#ifdef XM_NEXA_SPLIT
#define PR_ROW 68
// The same work as three kernels over buffers (n nonces per launch, thread t = blockIdx.x TPB + threadIdx.x):
//   a: stage 1 -> st (27 words per nonce, word-major: st[w n + t]; 0..7 x, 8..15 k, 16..23 h1, 24 valid)
//   b: P = x G, R = k G -> pr: words w 0..47 per nonce: X, Y, Z of P then of R, each 8 x 32-bit little-endian
//      words < 2^256. Word-major (pr[w n + t], NEXA_WM) or, for the generated build, thread-major rows (pr[PR_ROW t +
//      w]; 48..65: its marker pins)
//   s: 1 / (PZ RZ) per nonce (NEXA_SCAN)
//   c: stage 2.
// The generated sm_86 build (tools/ngen) replaces b's comb multiplications with generated code: a large block of
// it needs a kernel around it that keeps almost nothing live (ptxas moves code freely across a block it cannot see).
// That cubin holds b only; the miner adds it next to the run-time compiled kernels.
#ifdef NEXA_WM
#define PRW(w, t) pr[(size_t)(w) * n + (t)]
#else
#define PRW(w, t) pr[PR_ROW * (size_t)(t) + (w)]
#endif
#ifndef XM_NEXA_GEN_SHELL
extern "C" __global__ __launch_bounds__(TPB, MINB) void xm_nexa_a(Params p, u32* __restrict__ st, u32 n) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
  u32 x[8], kk[8], h1[8];
  bool valid;
  nexa_stage1(p, p.base + t, x, kk, h1, valid);
#pragma unroll
  for (int i = 0; i < 8; ++i) { st[i * n + t] = x[i]; st[(8 + i) * n + t] = kk[i]; st[(16 + i) * n + t] = h1[i]; }
  st[24 * n + t] = valid;
}
#endif

extern "C" __global__ __launch_bounds__(TPB, MINB) void xm_nexa_b(const uint4* __restrict__ tab,
                                                                  const u32* __restrict__ st, u32* __restrict__ pr,
                                                                  u32 n) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
#ifdef XM_NEXA_GEN_SHELL
  // markers 0..7: x, 8..15: k, 16, 17: the address of this nonce's pr row. The generated block reads the table
  // pointer from the kernel parameters, leaves the marker registers unchanged and stores P, R to the pr row. Every
  // marker result is stored to the row's pin words (stores are kept and stay in place; unused or pure uses would let
  // ptxas drop or move the markers).
  u32 v[18];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = st[i * n + t];
  u32* row_p = pr + PR_ROW * (size_t)t;
  const u64 row = (u64)(size_t)row_p;
  v[16] = (u32)row;
  v[17] = (u32)(row >> 32);
#define XM_MK(i, n) "xor.b32 %" #i ", %" #i ", 0x0f1f4f" #n ";"
#define XM_MK8(a, b, c, d, e, f, g, h) XM_MK(0, a) XM_MK(1, b) XM_MK(2, c) XM_MK(3, d) XM_MK(4, e) XM_MK(5, f) \
  XM_MK(6, g) XM_MK(7, h)
#define XM_R8(o) "+r"(v[o]), "+r"(v[o + 1]), "+r"(v[o + 2]), "+r"(v[o + 3]), "+r"(v[o + 4]), "+r"(v[o + 5]), \
  "+r"(v[o + 6]), "+r"(v[o + 7])
  asm volatile(XM_MK8(00, 01, 02, 03, 04, 05, 06, 07) : XM_R8(0));
  asm volatile(XM_MK8(08, 09, 0a, 0b, 0c, 0d, 0e, 0f) : XM_R8(8));
  asm volatile(XM_MK(0, 10) XM_MK(1, 11) : "+r"(v[16]), "+r"(v[17]));
#undef XM_R8
#undef XM_MK8
#undef XM_MK
#pragma unroll
  for (int i = 0; i < 18; ++i) row_p[48 + i] = v[i];
#else
  u32 x[8], kk[8], P[3][9], R[3][9], w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { x[i] = st[i * n + t]; kk[i] = st[(8 + i) * n + t]; }
  comb_mul(P[0], P[1], P[2], x, tab);
  comb_mul(R[0], R[1], R[2], kk, tab);
#pragma unroll
  for (int c = 0; c < 6; ++c) {
    u32* a = c < 3 ? P[c] : R[c - 3];
    fe_canon(a);
    fe_to32(w, a);
#pragma unroll
    for (int i = 0; i < 8; ++i) PRW(8 * c + i, t) = w[i];
  }
#endif
}

#ifndef XM_NEXA_GEN_SHELL
// Kernel s: 1 / (PZ RZ) of every nonce by Montgomery's trick, NS_K nonces per thread (j, j + G, ... for G = n / NS_K
// threads): prefix products forward, one inversion, the inverses backward (~3 multiplies per nonce plus 1 / NS_K of an
// inversion; no barriers, no idle lanes). Word-major regions after P and R: q = PZ RZ, then 1 / q, at
// pr[68 n + w n + t] (9 x 29), the prefixes at pr[77 n + w n + t].
#ifndef NS_K
#define NS_K 32
#endif
extern "C" __global__ __launch_bounds__(TPB) void xm_nexa_s(u32* __restrict__ pr, u32 n) {
  const u32 G = n / NS_K;
  const u32 j = blockIdx.x * TPB + threadIdx.x;
  if (j >= G) return;
  u32* qb = pr + (size_t)68 * n + j;
  u32* pb = pr + (size_t)77 * n + j;
  u32 acc[9];
#pragma unroll
  for (int k = 0; k < 9; ++k) acc[k] = k ? 0u : 1u;
#pragma unroll 1
  for (int i = 0; i < NS_K; ++i) {
    const size_t t = (size_t)i * G;
    u32 a[9], b[9], q[9];
    {
      u32 w0[8], w1[8];  // PZ (words 16..23), RZ (40..47)
#pragma unroll
      for (int k = 0; k < 8; ++k) { w0[k] = PRW(16 + k, j + t); w1[k] = PRW(40 + k, j + t); }
      fe_from32(a, w0);
      fe_from32(b, w1);
    }
    fe_mul(q, a, b);
    fe_mul(acc, acc, q);
#pragma unroll
    for (int k = 0; k < 9; ++k) { qb[(size_t)k * n + t] = q[k]; pb[(size_t)k * n + t] = acc[k]; }
  }
  u32 inv[9];
  fe_inv_var(inv, acc);
#pragma unroll 1
  for (int i = NS_K - 1; i >= 0; --i) {
    const size_t t = (size_t)i * G;
    u32 q[9], iq[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) q[k] = qb[(size_t)k * n + t];
    if (i > 0) {
      u32 pp[9];
#pragma unroll
      for (int k = 0; k < 9; ++k) pp[k] = pb[(size_t)k * n + t - G];
      fe_mul(iq, inv, pp);
    } else {
      fe_copy(iq, inv);
    }
    fe_mul(inv, inv, q);
#pragma unroll
    for (int k = 0; k < 9; ++k) qb[(size_t)k * n + t] = iq[k];
  }
}
#endif

#ifndef XM_NEXA_GEN_SHELL
#ifndef MINB_C
#define MINB_C MINB
#endif
extern "C" __global__ __launch_bounds__(TPB, MINB_C) void xm_nexa_c(Params p, const u32* __restrict__ st,
                                                                    const u32* __restrict__ pr, u32 n,
                                                                    u32* __restrict__ found, u32 maxf,
                                                                    u32* __restrict__ dbg) {
  __shared__ u32 sh_t[2 * TPB][9];
  const u32 tid = threadIdx.x;
  const u32 t = blockIdx.x * TPB + tid;
  // words 8 c..8 c + 7: c = 0..5 for PX, PY, PZ, RX, RY, RZ
  auto load = [&](u32 r[9], int c) {
    u32 w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) w[k] = PRW(8 * c + k, t);
    fe_from32(r, w);
  };
  u32 izp[9], izr[9];
  bool qr;
  const bool doqr = NEXA_QR_GPU || p.dump;
  {
    u32 PZ[9], RY[9], RZ[9];
    load(PZ, 2);
    load(RY, 4);
    load(RZ, 5);
#ifdef NEXA_SCAN
    {  // 1 / (PZ RZ) from kernel s
      u32 iq[9];
#pragma unroll
      for (int k = 0; k < 9; ++k) iq[k] = pr[(size_t)68 * n + (size_t)k * n + t];
      fe_mul(izp, iq, RZ);
      fe_mul(izr, iq, PZ);
      qr = doqr ? ry_is_qr(RY, RZ) : true;
    }
#else
    nexa_stage2_inv(tid, PZ, RZ, RY, sh_t, izp, izr, qr, doqr);
#endif
  }
  u32 x[8], kk[8], h1[8], PX[9], PY[9], RX[9];
#pragma unroll
  for (int i = 0; i < 8; ++i) { x[i] = st[i * n + t]; kk[i] = st[(8 + i) * n + t]; h1[i] = st[(16 + i) * n + t]; }
  const bool valid = st[24 * n + t] != 0u;
  load(PX, 0);
  load(PY, 1);
  load(RX, 3);
  nexa_stage2_sig(p, tid, x, kk, h1, valid, PX, PY, RX, izp, izr, qr, doqr, found, maxf, dbg);
}
#endif
#endif

#ifdef XM_NEXA_TABLE_ONLY
// The comb table on the GPU (prebuilt as kernels/prebuilt/nexatab_*, see kernels/nexatab/nexatab.cu: compiled at run
// time it added ~18 s to every start): entry e = i 2^(w-1) + m - 1 = m 2^(w i) G (window i, 1 <= m <= 2^(w-1)),
// affine x then y as 8 little-endian words each (64 bytes). One thread per entry: m B_i by double-and-add from the
// window base B_i (bases: one affine point per window from the host, 16 words each), then the block's points made
// affine with one inversion (tree_up / tree_down). The w 24 table is 5.9 GB: on the CPU it took ~25 s on 24 cores
// and as much host memory per GPU. Grid: windows 2^(w-1) / TPB blocks.
extern "C" __global__ __launch_bounds__(TPB) void xm_nexa_table(const u32* __restrict__ bases, uint4* __restrict__ out,
                                                                u32 w) {
  __shared__ u32 sh_t[2 * TPB][9];
  const u32 tid = threadIdx.x;
  const u32 e = blockIdx.x * TPB + tid;
  const u32 wi = e >> (w - 1), m = (e & ((1u << (w - 1)) - 1u)) + 1u;
  u32 bx[9], by[9], X[9], Y[9], Z[9];
  fe_from32(bx, bases + 16 * wi);
  fe_from32(by, bases + 16 * wi + 8);
#pragma unroll
  for (int i = 0; i < 9; ++i) { X[i] = bx[i]; Y[i] = by[i]; Z[i] = i ? 0u : 1u; }
  const int top = 31 - __clz((int)m);
#pragma unroll 1
  for (int b = top - 1; b >= 0; --b) {
    jdouble(X, Y, Z);
    if (m >> b & 1u) madd(X, Y, Z, bx, by);
  }
  // affine: x = X / Z^2, y = Y / Z^3
  tree_up(tid, Z, sh_t);
  if (tid == 0) {
    u32 r[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) r[i] = sh_t[1][i];
    fe_inv_var(r, r);
#pragma unroll
    for (int i = 0; i < 9; ++i) sh_t[1][i] = r[i];
  }
  __syncthreads();
  u32 zi[9], zi2[9], t[9], xw[8], yw[8];
  tree_down(tid, sh_t, zi);
  fe_sqr(zi2, zi);
  fe_mul(X, X, zi2);
  fe_mul(t, zi2, zi);
  fe_mul(Y, Y, t);
  fe_canon(X);
  fe_canon(Y);
  fe_to32(xw, X);
  fe_to32(yw, Y);
  uint4* o = out + 4 * (size_t)e;
  o[0] = make_uint4(xw[0], xw[1], xw[2], xw[3]);
  o[1] = make_uint4(xw[4], xw[5], xw[6], xw[7]);
  o[2] = make_uint4(yw[0], yw[1], yw[2], yw[3]);
  o[3] = make_uint4(yw[4], yw[5], yw[6], yw[7]);
}
#endif
