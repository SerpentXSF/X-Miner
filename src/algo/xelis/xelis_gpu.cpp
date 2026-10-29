#include "algo/xelis/xelis_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "core/console.hpp"

namespace xm::xelis {

namespace {

// One thread per nonce; thread t owns scratchpad words [t * XL_MEM, (t + 1) * XL_MEM). Three kernels: fill (stage 1),
// walk (stage 3, resumable slices) and final (stage 4 + share check), so no launch runs long and the walk kernel
// carries no BLAKE3 tree state (fewer registers, more threads resident). 128-bit division follows
// Hacker's Delight (divlu: normalized two-digit long division; divdu for 128-bit divisors); isqrt uses the double
// square root exactly as the reference (including its wrap-around near 2^64).
const char* kKernel = R"KRN(
typedef unsigned int u32;
typedef unsigned long long u64;
#define XL_MEM 67968u
#define XL_BUF 33984u
__device__ __forceinline__ u32 bswap32(u32 x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
__device__ __forceinline__ u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32u - n)); }
__device__ __forceinline__ u64 rotl64(u64 x, u32 r) { r &= 63u; return r ? (x << r) | (x >> (64u - r)) : x; }
__device__ __forceinline__ u64 rotr64(u64 x, u32 r) { r &= 63u; return r ? (x >> r) | (x << (64u - r)) : x; }
__device__ __forceinline__ u64 mulhi64(u64 a, u64 b) { return __umul64hi(a, b); }

// ---------------------------------------------------------------- BLAKE3
#define B3_START 1u
#define B3_END 2u
#define B3_PARENT 4u
#define B3_ROOT 8u
#define B3G(a, b, c, d, x, y)                                                   \
  a = a + b + (x); d = rotr32(d ^ a, 16); c = c + d; b = rotr32(b ^ c, 12);    \
  a = a + b + (y); d = rotr32(d ^ a, 8); c = c + d; b = rotr32(b ^ c, 7);
__device__ __forceinline__ void b3_compress(const u32 cv[8], const u32 msg[16], u64 counter, u32 len, u32 flags, u32 out[8]) {
  u32 v[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
               0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, (u32)counter, (u32)(counter >> 32), len, flags};
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    B3G(v[0], v[4], v[8], v[12], m[0], m[1]);
    B3G(v[1], v[5], v[9], v[13], m[2], m[3]);
    B3G(v[2], v[6], v[10], v[14], m[4], m[5]);
    B3G(v[3], v[7], v[11], v[15], m[6], m[7]);
    B3G(v[0], v[5], v[10], v[15], m[8], m[9]);
    B3G(v[1], v[6], v[11], v[12], m[10], m[11]);
    B3G(v[2], v[7], v[8], v[13], m[12], m[13]);
    B3G(v[3], v[4], v[9], v[14], m[14], m[15]);
    const u32 t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}
__constant__ u32 B3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                             0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

// ---------------------------------------------------------------- ChaCha8 block (IETF)
__device__ __forceinline__ u32 rotl32(u32 x, u32 n) { return (x << n) | (x >> (32u - n)); }
#define CQR(a, b, c, d) \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);
__device__ __forceinline__ void chacha8(const u32 key[8], u32 counter, const u32 nonce[3], u32 out[16]) {
  u32 in[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3],
                key[4], key[5], key[6], key[7], counter, nonce[0], nonce[1], nonce[2]};
  u32 x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) x[i] = in[i];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    CQR(0, 4, 8, 12); CQR(1, 5, 9, 13); CQR(2, 6, 10, 14); CQR(3, 7, 11, 15);
    CQR(0, 5, 10, 15); CQR(1, 6, 11, 12); CQR(2, 7, 8, 13); CQR(3, 4, 9, 14);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) out[i] = x[i] + in[i];
}

// ---------------------------------------------------------------- AES round
__constant__ unsigned char SBOX[256] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
  0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
  0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
  0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
  0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
  0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
__device__ __forceinline__ unsigned char xtime(unsigned char x) { return (unsigned char)((x << 1) ^ ((x >> 7) * 0x1b)); }
__device__ void aes_round(unsigned char s[16]) {
  const unsigned char key[16] = {'x','e','l','i','s','h','a','s','h','-','p','o','w','-','v','3'};
  unsigned char t[16];
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < 4; ++r) t[4 * c + r] = SBOX[s[4 * ((c + r) & 3) + r]];
  for (int c = 0; c < 4; ++c) {
    const unsigned char a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
    const unsigned char all = a0 ^ a1 ^ a2 ^ a3;
    s[4 * c] = a0 ^ all ^ xtime(a0 ^ a1) ^ key[4 * c];
    s[4 * c + 1] = a1 ^ all ^ xtime(a1 ^ a2) ^ key[4 * c + 1];
    s[4 * c + 2] = a2 ^ all ^ xtime(a2 ^ a3) ^ key[4 * c + 2];
    s[4 * c + 3] = a3 ^ all ^ xtime(a3 ^ a0) ^ key[4 * c + 3];
  }
}

// ---------------------------------------------------------------- 128-bit division
__device__ __forceinline__ int clz64(u64 x) { return x ? __clzll((long long)x) : 64; }
// (u1:u0) / v with u1 < v; the remainder in *rem.
__device__ u64 divlu(u64 u1, u64 u0, u64 v, u64* rem) {
  const u64 b = 1ull << 32;
  const int s = clz64(v);
  v <<= s;
  const u64 vn1 = v >> 32, vn0 = v & 0xffffffffull;
  const u64 un32 = s ? (u1 << s) | (u0 >> (64 - s)) : u1;
  const u64 un10 = u0 << s;
  const u64 un1 = un10 >> 32, un0 = un10 & 0xffffffffull;
  u64 q1 = un32 / vn1, rhat = un32 - q1 * vn1;
  while (q1 >= b || q1 * vn0 > b * rhat + un1) { q1 -= 1; rhat += vn1; if (rhat >= b) break; }
  const u64 un21 = un32 * b + un1 - q1 * v;
  u64 q0 = un21 / vn1;
  rhat = un21 - q0 * vn1;
  while (q0 >= b || q0 * vn0 > b * rhat + un0) { q0 -= 1; rhat += vn1; if (rhat >= b) break; }
  *rem = (un21 * b + un0 - q0 * v) >> s;
  return q1 * b + q0;
}
// n / d for 128-bit n and d (d != 0): the low 64 bits of the quotient and of the remainder.
__device__ void divmod128(u64 nh, u64 nl, u64 dh, u64 dl, u64* q, u64* r) {
  if (dh == 0) {
    u64 k = nh;
    if (nh >= dl) k = nh % dl;  // the high quotient word is not needed
    *q = divlu(k, nl, dl, r);
    return;
  }
  const int s = clz64(dh);
  const u64 v1 = s ? (dh << s) | (dl >> (64 - s)) : dh;
  const u64 u1h = nh >> 1, u1l = (nh << 63) | (nl >> 1);
  u64 tmp;
  const u64 q1 = divlu(u1h, u1l, v1, &tmp);
  u64 q0 = q1 >> (63 - s);
  if (q0 != 0) q0 -= 1;
  // rem = n - q0 * d (fits in 128 bits); one more step if rem >= d
  const u64 pl = q0 * dl, ph = q0 * dh + mulhi64(q0, dl);
  u64 rl = nl - pl, rh = nh - ph - (nl < pl ? 1ull : 0ull);
  if (rh > dh || (rh == dh && rl >= dl)) {
    q0 += 1;
    const u64 borrow = rl < dl ? 1ull : 0ull;
    rl -= dl;
    rh -= dh + borrow;
  }
  *q = q0;
  *r = rl;
}

// ---------------------------------------------------------------- XelisHash v3 stage 3 helpers
__device__ __forceinline__ u64 murmur(u64 s) {
  s ^= s >> 55; s *= 0xff51afd7ed558ccdull; s ^= s >> 32; s *= 0xc4ceb9fe1a85ec53ull; s ^= s >> 15;
  return s;
}
__device__ __forceinline__ u32 map_index(u64 x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  return (u32)mulhi64(x, (u64)XL_BUF);
}
__device__ __forceinline__ u64 isqrt64(u64 n) {
  if (n < 2) return n;
  const u64 ap = (u64)sqrt((double)n);
  if (ap * ap > n) return ap - 1;
  if ((ap + 1) * (ap + 1) <= n) return ap + 1;
  return ap;
}
__device__ u64 modpow(u64 base, u64 exp, u64 mod) {
  u64 result = 1, r;
  if (mod == 0) return 0;  // the reference divides by zero here (never seen: needs a zero 64-bit result)
  divmod128(0, base, 0, mod, &r, &base);
  while (exp > 0) {
    if (exp & 1) divmod128(mulhi64(result, base), result * base, 0, mod, &r, &result);
    divmod128(mulhi64(base, base), base * base, 0, mod, &r, &base);
    exp >>= 1;
  }
  return result;
}

struct XlInput { u32 w[28]; };
// Per-thread state between the stage 3 slices: addr_a, addr_b, result, r | step << 32 (step = i * XL_BUF + j).
#define XL_STEPS (2u * XL_BUF)

// ---- stage 1: input hash, then one ChaCha8 keystream per 32-byte input chunk (4 chunks of 135,936 bytes)
extern "C" __global__ __launch_bounds__(TPB) void xm_xelis_fill(u64* __restrict__ scratch, u64* __restrict__ state,
                                                                XlInput in, u64 start_nonce) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
  const u64 nonce = start_nonce + t;
  u64* const S = scratch + (size_t)t * XL_MEM;
  u32 inw[28];
#pragma unroll
  for (int i = 0; i < 28; ++i) inw[i] = in.w[i];
  inw[10] = bswap32((u32)(nonce >> 32));  // bytes 40..47: the nonce, big-endian
  inw[11] = bswap32((u32)nonce);
  u32 h[8], cv[8];
  {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = inw[i];
    b3_compress(B3_IV, m, 0, 64u, B3_START, cv);
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = i < 12 ? inw[16 + i] : 0u;
    b3_compress(cv, m, 0, 48u, B3_END | B3_ROOT, h);
  }
  u32 nonce12[3] = {h[0], h[1], h[2]};
  for (u32 ci = 0; ci < 4; ++ci) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = h[i];
#pragma unroll
    for (int i = 0; i < 8; ++i) m[8 + i] = (ci < 3 || i < 4) ? inw[ci * 8 + i] : 0u;
    b3_compress(B3_IV, m, 0, 64u, B3_START | B3_END | B3_ROOT, h);
    u64* part = S + ci * 16992u;  // 135,936 bytes = 2124 ChaCha blocks
    u32 ks[16];
    for (u32 blk = 0; blk < 2124u; ++blk) {
      chacha8(h, blk, nonce12, ks);
#pragma unroll
      for (int j = 0; j < 8; ++j) part[blk * 8u + j] = (u64)ks[2 * j] | ((u64)ks[2 * j + 1] << 32);
    }
    nonce12[0] = ks[13];  // the part's last 12 bytes
    nonce12[1] = ks[14];
    nonce12[2] = ks[15];
  }
  u64* st = state + (size_t)t * 4;
  st[0] = S[XL_MEM - 1];       // addr_a = B[XL_BUF - 1]
  st[1] = S[XL_BUF - 1] >> 32;  // addr_b = A[XL_BUF - 1] >> 32
  st[2] = 0;
  st[3] = 0;
}

// ---- stage 3, `steps` steps from where the previous slice stopped
extern "C" __global__ __launch_bounds__(TPB) void xm_xelis_walk(u64* __restrict__ scratch, u64* __restrict__ state,
                                                                u32 steps) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
  u64* const S = scratch + (size_t)t * XL_MEM;
  u64* const A = S;
  u64* const B = S + XL_BUF;
  u64* st = state + (size_t)t * 4;
  u64 addr_a = st[0], addr_b = st[1], result = st[2];
  u32 r = (u32)st[3], k = (u32)(st[3] >> 32);
  const u32 end = k + steps < XL_STEPS ? k + steps : XL_STEPS;
  for (; k < end; ++k) {
    const u32 i = k >= XL_BUF ? 1u : 0u, j = k - i * XL_BUF;
    if (j == 0) {  // start of a pass: one AES round on two scratchpad words
      const u64 mem_a = A[map_index(addr_a)];
      const u64 mem_b = B[map_index(mem_a ^ addr_b)];
      unsigned char blk[16];
#pragma unroll
      for (int q = 0; q < 8; ++q) { blk[q] = (unsigned char)(mem_b >> (8 * q)); blk[8 + q] = (unsigned char)(mem_a >> (8 * q)); }
      aes_round(blk);
      u64 h1 = 0, h2 = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) { h1 |= (u64)blk[q] << (8 * q); h2 |= (u64)blk[8 + q] << (8 * q); }
      result = ~(h1 ^ h2);
    }
    const u64 a = A[map_index(result)];
    const u64 b = B[map_index(a ^ ~rotr64(result, r))];
    const u64 c = S[r];
    r = r < XL_MEM - 1u ? r + 1u : 0u;
    u64 v, q, rm;
    switch ((u32)(rotl64(result, (u32)c) & 0xfull)) {
      case 0: divmod128(a + i, isqrt64(b + j), 0, murmur(c ^ result ^ i ^ j) | 1ull, &q, &rm); v = rm; break;
      case 1: v = rotl64((c + i) % isqrt64(b | 2ull), i + j) * isqrt64(a + j); break;
      case 2: v = (isqrt64(a + i) * isqrt64(c + j)) ^ (b + i + j); break;
      case 3: v = (a + b) * c; break;
      case 4: v = (b - c) * a; break;
      case 5: v = c - a + b; break;
      case 6: v = a - b + c; break;
      case 7: v = b * c + a; break;
      case 8: v = c * a + b; break;
      case 9: v = a * b * c; break;
      case 10: divmod128(a, b, 0, c | 1ull, &q, &rm); v = rm; break;
      case 11: {
        const u64 th = rotl64(result, r), tl = a | 2ull;  // t2 = (th:tl), t1 = (b:c)
        if (th > b || (th == b && tl > c)) v = c;
        else { divmod128(b, c, th, tl, &q, &rm); v = rm; }
        break;
      }
      case 12: divmod128(c, a, 0, b | 4ull, &q, &rm); v = q; break;
      case 13: {
        const u64 th = rotl64(result, r), tl = b, dl = c | 8ull;  // t1 = (th:tl), t2 = (a:dl)
        if (th > a || (th == a && tl > dl)) { divmod128(th, tl, a, dl, &q, &rm); v = q; }
        else v = a ^ b;
        break;
      }
      case 14: v = b * c + mulhi64(a, c); break;
      default: { const u64 rh = rotr64(result, r); v = a * b + c * rh + mulhi64(c, b); break; }
    }
    const u64 seed = v ^ result;
    result = rotl64(seed, r);
    const u32 it = map_index(seed);
    const u64 tv = ((murmur(v) & (1ull << 58)) ? B[it] : A[it]) ^ result;
    const u32 ia = map_index(tv ^ result ^ 0x9e3779b97f4a7c15ull);
    const u32 ib = map_index((u64)ia ^ ~result ^ 0xd2b74407b1ce6e93ull);
    const u64 old = A[ia];
    A[ia] = tv;
    B[ib] ^= old ^ rotr64(tv, i + j);
    if (j == XL_BUF - 1u) {  // end of a pass
      addr_a = modpow(addr_a, addr_b, result);
      addr_b = isqrt64(result) * ((u64)r + 1ull) * isqrt64(addr_a);
    }
  }
  st[0] = addr_a;
  st[1] = addr_b;
  st[2] = result;
  st[3] = (u64)r | ((u64)k << 32);
}

// ---- stage 4: BLAKE3 of the 543,744-byte scratchpad (531 chunks of 1024 bytes), then the share check
extern "C" __global__ __launch_bounds__(TPB) void xm_xelis_final(const u64* __restrict__ scratch, u64 start_nonce,
                                                                 u64 target_hi, u32* __restrict__ found, u32 max_found) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
  const u64 nonce = start_nonce + t;
  const u64* const S = scratch + (size_t)t * XL_MEM;
  u32 stack[10][8];
  int depth = 0;
  u32 node[8];
  for (u32 chunk = 0; chunk < 531u; ++chunk) {
    u32 c8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) c8[i] = B3_IV[i];
    const u64* src = S + chunk * 128u;
    for (u32 bi = 0; bi < 16u; ++bi) {
      u32 m[16];
#pragma unroll
      for (int q = 0; q < 8; ++q) { const u64 w = src[bi * 8u + q]; m[2 * q] = (u32)w; m[2 * q + 1] = (u32)(w >> 32); }
      const u32 fl = (bi == 0 ? B3_START : 0u) | (bi == 15u ? B3_END : 0u);
      b3_compress(c8, m, chunk, 64u, fl, c8);
    }
    if (chunk == 530u) {
#pragma unroll
      for (int i = 0; i < 8; ++i) node[i] = c8[i];
      break;
    }
    u32 total = chunk + 1u;  // merge completed subtrees
    while ((total & 1u) == 0u) {
      u32 m[16];
      --depth;
#pragma unroll
      for (int i = 0; i < 8; ++i) { m[i] = stack[depth][i]; m[8 + i] = c8[i]; }
      b3_compress(B3_IV, m, 0, 64u, B3_PARENT, c8);
      total >>= 1;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) stack[depth][i] = c8[i];
    ++depth;
  }
  while (depth > 0) {
    u32 m[16];
    --depth;
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = stack[depth][i]; m[8 + i] = node[i]; }
    b3_compress(B3_IV, m, 0, 64u, B3_PARENT | (depth == 0 ? B3_ROOT : 0u), node);
  }
  const u64 top = ((u64)bswap32(node[0]) << 32) | bswap32(node[1]);  // the hash read big-endian
  if (top <= target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)nonce;
      found[2 + 2 * slot] = (u32)(nonce >> 32);
    }
  }
}
)KRN";

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed, uint32_t max_threads)
    : dev_(dev), rng_(seed), max_threads_(max_threads) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_scratch_);
  dev_.free(d_state_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_XELIS_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  // Hashes in flight: each is a chain of dependent random reads in its own 544 KB, so more threads hide latency until
  // the scratchpads thrash the caches / TLB (RX 9070 XT: 3072 4.71, 4096 5.00, 6144 5.04, 8192 4.08, 16384 3.59 kH/s).
  // NVIDIA (RTX 3060) keeps gaining up to its memory limit (8192 3.47, 16384 4.28 kH/s); AMD peaks at ~4096.
  uint64_t want = dev_.info().vendor == gpu::Vendor::Amd ? 4096 : 32768;
  if (const char* e = std::getenv("XM_XELIS_THREADS")) want = std::strtoull(e, nullptr, 10);
  if (max_threads_) want = std::min<uint64_t>(want, max_threads_);
  const uint64_t mem = dev_.info().memory;
  const uint64_t fit = mem > (uint64_t(768) << 20) ? (mem - (uint64_t(768) << 20)) * 8 / 10 / kMemoryBytes : 0;
  threads_ = uint32_t(std::min(want, fit) / tpb_ * tpb_);
  if (threads_ == 0) {
    err = "not enough GPU memory for XelisHash v3 scratchpads";
    return false;
  }
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  d_scratch_ = dev_.alloc(size_t(threads_) * kMemoryBytes);
  d_state_ = dev_.alloc(size_t(threads_) * 32);
  if (const char* e = std::getenv("XM_XELIS_SLICE")) slice_steps_ = std::max(1u, uint32_t(std::atoi(e)));
  if (!d_found_ || !d_scratch_ || !d_state_) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  const std::string src = "#define TPB " + std::to_string(tpb_) + "\n" + kKernel;
  if (!gpu::rtc_compile(dev_.info(), src, "xelis.cu", opts, module_)) {
    err = "XelisHash kernel did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "xelishashv3"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  if (!have_job_ || std::memcmp(job.work.bytes, job_.work.bytes, kInputSize) != 0) {
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    counter_ = r >> 1;
  }
  job_ = job;
  have_job_ = true;
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  struct { uint32_t w[28]; } in;
  std::memcpy(in.w, job_.work.bytes, kInputSize);
  // Top 64 bits of floor((2^256 - 1) / difficulty): the GPU pre-filter; the CPU applies the exact rule.
  uint64_t target_hi = ~0ull;
  if (job_.difficulty > 1) {
    uint64_t rem = 0, q = 0;
    for (int bit = 255; bit >= 192; --bit) {
      const bool carry = rem >> 63;
      rem = rem << 1 | 1;
      q <<= 1;
      if (carry || rem >= job_.difficulty) {
        rem -= job_.difficulty;
        q |= 1;
      }
    }
    target_hi = q;
  }
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  const uint32_t blocks = threads_ / tpb_;
  void* fill_args[] = {&d_scratch_, &d_state_, &in, &base};
  void* final_args[] = {&d_scratch_, &base, &target_hi, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  bool ok = dev_.memset(d_found_, 0, 4) && dev_.launch("xm_xelis_fill", blocks, 1, 1, tpb_, 1, 1, 0, fill_args);
  // Stage 3 in slices so that no launch runs long (the Windows display watchdog resets the GPU after ~2 s).
  for (uint32_t done = 0; ok && done < uint32_t(2 * kBufferWords); done += slice_steps_) {
    uint32_t steps = slice_steps_;
    void* walk_args[] = {&d_scratch_, &d_state_, &steps};
    ok = dev_.launch("xm_xelis_walk", blocks, 1, 1, tpb_, 1, 1, 0, walk_args) && dev_.wait();
  }
  if (!ok || !dev_.launch("xm_xelis_final", blocks, 1, 1, tpb_, 1, 1, 0, final_args) || !dev_.wait() ||
      !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += threads_;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    Work w = job_.work;
    w.set_nonce(s.nonce);
    s.hash = hash(w.bytes, kInputSize, cpu_scratch_);
    if (!meets_difficulty(s.hash, job_.difficulty)) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.work = double(job_.difficulty);
    shares.push_back(s);
  }
  return threads_;
}

}  // namespace xm::xelis
