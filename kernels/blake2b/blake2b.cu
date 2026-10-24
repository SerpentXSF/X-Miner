// X-Miner BLAKE2b (Bitcoin Knots RDTS) search kernel, compiled for AMD (HIP) and NVIDIA (CUDA) from this one file.
//
// An 80-byte work header is one BLAKE2b block: message words m0..m3 prevhash, m4 nonce (the only word that varies),
// m5 ntime, m6..m9 work root, m10..m15 zero. The host runs the three column steps of round 1 that do not touch m4
// and passes that state, so each nonce costs 93 of the 96 G functions. Only the first output word is formed: a nonce
// is kept when its top 64 digest bits (read big-endian) are <= the share target's top 64 bits, and the host
// re-checks every candidate with the full CPU reference.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

struct B2bJob {
  uint64_t v[16];      // state after round-1 column steps 0, 1 and 3 (the host's precompute)
  uint64_t m[10];      // header words; m[4] is ignored
  uint64_t target_hi;  // top 64 bits of the share target
};

#if defined(__HIP__)
// Two v_alignbit_b32 per rotate (one per half); the plain shift form compiles to ~2.9 instructions on gfx12.
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
  const uint32_t lo = static_cast<uint32_t>(x), hi = static_cast<uint32_t>(x >> 32);
  if (n == 32) return (uint64_t(lo) << 32) | hi;
  if (n > 32) {  // swap the halves, then rotate by n - 32
    n -= 32;
    return (uint64_t(__builtin_amdgcn_alignbit(hi, lo, n)) << 32) | __builtin_amdgcn_alignbit(lo, hi, n);
  }
  return (uint64_t(__builtin_amdgcn_alignbit(lo, hi, n)) << 32) | __builtin_amdgcn_alignbit(hi, lo, n);
}
#else
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
#endif

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
#if defined(__HIP__)
  return __builtin_bswap64(x);
#else
  const uint32_t lo = static_cast<uint32_t>(x), hi = static_cast<uint32_t>(x >> 32);
  return (uint64_t(__byte_perm(lo, 0, 0x0123)) << 32) | __byte_perm(hi, 0, 0x0123);
#endif
}

#define B2B_G(a, b, c, d, x, y) \
  a = a + b + (x);              \
  d = rotr64(d ^ a, 32);        \
  c = c + d;                    \
  b = rotr64(b ^ c, 24);        \
  a = a + b + (y);              \
  d = rotr64(d ^ a, 16);        \
  c = c + d;                    \
  b = rotr64(b ^ c, 63);

// Message word s: constant indices after macro expansion, so the zero words fold away and the rest are uniform loads.
#define B2B_M(s) ((s) == 4 ? nonce : (s) >= 10 ? 0ull : job.m[(s) < 10 ? (s) : 0])

#define B2B_ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf) \
  B2B_G(v0, v4, v8, v12, B2B_M(s0), B2B_M(s1))                                    \
  B2B_G(v1, v5, v9, v13, B2B_M(s2), B2B_M(s3))                                    \
  B2B_G(v2, v6, v10, v14, B2B_M(s4), B2B_M(s5))                                   \
  B2B_G(v3, v7, v11, v15, B2B_M(s6), B2B_M(s7))                                   \
  B2B_G(v0, v5, v10, v15, B2B_M(s8), B2B_M(s9))                                   \
  B2B_G(v1, v6, v11, v12, B2B_M(sa), B2B_M(sb))                                   \
  B2B_G(v2, v7, v8, v13, B2B_M(sc), B2B_M(sd))                                    \
  B2B_G(v3, v4, v9, v14, B2B_M(se), B2B_M(sf))

// Grid of 1-D blocks, one nonce per thread: nonce = nonce_base + global thread index. found[0] counts candidates;
// found[1 + 2k], found[2 + 2k] hold candidate k's low and high nonce words (up to max_found).
extern "C" __global__ void xm_blake2b_search(B2bJob job, uint64_t nonce_base, uint32_t* __restrict__ found,
                                             uint32_t max_found) {
  const uint64_t nonce = nonce_base + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t v0 = job.v[0], v1 = job.v[1], v2 = job.v[2], v3 = job.v[3], v4 = job.v[4], v5 = job.v[5],
           v6 = job.v[6], v7 = job.v[7], v8 = job.v[8], v9 = job.v[9], v10 = job.v[10], v11 = job.v[11],
           v12 = job.v[12], v13 = job.v[13], v14 = job.v[14], v15 = job.v[15];
  // Round 1: column steps 0, 1, 3 are in job.v; column step 2 takes the nonce.
  B2B_G(v2, v6, v10, v14, nonce, job.m[5])
  B2B_G(v0, v5, v10, v15, job.m[8], job.m[9])
  B2B_G(v1, v6, v11, v12, 0ull, 0ull)
  B2B_G(v2, v7, v8, v13, 0ull, 0ull)
  B2B_G(v3, v4, v9, v14, 0ull, 0ull)
  B2B_ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
  B2B_ROUND(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4)
  B2B_ROUND(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8)
  B2B_ROUND(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13)
  B2B_ROUND(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9)
  B2B_ROUND(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11)
  B2B_ROUND(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10)
  B2B_ROUND(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5)
  B2B_ROUND(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0)
  B2B_ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  B2B_ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
  // First output word only (the compiler drops the rest of the last round); h0 = IV0 ^ param block.
  const uint64_t h0 = 0x6a09e667f3bcc908ull ^ 0x01010020ull ^ v0 ^ v8;
  if (bswap64(h0) <= job.target_hi) {
    const uint32_t slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = static_cast<uint32_t>(nonce);
      found[2 + 2 * slot] = static_cast<uint32_t>(nonce >> 32);
    }
  }
}
