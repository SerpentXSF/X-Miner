// X-Miner Parano1d (NOID) search kernel, compiled for AMD (HIP) and NVIDIA (CUDA) from this one file.
//
// Per job the host absorbs header fields 0..9 into a flat-basis midstate. Per nonce the kernel finishes the sponge
// with 3 permutations (nonce/f11, f12/f13, f14/f15), maps the top 64 bits of the digest back to the tower basis,
// and keeps nonces whose top 64 bits are <= the share target's top 64 bits. The host re-checks every candidate
// with the full reference digest. GF(2^128) arithmetic: noid_math.h / noid_perm.h (ported from NoidMiner,
// Apache-2.0).
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include "noid/noid_perm.h"

struct NoidJob {
  W4 mid[4];          // sponge state after fields 0..9 (flat)
  W4 f[5];            // fields 11..15 (flat)
  W4 nonce_base;      // flat image of the nonce with the low 32 bits zero
  W4 t2f_lo[32];      // flat images of the low 32 nonce bits (tower bit i -> flat)
  uint64_t target_hi; // top 64 bits of the share target (bytes 24..31, little-endian)
};

struct RcGlobal {
  const W4* rc;  // [4][66] flat round constants
  __device__ __forceinline__ W4 get(int lane, int r) { return rc[lane * 66 + r]; }
};

// Grid of 1-D blocks; each thread tests one nonce (low 32 bits = counter_base + thread index).
// f2t_hi[i] = top 64 bits of flat_to_tower(bit i).
// Constant multiplies use nibble tables (NOID_TBL_WORDS words = 48 KB) copied from tbl into shared memory / LDS;
// launch with 48 KB of dynamic shared memory.
extern "C" __global__ void xm_noid_search(NoidJob job, const W4* __restrict__ rc, const uint32_t* __restrict__ tbl,
                                          const uint64_t* __restrict__ f2t_hi, uint32_t counter_base,
                                          uint32_t* __restrict__ found, uint32_t max_found) {
  extern __shared__ uint32_t stbl[];
  for (uint32_t i = threadIdx.x; i < NOID_TBL_WORDS; i += blockDim.x) stbl[i] = tbl[i];
  __syncthreads();
  const uint32_t lo = counter_base + blockIdx.x * blockDim.x + threadIdx.x;
  W4 n = job.nonce_base;
#pragma unroll 4
  for (int b = 0; b < 32; ++b)
    if ((lo >> b) & 1u) n = w4x(n, job.t2f_lo[b]);

  CMulTable mul{stbl};
  RcGlobal rcg{rc};
  W4 s[4] = {w4x(job.mid[0], n), w4x(job.mid[1], job.f[0]), job.mid[2], job.mid[3]};
  permute(s, mul, rcg);
  s[0] = w4x(s[0], job.f[1]);
  s[1] = w4x(s[1], job.f[2]);
  permute(s, mul, rcg);
  s[0] = w4x(s[0], job.f[3]);
  s[1] = w4x(s[1], job.f[4]);
  permute(s, mul, rcg);

  // Top 64 bits of the digest = high half of flat_to_tower(s[1]).
  uint64_t hi = 0;
  const uint32_t w[4] = {s[1].w0, s[1].w1, s[1].w2, s[1].w3};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t v = w[k];
#pragma unroll 8
    for (int b = 0; b < 32; ++b)
      if ((v >> b) & 1u) hi ^= f2t_hi[k * 32 + b];
  }
  if (hi > job.target_hi) return;
  uint32_t slot = atomicAdd(&found[0], 1u);
  if (slot < max_found) found[1 + slot] = lo;
}
