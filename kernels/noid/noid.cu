// X-Miner Parano1d (NOID) search kernel, compiled for AMD (HIP) and NVIDIA (CUDA) from this one file.
//
// Per job the host absorbs header fields 0..9 into a midstate. Per nonce the kernel finishes the sponge with 3
// permutations (nonce/f11, f12/f13, f14/f15), maps the top 64 bits of the digest back to the tower basis, and keeps
// nonces whose top 64 bits are <= the share target's top 64 bits. The host re-checks every candidate with the full
// reference digest. The kernel works in the composite basis of noid_comp.h (the host converts every input);
// noid_math.h / noid_perm.h hold the carry-less products and the permutation (ported from NoidMiner, Apache-2.0).
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include "noid/noid_comp.h"
#include "noid/noid_perm.h"

struct NoidJob {      // all W4 values in the composite basis
  W4 mid[4];          // sponge state after fields 0..9
  W4 f[5];            // fields 11..15
  W4 nonce_base;      // the nonce with the low 32 bits zero
  W4 t2f_lo[32];      // images of the low 32 nonce bits (tower bit i)
  uint64_t target_hi; // top 64 bits of the share target (bytes 24..31, little-endian)
};

struct RcGlobal {
  const W4* rc;  // [4][66] round constants
  __device__ __forceinline__ W4 get(int lane, int r) { return rc[lane * 66 + r]; }
};

// Grid of 1-D blocks; each thread tests one nonce (low 32 bits = counter_base + thread index).
// f2t_hi[i] = top 64 bits of the tower image of composite bit i.
// Constant multiplies use byte tables (NOID_CTBL_WORDS words = 24 KB) copied from tbl into shared memory / LDS;
// launch with 24 KB of dynamic shared memory.
extern "C" __global__ void xm_noid_search(NoidJob job, const W4* __restrict__ rc, const uint32_t* __restrict__ tbl,
                                          const uint64_t* __restrict__ f2t_hi, uint32_t counter_base,
                                          uint32_t* __restrict__ found, uint32_t max_found) {
  extern __shared__ uint32_t stbl[];
  for (uint32_t i = threadIdx.x; i < NOID_CTBL_WORDS; i += blockDim.x) stbl[i] = tbl[i];
  __syncthreads();
#ifndef XM_NOID_LANES
#define XM_NOID_LANES 1
#endif
  constexpr int L = XM_NOID_LANES;  // nonces per thread (only 1 is supported by permute())
  static_assert(L == 1, "XM_NOID_LANES");
  const uint32_t lo0 = counter_base + (blockIdx.x * blockDim.x + threadIdx.x) * L;
  CMulByte mul{stbl};
  RcGlobal rcg{rc};
  W4 s[L][4];
#pragma unroll
  for (int n = 0; n < L; ++n) {
    W4 nf = job.nonce_base;
    const uint32_t lo = lo0 + n;
#pragma unroll 4
    for (int b = 0; b < 32; ++b)
      if ((lo >> b) & 1u) nf = w4x(nf, job.t2f_lo[b]);
    s[n][0] = w4x(job.mid[0], nf);
    s[n][1] = w4x(job.mid[1], job.f[0]);
    s[n][2] = job.mid[2];
    s[n][3] = job.mid[3];
  }
  permute(s[0], mul, rcg);
#pragma unroll
  for (int n = 0; n < L; ++n) {
    s[n][0] = w4x(s[n][0], job.f[1]);
    s[n][1] = w4x(s[n][1], job.f[2]);
  }
  permute(s[0], mul, rcg);
#pragma unroll
  for (int n = 0; n < L; ++n) {
    s[n][0] = w4x(s[n][0], job.f[3]);
    s[n][1] = w4x(s[n][1], job.f[4]);
  }
  permute(s[0], mul, rcg);

#pragma unroll
  for (int n = 0; n < L; ++n) {
    // Top 64 bits of the digest = high half of the tower image of s[n][1].
    uint64_t hi = 0;
    const uint32_t w[4] = {s[n][1].w0, s[n][1].w1, s[n][1].w2, s[n][1].w3};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      uint32_t v = w[k];
#pragma unroll 8
      for (int bb = 0; bb < 32; ++bb)
        if ((v >> bb) & 1u) hi ^= f2t_hi[k * 32 + bb];
    }
    if (hi <= job.target_hi) {
      uint32_t slot = atomicAdd(&found[0], 1u);
      if (slot < max_found) found[1 + slot] = lo0 + n;
    }
  }
}
