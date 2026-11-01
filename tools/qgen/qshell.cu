// CUDA shell for the generated sm_86 Quantus kernel (see qgen.py). nvcc compiles it; qgen.py then replaces the two
// marker XORs (0x0f1f4f00, 0x0f1f4f01) with generated SASS that builds the state from (x0, x1) and job.pre, runs
// permutation 3's tail, the absorb and permutation 4, and leaves s0 in (x0, x1). The nonce setup and the target
// check stay compiled CUDA, identical in behaviour to xm_quantus_search in kernels/quantus/quantus.cu (one nonce
// per thread).
#define XM_QTC_NO_SEARCH
#include "../../kernels/quantus/quantus.cu"
#include "qgen_table.h"  // written by qgen.py: __constant__ kQgen[], the generated code's round-constant table

extern "C" __global__ void xm_quantus_search(QtcJob job, u64 nonce_base, u32* __restrict__ found, u32 max_found) {
  const u64 n0 = nonce_base + ((u64)blockIdx.x * blockDim.x + threadIdx.x);
  if (blockIdx.x == 0xffffffffu) found[1] = (u32)kQgen[237];  // keeps the table (never taken)
  u32 x0 = bswap32((u32)(n0 >> 32)), x1 = bswap32((u32)n0);
  asm volatile("xor.b32 %0, %0, 0x0f1f4f00; xor.b32 %1, %1, 0x0f1f4f01;" : "+r"(x0), "+r"(x1));
  u64 h = ((u64)x1 << 32) | x0;
  if (h >= 0xffffffff00000001ull) h -= 0xffffffff00000001ull;  // canonical (lazy values can sit in [p, 2^64))
  // The first 8 output bytes are s0 little-endian; read big-endian that is bswap64(s0).
  const u64 top = ((u64)bswap32((u32)h) << 32) | bswap32((u32)(h >> 32));
  if (top <= job.target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)n0;
      found[2 + 2 * slot] = (u32)(n0 >> 32);
    }
  }
}
