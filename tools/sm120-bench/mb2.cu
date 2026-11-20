// Microbenchmark 2: MMA loop + operand loads. Per k-step: NLD 16-byte loads per lane, then 32 MMAs.
//  SRC 0: shared memory (cycling over SPAN bytes), 1: global memory (cycling over SPAN bytes per block)
//  PIPE 0: operands used by the same step's MMAs; 1: loaded one step ahead (volatile loads keep the order)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef WARPS
#define WARPS 8
#endif
#ifndef NLD
#define NLD 8
#endif
#ifndef SRC
#define SRC 0
#endif
#ifndef PIPE
#define PIPE 0
#endif
#ifndef SPAN
#define SPAN 65536
#endif
#ifndef VOL
#define VOL 1
#endif
#ifndef SHARE
#define SHARE 0  // 1: all warps read the same lines; 2: pairs (w, w+4) share; 4: groups of 4 share
#endif
#ifndef DATA
#define DATA 0
#endif
#ifndef SIG
#define SIG 63
#endif
#ifndef SIGBIT
#define SIGBIT 1
#endif
#ifndef SIGB
#define SIGB SIG
#endif
#ifndef LDM
#define LDM 0   // 1: ldmatrix.x4 instead of 16-byte loads (shared only)
#endif
__device__ __forceinline__ void mma(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds(unsigned addr) {
  uint4 r;
#if LDM
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
#else
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
#endif
  return r;
}
__device__ __forceinline__ uint4 ldg(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__global__ void __launch_bounds__(WARPS * 32) k(int steps, const uint4* gdata, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#if DATA == 1  // Pearl-like bytes: 63 or 64 plus the difference of two values in [-32, 31]
  if (SRC == 0) for (int i = threadIdx.x; i < SPAN; i += blockDim.x) {
    unsigned h = (i + blockIdx.x * 7919u) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    const int sg = ((i / 512) % 8) < 4 ? (SIG) : (SIGB);  // blocks 0-3 of each 4 KB feed A fragments, 4-7 B fragments
    smem[i] = (unsigned char)(sg + (h & 1) * SIGBIT + (int)((h >> 1) & 63) - (int)((h >> 7) & 63));
  }
#elif DATA == 2  // constant
  if (SRC == 0) for (int i = threadIdx.x; i < SPAN; i += blockDim.x) smem[i] = 64;
#else
  if (SRC == 0) for (int i = threadIdx.x; i < SPAN / 4; i += blockDim.x) ((unsigned*)smem)[i] = i * 2654435761u;
#endif
  __syncthreads();
  const unsigned sb = (unsigned)__cvta_generic_to_shared(smem);
  const uint4* gb = gdata + (size_t)blockIdx.x * (SPAN / 16);
  int acc[32][4];
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  uint4 nx[8], cur[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) nx[i] = make_uint4(lane + i, 2, 3, 4);
  unsigned pos = (SHARE == 1 ? 0 : SHARE == 2 ? (warp & 3) : SHARE == 4 ? (warp >> 2) : warp) * 512 * NLD;
  for (int s = 0; s < steps; ++s) {
#if PIPE
#pragma unroll
    for (int i = 0; i < 8; ++i) cur[i] = nx[i];
#endif
#pragma unroll
    for (int i = 0; i < NLD; ++i) {
      const unsigned off = (pos + i * 512 + lane * 16) & (SPAN - 1);
      const uint4 v = SRC == 0 ? lds(sb + (LDM ? (off & ~15u) : off)) : ldg(gb + off / 16);
#if PIPE
      nx[i] = v;
#else
      cur[i] = v;
#endif
    }
    pos += 512 * NLD * (WARPS > 8 ? 1 : 1);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint4& a = cur[i % (NLD > 4 ? 4 : NLD)];
        const uint4& b = cur[NLD > 4 ? 4 + (j >> 1) % (NLD - 4) : 0];
        unsigned af[4] = {a.x, a.y, a.z, a.w};
        mma(acc[i * 8 + j], af, (j & 1) ? b.z : b.x, (j & 1) ? b.w : b.y);
      }
  }
  unsigned r = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 20000; double mhz = argc > 2 ? atof(argv[2]) : 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  uint4* g; unsigned* sink; cudaMalloc(&g, (size_t)sms * SPAN); cudaMalloc(&sink, 64); cudaMemset(g, 0x21, (size_t)sms * SPAN);
  int sm = SRC == 0 ? SPAN : 0;
  if (sm > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, WARPS * 32, sm>>>(200, g, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed\n"); return 1; }
  k<<<sms, WARPS * 32, sm>>>(steps, g, sink); cudaDeviceSynchronize();
  cudaEventRecord(e0); k<<<sms, WARPS * 32, sm>>>(steps, g, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)steps * 32 * WARPS * sms * 4096 / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6;
  printf("WARPS=%d NLD=%d SRC=%s%s SPAN=%dK PIPE=%d: %.1f%% of peak\n", WARPS, NLD, SRC ? "global" : "shared", LDM ? "(ldmatrix)" : "", SPAN / 1024, PIPE, 100 * rate / peak);
  return 0;
}
