// Microbenchmark 4: cost of mbarrier operations inside the MMA loop. Per PERIOD MMAs: NTEST test_waits (always
// satisfied) and NARR lane-0 arrives on a count-(WARPS) barrier that a helper would normally complete.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef WARPS
#define WARPS 8
#endif
#ifndef PERIOD
#define PERIOD 128
#endif
#ifndef NTEST
#define NTEST 1
#endif
#ifndef NARR
#define NARR 0
#endif
#ifndef SMLD
#define SMLD 0   // plain shared loads per period (for comparison)
#endif
#ifndef SMST
#define SMST 0   // plain shared stores per period
#endif
__device__ __forceinline__ void mma(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(WARPS * 32) k(int iters, unsigned* sink) {
  __shared__ __align__(8) unsigned long long bar[2];
  __shared__ volatile unsigned word[64];
  const int lane = threadIdx.x & 31;
  const unsigned b0 = (unsigned)__cvta_generic_to_shared(bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0), "r"(1u) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0 + 8), "r"(0x7ffffu) : "memory");  // never completes
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(b0) : "memory");  // phase 0 done for good
  }
  if (threadIdx.x < 64) word[threadIdx.x] = threadIdx.x;
  __syncthreads();
  int acc[32][4]; unsigned a[4] = {1u + lane, 2, 3, 4}; unsigned x = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int q = 0; q < PERIOD / 32; ++q)
#pragma unroll
      for (int j = 0; j < 32; ++j) mma(acc[j], a, 5, 6);
#pragma unroll
    for (int e = 0; e < NTEST; ++e) {
      unsigned done;
      for (;;) {
        asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b0), "r"(0u) : "memory");
        if (done) break;
      }
    }
#pragma unroll
    for (int e = 0; e < NARR; ++e)
      asm volatile("{\n .reg .pred p;\n .reg .b64 st;\n setp.eq.s32 p, %1, 0;\n @p mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(b0 + 8), "r"(lane) : "memory");
#pragma unroll
    for (int e = 0; e < SMLD; ++e) { if (word[(lane + e) & 63] == 0xdeadbeefu) break; }
#pragma unroll
    for (int e = 0; e < SMST; ++e) asm volatile("{\n .reg .pred p;\n setp.eq.s32 p, %1, 0;\n @p st.shared.u32 [%0], %2;\n}\n" ::"r"(b0 + 16), "r"(lane), "r"(x) : "memory");
  }
  unsigned r = x;
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 8000; double mhz = 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  unsigned* sink; cudaMalloc(&sink, 64);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, WARPS * 32>>>(100, sink); cudaDeviceSynchronize();
  cudaEventRecord(e0); k<<<sms, WARPS * 32>>>(iters, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)iters * PERIOD * WARPS * sms * 4096 / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6, pct = 100 * rate / peak;
  printf("per %d MMAs: %d test_wait, %d arrive, %d smem load+branch, %d smem store: %.1f%% of peak -> %.0f cycles lost per period per warp pair\n", PERIOD, NTEST, NARR, SMLD, SMST, pct, (100 - pct) / 100 * PERIOD * 16 * 2);
  return 0;
}
