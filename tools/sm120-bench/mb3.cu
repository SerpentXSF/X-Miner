// Microbenchmark 3: how fast can cp.async move a 48 KB stage from L2 into shared memory on one SM?
// CW copy warps issue the stage (96 x 512 B), a waiter warp waits for it (mbarrier) and releases it. Optional MW
// math warps run a pure MMA loop next to it. Reports cycles per stage and GB/s per SM.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef CW
#define CW 1
#endif
#ifndef MW
#define MW 0
#endif
#ifndef STAGE_KB
#define STAGE_KB 48
#endif
#ifndef SPAN_KB
#define SPAN_KB 512   // per-block source region (cycled); total must stay inside the 32 MB L2
#endif
#ifndef PACE
#define PACE 0  // ns the waiter sleeps before releasing a stage (throttles the copy rate)
#endif
#ifndef RANDOMDATA
#define RANDOMDATA 0
#endif
#ifndef CACHE
#define CACHE cg      // cp.async.cg (L2 only) or ca (L1 too)
#endif
#define STR2(x) #x
#define STR(x) STR2(x)
__device__ __forceinline__ void cp16(unsigned s, const void* g) { asm volatile("cp.async." STR(CACHE) ".shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g)); }
__device__ __forceinline__ void mma(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__((CW + 1 + MW) * 32) k(int stages, const char* src, unsigned* sink) {
  extern __shared__ __align__(128) char smem[];
  __shared__ __align__(8) unsigned long long bar[2];
  __shared__ volatile unsigned stop;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned sb = (unsigned)__cvta_generic_to_shared(smem), b0 = (unsigned)__cvta_generic_to_shared(bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0), "r"(32u * CW) : "memory");      // full
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0 + 8), "r"(1u) : "memory");        // empty
    stop = 0;
  }
  __syncthreads();
  const char* base = src + (size_t)blockIdx.x * SPAN_KB * 1024 + lane * 16;
  constexpr int NI = STAGE_KB * 2;  // 512-byte instructions per stage
  if (warp < CW) {
    for (int g = 0; g < stages; ++g) {
      if (g > 0) {
        unsigned done;
        for (;;) { asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b0 + 8), "r"((unsigned)((g - 1) & 1)) : "memory"); if (done) break; }
      }
      const char* gp = base + ((size_t)g * STAGE_KB * 1024) % ((size_t)SPAN_KB * 1024);
      auto copies = [&](const int C) {
#pragma unroll
        for (int i = C; i < NI; i += CW) cp16(sb + i * 512 + lane * 16, gp + i * 512);
      };
      if (CW == 1 || warp == 0) copies(0); else if (warp == 1) copies(1); else if (warp == 2) copies(2); else copies(3);
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(b0) : "memory");
    }
    return;
  }
  if (warp == CW) {  // waiter
    for (int g = 0; g < stages; ++g) {
      unsigned done;
      for (;;) { asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b0), "r"((unsigned)(g & 1)) : "memory"); if (done) break; __nanosleep(20); }
      if (PACE) __nanosleep(PACE);
      if (lane == 0) asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(b0 + 8) : "memory");
      __syncwarp();
    }
    if (lane == 0) stop = 1;
    return;
  }
  // math warps: pure MMAs until the copy side is done
  int acc[32][4]; unsigned a[4] = {1u + lane, 2, 3, 4};
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  unsigned n = 0;
  while (!stop) {
#pragma unroll
    for (int j = 0; j < 32; ++j) mma(acc[j], a, 5, 6);
    ++n;
  }
  if (lane == 0) atomicAdd(sink + 1, n);
}
int main(int argc, char** argv) {
  int stages = argc > 1 ? atoi(argv[1]) : 20000; double mhz = 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  char* g; unsigned* sink; size_t sz = (size_t)sms * SPAN_KB * 1024 + (1 << 20); cudaMalloc(&g, sz); cudaMalloc(&sink, 64);
#if RANDOMDATA
  { char* hbuf = (char*)malloc(sz); unsigned x = 12345; for (size_t i = 0; i < sz; ++i) { x = x * 1664525u + 1013904223u; unsigned h = x >> 8; hbuf[i] = (char)(63 + (h & 1) + (int)((h >> 1) & 63) - (int)((h >> 7) & 63)); } cudaMemcpy(g, hbuf, sz, cudaMemcpyHostToDevice); free(hbuf); }
#else
  cudaMemset(g, 0x21, sz);
#endif cudaMemset(sink, 0, 64);
  int sm = STAGE_KB * 1024, th = (CW + 1 + MW) * 32;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, th, sm>>>(200, g, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  cudaMemset(sink, 0, 64);
  cudaEventRecord(e0); k<<<sms, th, sm>>>(stages, g, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); unsigned s[2]; cudaMemcpy(s, sink, 8, cudaMemcpyDeviceToHost);
  double cyc = mhz * 1e6 * ms * 1e-3 / stages, gbs = (double)STAGE_KB * 1024 * stages / (ms * 1e-3) / 1e9;
  double mmapct = MW ? 100.0 * (double)s[1] * 32 * 4096 / (ms * 1e-3) / (sms * 1024.0 * mhz * 1e6) * 1 : 0;  // lane0 counts per warp
  printf("CW=%d MW=%d stage=%dKB span=%dKB cache=%s: %.0f cycles per stage, %.1f GB/s per SM (%.0f GB/s total)%s", CW, MW, STAGE_KB, SPAN_KB, STR(CACHE), cyc, gbs, gbs * sms, MW ? "" : "\n");
  if (MW) printf(", MMA %.1f%% of peak  [needs <= 4096 cycles per 48 KB stage at 100%%]\n", mmapct);
  return 0;
}
