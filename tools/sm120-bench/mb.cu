// Microbenchmark: how does sm_120 schedule int8 m16n8k32 MMAs next to other instructions?
// Build: nvcc -O3 -arch=sm_120 -DNA=32 -DEXTRA=0 -DDEP=0 -DWARPS=8 mb.cu -o mb
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef NA
#define NA 32      // accumulator fragments per lane (x4 regs)
#endif
#ifndef EXTRA
#define EXTRA 0    // extra ALU instructions per MMA
#endif
#ifndef DEP
#define DEP 0      // 1: the extra instructions consume the MMA's outputs (like the fold)
#endif
#ifndef WARPS
#define WARPS 8
#endif
#ifndef BURST
#define BURST 0    // >0: instead of EXTRA per MMA, a burst of BURST ALU instrs after each NA MMAs
#endif
__device__ __forceinline__ void mma(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#ifndef OP
#define OP 0
#endif
__device__ __forceinline__ void alu(unsigned& x, unsigned y) {
#if OP == 0
  asm volatile("xor.b32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 1
  asm volatile("add.u32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 2
  asm volatile("shf.r.wrap.b32 %0, %0, %0, 7;\n" : "+r"(x));
#elif OP == 3
  asm volatile("shl.b32 %0, %0, 3;\n" : "+r"(x));
#elif OP == 4
  asm volatile("sub.u32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 5
  asm volatile("shr.u32 %0, %0, 8;\n" : "+r"(x));
#elif OP == 6
  asm volatile("or.b32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 7
  x = __byte_perm(x, y, 0x5140 + (x & 0));
#elif OP == 8
  asm volatile("and.b32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 9
  asm volatile("mad.lo.u32 %0, %0, 3, %1;\n" : "+r"(x) : "r"(y));
#elif OP == 10
  asm volatile("min.u32 %0, %0, %1;\n" : "+r"(x) : "r"(y));
#endif
}
__device__ __forceinline__ void gfn(unsigned* v, int a, int b, int c, int d, unsigned mx, unsigned my) {
  v[a] = v[a] + v[b] + mx; v[d] = ((v[d] ^ v[a]) >> 16) | ((v[d] ^ v[a]) << 16);
  v[c] = v[c] + v[d];      v[b] = ((v[b] ^ v[c]) >> 12) | ((v[b] ^ v[c]) << 20);
  v[a] = v[a] + v[b] + my; v[d] = ((v[d] ^ v[a]) >> 8) | ((v[d] ^ v[a]) << 24);
  v[c] = v[c] + v[d];      v[b] = ((v[b] ^ v[c]) >> 7) | ((v[b] ^ v[c]) << 25);
}
__global__ void __launch_bounds__(WARPS * 32) k(int iters, const unsigned* seed, unsigned* sink) {
  unsigned a[4], b0, b1; const unsigned* s = seed + (threadIdx.x & 31) * 8;
  for (int i = 0; i < 4; ++i) a[i] = s[i]; b0 = s[4]; b1 = s[5];
  int acc[NA][4];
#pragma unroll
  for (int j = 0; j < NA; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  unsigned x = s[6], y = s[7];
  unsigned hv[16];
  for (int i = 0; i < 16; ++i) hv[i] = s[i & 7] + i;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < NA; ++j) {
      mma(acc[j], a, b0, b1);
#pragma unroll
      for (int e = 0; e < EXTRA; ++e) alu(x, DEP ? (unsigned)acc[(j + NA - 1) % NA][e & 3] : y);
#ifdef GEVERY
      if (j % GEVERY == GEVERY - 1) gfn(hv, (j / GEVERY) & 3, 4 + ((j / GEVERY) & 3), 8 + ((j / GEVERY) & 3), 12 + ((j / GEVERY) & 3), x, y);
#endif
    }
#pragma unroll 1
    for (int e = 0; e < BURST; ++e) alu(x, DEP ? (unsigned)acc[e % NA][e & 3] : y);
  }
  unsigned r = x;
  for (int i = 0; i < 16; ++i) r ^= hv[i];
#pragma unroll
  for (int j = 0; j < NA; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000; double mhz = argc > 2 ? atof(argv[2]) : 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  std::vector<unsigned> h(256); for (auto& v : h) v = (unsigned)rand() * 2654435761u;
  unsigned *seed, *sink; cudaMalloc(&seed, 1024); cudaMalloc(&sink, 64); cudaMemcpy(seed, h.data(), 1024, cudaMemcpyHostToDevice);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, WARPS * 32>>>(200, seed, sink); cudaDeviceSynchronize();
  for (int rep = 0; rep < 3; ++rep) { k<<<sms, WARPS * 32>>>(iters, seed, sink); }
  cudaDeviceSynchronize();
  cudaEventRecord(e0); k<<<sms, WARPS * 32>>>(iters, seed, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double mmas = (double)iters * NA * WARPS * sms, mac = mmas * 4096, rate = mac / (ms * 1e-3);
  double peak = sms * 1024.0 * mhz * 1e6;
  printf("NA=%d EXTRA=%d DEP=%d BURST=%d WARPS=%d: %.1f TMAC/s = %.1f%% of peak (%.2f clk per MMA per SMSP)\n", NA, EXTRA, DEP, BURST, WARPS,
         rate / 1e12, 100 * rate / peak, mhz * 1e6 * ms * 1e-3 / (mmas / (sms * 4)));
  return 0;
}
