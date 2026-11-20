// Microbenchmark 8: in-SM operand generation by the math warps, conflict-free layout, 3 stage buffers.
// Tile 128 x 256 (8 warps of 64 x 64, 32 MMAs per warp per k-step, 128 accumulators per lane).
// Shared memory: TA[128 cols][132] (bytes 64 + u), TB[128 cols][260], UA / UB the same tables with plain u (0..63),
//   -> to save space one table per side holds u; word = 0x40404040 + x - y (GEN_T=0) .
// Stage buffer: 384 rows x 36 bytes (32 k + 4 pad), 3 buffers. Column pitch 132 / 260 and row pitch 36 keep the 32
// lanes of every load and store in different banks (lanes = 8 row groups x 4 k groups).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef GEN
#define GEN 1
#endif
#ifndef GENSYNC
#define GENSYNC 1
#endif
constexpr int PA = 132, PB = 260, PR = 36, NB = 3;
constexpr int OFF_UA = 0, OFF_UB = 128 * PA, OFF_ST = OFF_UB + 128 * PB, STAGE = 384 * PR, SMEM = OFF_ST + NB * STAGE;
__device__ __forceinline__ void mma(int* c, const uint4& a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned lds(unsigned a) { unsigned r; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(r) : "r"(a)); return r; }
__device__ __forceinline__ void sts(unsigned a, unsigned v) { asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void arrive0(unsigned bar, int lane) {
  asm volatile("{\n .reg .pred p;\n .reg .b64 st;\n setp.eq.s32 p, %1, 0;\n @p mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar), "r"(lane) : "memory");
}
__device__ __forceinline__ void waitbar(unsigned bar, unsigned parity) {
  unsigned done;
  for (;;) { asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(bar), "r"(parity) : "memory"); if (done) break; }
}
__global__ void __launch_bounds__(256) k(int steps, const uint4* offs, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bar[2 * NB];  // built[0..3), read[0..3)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < SMEM; i += blockDim.x) {
    unsigned h = (i + blockIdx.x * 7919u) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    smem[i] = (unsigned char)(h & 63);
  }
  const unsigned sb = (unsigned)__cvta_generic_to_shared(smem), b0 = (unsigned)__cvta_generic_to_shared(bar);
  if (threadIdx.x == 0) for (int s = 0; s < 2 * NB; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0 + 8 * s), "r"(8u) : "memory");
  __syncthreads();
  int acc[32][4];
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  // Build this warp's share (3 of the 24 lane-groups) of k-step `step` into buffer `buf`.
  auto generate = [&](int step, int buf) {
#pragma unroll
    for (int it = 0; it < 3; ++it) {
      const int q = warp * 3 + it, rg = (q >> 1) * 8 + (lane >> 2), lg = (q & 1) * 4 + (lane & 3);  // 96 row groups x 8 k groups
      const bool isA = q < 8;                                                                       // row groups 0..31 = A's 128 rows
      const unsigned ubase = sb + (isA ? OFF_UA + rg * 4 : OFF_UB + (rg - 32) * 4);
      const uint4 o = offs[((step & 255) * 8 + lg) * 2 + (isA ? 0 : 1)];
      const unsigned w[4] = {o.x, o.y, o.z, o.w};
      unsigned v[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) v[b] = 0x40404040u + lds(ubase + (w[b] & 0xffff)) - lds(ubase + (w[b] >> 16));
      const unsigned dst = sb + OFF_ST + buf * STAGE + rg * 4 * PR + lg * 4;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        sts(dst + j * PR, ((v[0] >> (8 * j)) & 0xff) | (((v[1] >> (8 * j)) & 0xff) << 8) | (((v[2] >> (8 * j)) & 0xff) << 16) | (((v[3] >> (8 * j)) & 0xff) << 24));
    }
  };
  const int g = lane >> 2, t = lane & 3, wm = warp >> 2, wn = warp & 3;
#if GEN
  generate(0, 0); generate(1, 1);
  arrive0(b0, lane); arrive0(b0 + 8, lane);
#endif
  for (int s = 0; s < steps; ++s) {
    const int buf = s % NB;
#if GEN
    waitbar(b0 + 8 * buf, (s / NB) & 1);
#endif
    uint4 af[4]; unsigned bf[8][2];
    const unsigned sa = sb + OFF_ST + buf * STAGE;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned r0 = sa + (wm * 64 + 16 * i + g) * PR + 4 * t;
      af[i] = make_uint4(lds(r0), lds(r0 + 8 * PR), lds(r0 + 16), lds(r0 + 8 * PR + 16));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned c0 = sa + (128 + wn * 64 + 8 * j + g) * PR + 4 * t;
      bf[j][0] = lds(c0), bf[j][1] = lds(c0 + 16);
    }
#if GEN
    arrive0(b0 + 8 * (NB + buf), lane);                         // this warp has read step s
#if GENSYNC
    if (s > 0) waitbar(b0 + 8 * (NB + (s - 1) % NB), ((s - 1) / NB) & 1);   // everyone has read step s - 1: its buffer is free
#endif
    generate(s + 2, (s + 2) % NB);
    arrive0(b0 + 8 * ((s + 2) % NB), lane);
#endif
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) mma(acc[i * 8 + j], af[i], bf[j][0], bf[j][1]);
  }
  unsigned r = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 40000; double mhz = 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  uint4* offs; unsigned* sink; const int NO = 256 * 8 * 2; cudaMalloc(&offs, NO * 16); cudaMalloc(&sink, 64);
  unsigned* h = (unsigned*)malloc(NO * 16); unsigned x = 4242;
  for (int e = 0; e < NO; ++e) for (int b = 0; b < 4; ++b) {
    x = x * 1664525u + 1013904223u; unsigned c0 = (x >> 9) & 127; x = x * 1664525u + 1013904223u; unsigned c1 = (x >> 9) & 127;
    unsigned P = (e & 1) ? PB : PA; h[e * 4 + b] = (c0 * P) | ((c1 * P) << 16);
  }
  cudaMemcpy(offs, h, NO * 16, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, 256, SMEM>>>(200, offs, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  cudaEventRecord(e0); k<<<sms, 256, SMEM>>>(steps, offs, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)steps * 32 * 8 * sms * 4096 / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6;
  printf("generation %s (sync %d), tile 128x256: %.1f TMAC/s = %.1f%% of peak (smem %d KB)\n", GEN ? "ON" : "off", GENSYNC, rate / 1e12, 100 * rate / peak, SMEM / 1024);
  return 0;
}
