// Microbenchmark 5: int8 MMA shapes, throughput and (with pw.py) power. SHAPE 32: m16n8k32, 16: m16n8k16, 8: m8n8k16.
// Operands change every MMA (Pearl-like bytes from shared memory), accumulators in registers.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef SHAPE
#define SHAPE 32
#endif
#ifndef WARPS
#define WARPS 8
#endif
__global__ void __launch_bounds__(WARPS * 32) k(int steps, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 65536; i += blockDim.x) {
    unsigned h = (i + blockIdx.x * 7919u) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    smem[i] = (unsigned char)(63 + (h & 1) + (int)((h >> 1) & 63) - (int)((h >> 7) & 63));
  }
  __syncthreads();
  const unsigned* sp = (const unsigned*)smem;
  int acc[32][4];
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  unsigned pos = warp * 1024 + lane * 4;
  for (int s = 0; s < steps; ++s) {
    unsigned f[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) f[i] = sp[(pos + i * 32) & 16383];  // 32 words per lane per step (4 KB per warp)
    pos += 1024;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int* c = acc[i * 8 + j];
#if SHAPE == 32
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(f[i * 4]), "r"(f[i * 4 + 1]), "r"(f[i * 4 + 2]), "r"(f[i * 4 + 3]), "r"(f[16 + j * 2]), "r"(f[17 + j * 2]));
#elif SHAPE == 16
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
              : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(f[i * 4 + 2 * h2]), "r"(f[i * 4 + 2 * h2 + 1]), "r"(f[16 + j * 2 + h2]));
#else
#pragma unroll
        for (int h2 = 0; h2 < 4; ++h2)
          asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
              : "+r"(c[(h2 & 1) * 2]), "+r"(c[(h2 & 1) * 2 + 1]) : "r"(f[i * 4 + h2]), "r"(f[16 + j * 2 + (h2 >> 1)]));
#endif
      }
  }
  unsigned r = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 20000; double mhz = 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  unsigned* sink; cudaMalloc(&sink, 64);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, WARPS * 32, 65536>>>(200, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed\n"); return 1; }
  cudaEventRecord(e0); k<<<sms, WARPS * 32, 65536>>>(steps, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)steps * 32 * WARPS * sms * 4096 / (ms * 1e-3);
  printf("shape %d, %d warps: %.1f TMAC/s (%.1f%% of the m16n8k32 peak)\n", SHAPE, WARPS, rate / 1e12, 100 * rate / (sms * 1024.0 * mhz * 1e6));
  return 0;
}
