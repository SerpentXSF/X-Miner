// Microbenchmark 7: math warps generate their own operands in shared memory, vectorised.
// Tile 128 x 384 (8 warps, 48 MMAs per warp per k-step, 192 accumulators per lane). Resident tables, column-major:
// UA[128 cols][128 rows], UB[128 cols][384 rows] (values 0..63). A k-step's operands (512 rows x 32 k = 16 KB) are
// built in 4x4 blocks: for 4 consecutive l, word_l = 0x40404040 + U[col p0(l)][r..r+3] - U[col p1(l)][r..r+3]
// (4 rows of one k), then a 4x4 byte transpose gives 4 row words (4 k of one row), stored row-major in the stage
// buffer that the MMAs read. Each warp builds one eighth of the next k-step, then runs its 48 MMAs.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef GEN
#define GEN 1      // 0: no generation (static stage data): the loop's ceiling
#endif
#ifndef GENSYNC
#define GENSYNC 1
#endif
#ifndef GENWORK
#define GENWORK 1
#endif
#ifndef NACC
#define NACC 48    // MMAs per warp per k-step (48 = 8 x 6)
#endif
__device__ __forceinline__ void mma(int* c, const uint4& a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(256) k(int steps, const uint4* offs, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char smem[];  // UA 16 KB | UB 48 KB | stage 0 16 KB | stage 1 16 KB
  __shared__ __align__(8) unsigned long long bar[4];  // built[0..2), read[0..2)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < 98304; i += blockDim.x) {
    unsigned h = (i + blockIdx.x * 7919u) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    smem[i] = i < 65536 ? (unsigned char)(h & 63) : (unsigned char)(63 + (h & 1) + (int)((h >> 1) & 63) - (int)((h >> 7) & 63));
  }
  const unsigned sb = (unsigned)__cvta_generic_to_shared(smem), b0 = (unsigned)__cvta_generic_to_shared(bar);
  if (threadIdx.x == 0) for (int s = 0; s < 4; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0 + 8 * s), "r"(8u) : "memory");
  __syncthreads();
  int acc[NACC][4];
#pragma unroll
  for (int j = 0; j < NACC; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  const unsigned stage0 = sb + 65536;
  auto generate = [&](int step, int buf) {
    // this warp's 4 iterations: block = (warp * 4 + it) * 32 + lane -> row group rg (4 rows), l group lg (4 k)
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int blk = (warp * 4 + it) * 32 + lane, rg = blk >> 3, lg = blk & 7;
      const bool isA = rg < 32;                                  // row groups 0..31: A's 128 rows, then B's 384
      const unsigned ubase = sb + (isA ? 0 : 16384) + (isA ? rg * 4 : (rg - 32) * 4);
      // offsets for this 4-k group: 8 x u16 for A (p0, p1 per k, premultiplied by 128), 8 x u16 for B (x 384)
      const uint4 o = offs[((step & 255) * 8 + lg) * 2 + (isA ? 0 : 1)];
      const unsigned w[4] = {o.x, o.y, o.z, o.w};
      unsigned v[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        unsigned x, y;
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(x) : "r"(ubase + (w[b] & 0xffff)));
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(y) : "r"(ubase + (w[b] >> 16)));
        v[b] = 0x40404040u + x - y;
      }
      const unsigned dst = stage0 + buf * 16384 + rg * 4 * 32 + lg * 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned t = ((v[0] >> (8 * j)) & 0xff) | (((v[1] >> (8 * j)) & 0xff) << 8) | (((v[2] >> (8 * j)) & 0xff) << 16) | (((v[3] >> (8 * j)) & 0xff) << 24);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dst + j * 32), "r"(t) : "memory");
      }
    }
  };
  // fragment loads: row-major stage, 32 bytes per row. A fragment i (16 rows): lane (g, t) needs rows 16i+g, 16i+g+8,
  // k 4t..4t+3 and 16+4t..: four words. B fragment j: row 128 + 64j' ... : two words.
  const int g = lane >> 2, t = lane & 3;
  for (int s = 0; s < steps; ++s) {
    const int buf = s & 1;
#if GEN
    {  // wait until every warp has built this step's buffer
      unsigned done;
      for (;;) { asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b0 + 8 * buf), "r"((unsigned)(((s - 2) >> 1) & 1)) : "memory"); if (done || s < 2) break; }
    }
#endif
    uint4 af[8]; unsigned bf[6][2];
    const unsigned sa = stage0 + buf * 16384;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned r0 = sa + (16 * i + g) * 32 + 4 * t;
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(af[i].x) : "r"(r0));
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(af[i].y) : "r"(r0 + 8 * 32));
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(af[i].z) : "r"(r0 + 16));
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(af[i].w) : "r"(r0 + 8 * 32 + 16));
    }
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const unsigned c0 = sa + (128 + 64 * j + 8 * warp + g) * 32 + 4 * t;
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(bf[j][0]) : "r"(c0));
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(bf[j][1]) : "r"(c0 + 16));
    }
#if GEN
#if GENSYNC
    // every warp must have read this buffer before anyone rebuilds it
    asm volatile("{\n .reg .pred p;\n .reg .b64 st;\n setp.eq.s32 p, %1, 0;\n @p mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(b0 + 16 + 8 * buf), "r"(lane) : "memory");
    {
      unsigned done;
      for (;;) { asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b0 + 16 + 8 * buf), "r"((unsigned)((s >> 1) & 1)) : "memory"); if (done) break; }
    }
#endif
#if GENWORK
    generate(s + 2, buf);   // step s + 2 goes into the buffer just read
#endif
    asm volatile("{\n .reg .pred p;\n .reg .b64 st;\n setp.eq.s32 p, %1, 0;\n @p mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(b0 + 8 * buf), "r"(lane) : "memory");
#endif
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 6; ++j)
        if (i * 6 + j < NACC) mma(acc[i * 6 + j], af[i], bf[j][0], bf[j][1]);
  }
  unsigned r = 0;
#pragma unroll
  for (int j = 0; j < NACC; ++j) for (int v = 0; v < 4; ++v) r ^= (unsigned)acc[j][v];
  if (r == 0x1234567u) sink[0] = r;
}
int main(int argc, char** argv) {
  int steps = argc > 1 ? atoi(argv[1]) : 40000; double mhz = 2745;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  uint4* offs; unsigned* sink; const int NO = 256 * 8 * 2; cudaMalloc(&offs, NO * 16); cudaMalloc(&sink, 64);
  unsigned* h = (unsigned*)malloc(NO * 16); unsigned x = 4242;
  for (int e = 0; e < NO; ++e) for (int b = 0; b < 4; ++b) {   // word b: lo16 = col(p0) * R, hi16 = col(p1) * R (R = 128 for A, 384 for B)
    x = x * 1664525u + 1013904223u; unsigned c0 = (x >> 9) & 127; x = x * 1664525u + 1013904223u; unsigned c1 = (x >> 9) & 127;
    unsigned R = (e & 1) ? 384 : 128; h[e * 4 + b] = (c0 * R) | ((c1 * R) << 16);
  }
  cudaMemcpy(offs, h, NO * 16, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, 256, 98304>>>(200, offs, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  cudaEventRecord(e0); k<<<sms, 256, 98304>>>(steps, offs, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)steps * NACC * 8 * sms * 4096 / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6;
  printf("generation %s, %d MMAs per warp-step: %.1f TMAC/s = %.1f%% of peak\n", GEN ? "ON" : "off", NACC, rate / 1e12, 100 * rate / peak);
  return 0;
}
