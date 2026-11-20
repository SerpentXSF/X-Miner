// Microbenchmark 9: in-SM operand generation with only "free" instructions next to the MMAs.
// Tile 128 x 256. Tables at fixed shared addresses so a gather address is an OR, not an add:
//   UA at 0x4000: column c at 0x4000 | c << 7 (128 rows), UB at 0x8000: column c at 0x8000 | c << 8 (256 rows).
// One generation step = 32 lanes x (4 rows) for the same 4 k: every lane loads the same two column offsets per k
// (lane = row group, so loads hit 32 different banks), builds word_k = (U[p0] | 0x40404040) - U[p1] (bytes 64+x-y,
// one SUB per word), transposes the 4x4 bytes with delta swaps (shifts, AND, XOR only) and stores 4 row words.
// Stage rows: 36 bytes (32 k + pad), 148 bytes per 4 rows, so the stores of 32 row groups hit 32 banks too.
// 3 stage buffers (built / read barriers): buffer 0 below 0x4000, buffers 1 and 2 above 0x10000.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef GEN
#define GEN 1
#endif
constexpr unsigned UA = 0x4000, UB = 0x8000, PG = 148, PR = 36, STAGE = 96 * PG;
__device__ __forceinline__ unsigned stage_base(unsigned sb, int buf) { return buf == 0 ? sb : 0x10000u + (buf - 1) * STAGE; }
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
__device__ __forceinline__ void dswap(unsigned& a, unsigned& b, int sh, unsigned m) { const unsigned t = ((a >> sh) ^ b) & m; b ^= t; a ^= t << sh; }
__global__ void __launch_bounds__(256) k(int steps, const uint4* offs, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bar[6];  // built[0..3), read[0..3)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned sb = (unsigned)__cvta_generic_to_shared(smem), b0 = (unsigned)__cvta_generic_to_shared(bar);
  for (unsigned a = 0x4000 + threadIdx.x * 4; a < 0x10000; a += 1024) {   // u values 0..63
    unsigned h = (a + blockIdx.x * 7919u) * 2654435761u; h ^= h >> 15; h *= 2246822519u;
    sts(a, h & 0x3f3f3f3fu);
  }
  if (threadIdx.x == 0) { for (int s = 0; s < 6; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(b0 + 8 * s), "r"(8u) : "memory"); sink[4] = sb; }
  __syncthreads();
  int acc[32][4];
#pragma unroll
  for (int j = 0; j < 32; ++j) for (int v = 0; v < 4; ++v) acc[j][v] = 0;
  // This warp builds 3 of the 24 (row chunk, k group) pieces of a k-step: piece q -> rows chunk q / 8 (0: A's 128
  // rows, 1-2: B's 256), k group q % 8; lane = row group inside the chunk.
  auto generate = [&](int step, unsigned stb) {
#pragma unroll
    for (int it = 0; it < 3; ++it) {
      const int q = warp * 3 + it, chunk = q >> 3, lg = q & 7;
      const unsigned ubase = chunk == 0 ? (UA | lane * 4) : (UB | ((chunk - 1) * 128 + lane * 4));
      const uint4 o = offs[((step & 255) * 8 + lg) * 2 + (chunk == 0 ? 0 : 1)];   // 4 k: lo16 = col(p0) << shift, hi16 = col(p1) << shift
      unsigned v0 = (lds(ubase | (o.x & 0xffff)) | 0x40404040u) - lds(ubase | (o.x >> 16));
      unsigned v1 = (lds(ubase | (o.y & 0xffff)) | 0x40404040u) - lds(ubase | (o.y >> 16));
      unsigned v2 = (lds(ubase | (o.z & 0xffff)) | 0x40404040u) - lds(ubase | (o.z >> 16));
      unsigned v3 = (lds(ubase | (o.w & 0xffff)) | 0x40404040u) - lds(ubase | (o.w >> 16));
      // v_k = 4 rows of k (byte j = row j). Transpose to row words (byte b = k b).
      dswap(v1, v0, 8, 0x00ff00ffu); dswap(v3, v2, 8, 0x00ff00ffu);
      dswap(v2, v0, 16, 0x0000ffffu); dswap(v3, v1, 16, 0x0000ffffu);
      const unsigned dst = stb + (chunk * 32 + lane) * PG + lg * 4;
      sts(dst, v0); sts(dst + PR, v1); sts(dst + 2 * PR, v2); sts(dst + 3 * PR, v3);
    }
  };
  const int g = lane >> 2, t = lane & 3, wm = warp >> 2, wn = warp & 3;
  auto rowaddr = [&](int r) { return (unsigned)((r >> 2) * PG + (r & 3) * PR); };
#if GEN
  generate(0, stage_base(sb, 0)); generate(1, stage_base(sb, 1));
  arrive0(b0, lane); arrive0(b0 + 8, lane);
#endif
  int buf = 0, phase = 0;   // step s uses buffer s % 3; phase = (s / 3) & 1
  for (int s = 0; s < steps; ++s) {
    const unsigned sa = stage_base(sb, buf);
#if GEN
    waitbar(b0 + 8 * buf, phase);
#endif
    uint4 af[4]; unsigned bf[8][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned r0 = sa + rowaddr(wm * 64 + 16 * i + g) + 4 * t, r1 = sa + rowaddr(wm * 64 + 16 * i + g + 8) + 4 * t;
      af[i] = make_uint4(lds(r0), lds(r1), lds(r0 + 16), lds(r1 + 16));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned c0 = sa + rowaddr(128 + wn * 64 + 8 * j + g) + 4 * t;
      bf[j][0] = lds(c0), bf[j][1] = lds(c0 + 16);
    }
#if GEN
    arrive0(b0 + 8 * (3 + buf), lane);                           // this warp has read step s
    const int pb = buf == 0 ? 2 : buf - 1;                       // buffer of step s - 1 = buffer of step s + 2
    if (s > 0) waitbar(b0 + 8 * (3 + pb), buf == 0 ? phase ^ 1 : phase);   // everyone has read step s - 1
    generate(s + 2, stage_base(sb, pb));
    arrive0(b0 + 8 * pb, lane);
#endif
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) mma(acc[i * 8 + j], af[i], bf[j][0], bf[j][1]);
    if (++buf == 3) buf = 0, phase ^= 1;
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
    int sh = (e & 1) ? 8 : 7; h[e * 4 + b] = (c0 << sh) | ((c1 << sh) << 16);
  }
  cudaMemcpy(offs, h, NO * 16, cudaMemcpyHostToDevice);
  const int SMEM = 0x10000 + 2 * STAGE;   // the window must reach the end of buffer 2 (the base is a few hundred bytes in)
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k<<<sms, 256, SMEM>>>(200, offs, sink);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
  cudaEventRecord(e0); k<<<sms, 256, SMEM>>>(steps, offs, sink); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); unsigned sv[5]; cudaMemcpy(sv, sink, 20, cudaMemcpyDeviceToHost);
  double rate = (double)steps * 32 * 8 * sms * 4096 / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6;
  printf("generation %s, tile 128x256: %.1f TMAC/s = %.1f%% of peak (shared base 0x%x, window %d KB)\n", GEN ? "ON" : "off", rate / 1e12, 100 * rate / peak, sv[4], SMEM / 1024);
  return 0;
}
