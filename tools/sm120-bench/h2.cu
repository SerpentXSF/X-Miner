// Harness 2: the miner's own kernel (kernels/pearl/pearl.cu, 12-argument form) in any build configuration.
// Checks transcripts via the debug `out` path against a CPU-free golden file, prints a checksum of the found[] hits
// (compare across builds), and times the mining path.  nvcc ... -DTHREADS=<n> [-D kernel flags] h2.cu -o h2
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "pearl/pearl.cu"
#ifdef EXTRA_FILE
#include EXTRA_FILE
#endif
#ifndef H_THREADS
#define H_THREADS 288
#endif
#ifdef H_NOWAVE
#define H_WAVE nullptr
#else
#define H_WAVE wave
#endif
#ifndef H_SMEM
#define H_SMEM 98304
#endif
#ifndef H_DUMMY
#define H_DUMMY 0
#endif
#ifndef KERN
#define KERN xm_pearl_gemm
#endif
static size_t h_tiled(uint32_t row, uint32_t k, uint32_t K) {
#ifdef H_PRESWZ  // each row's 16-byte chunks in the shared-memory swizzle order: chunk c of row r at (c ^ (r & 7))
  return ((size_t)(row >> 7) * (K >> 7) + (k >> 7)) * 16384 + (row & 127) * 128 + ((((k & 127) >> 4) ^ (row & 7)) << 4) + (k & 15);
#else
  return ((size_t)(row >> 7) * (K >> 7) + (k >> 7)) * 16384 + (row & 127) * 128 + (k & 127);
#endif
}
__global__ void dummy_small(unsigned* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[100] += 1; }
__global__ void dummy_wide(unsigned* p) { extern __shared__ unsigned sm[]; if (threadIdx.x == 0) sm[0] = blockIdx.x; if (threadIdx.x == 0 && blockIdx.x == 0) p[100] += sm[0]; }
int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2304, N = argc > 2 ? atoi(argv[2]) : 8192, K = argc > 3 ? atoi(argv[3]) : 2048;
  int reps = argc > 4 ? atoi(argv[4]) : 400; double mhz = argc > 5 ? atof(argv[5]) : 2745; const char* label = argc > 6 ? argv[6] : "kern";
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int sms = p.multiProcessorCount;
  size_t sa = (size_t)M * K, sb = (size_t)N * K, so = (size_t)M * N;
  std::vector<int8_t> ta(sa), tb(sb); std::mt19937 rng(1234);
  auto val = [&]() { return (int8_t)(63 + (rng() & 1) + (int)(rng() & 63) - (int)(rng() & 63)); };
#ifdef H_FRAG  // fragment-major layouts of v2.cu (same values in the same generation order)
  for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k) {
    int rb = r / 64, i = (r % 64) / 16, r16 = r % 16, g = r16 % 8, hi = r16 / 8, ks = k / 32, kk = k % 32, half = kk / 16, t = (kk % 16) / 4;
    ta[((size_t)(rb * (K / 32) + ks) * 4 + i) * 512 + (g * 4 + t) * 16 + (half * 2 + hi) * 4 + kk % 4] = val();
  }
  for (int c = 0; c < N; ++c) for (int k = 0; k < K; ++k) {
    int cb = c / 64, j = (c % 64) / 8, g = c % 8, ks = k / 32, kk = k % 32, half = kk / 16, t = (kk % 16) / 4;
    tb[((size_t)(cb * (K / 32) + ks) * 4 + j / 2) * 512 + (g * 4 + t) * 16 + ((j % 2) * 2 + half) * 4 + kk % 4] = val();
  }
#else
  for (int r = 0; r < M; ++r) for (int k = 0; k < K; ++k) ta[h_tiled(r, k, K)] = val();
  for (int c = 0; c < N; ++c) for (int k = 0; k < K; ++k) tb[h_tiled(c, k, K)] = val();
#endif
  int8_t *A, *B; unsigned *o, *wave, *scratch; uint32_t* found; const int MAXF = 256; const size_t fb = (1 + MAXF * 18) * 4;
  cudaMalloc(&A, sa); cudaMalloc(&B, sb); cudaMalloc(&o, so); cudaMalloc(&wave, 64); cudaMalloc(&found, fb); cudaMalloc(&scratch, 2 << 20); cudaMemset(scratch, 0, 2 << 20);
  cudaMemcpy(A, ta.data(), sa, cudaMemcpyHostToDevice); cudaMemcpy(B, tb.data(), sb, cudaMemcpyHostToDevice);
  if (H_SMEM > 48 * 1024) cudaFuncSetAttribute(KERN, cudaFuncAttributeMaxDynamicSharedMemorySize, H_SMEM);
  cudaFuncSetAttribute(dummy_wide, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  int tiles = (M / 128) * (N / 256), grid = tiles < sms ? tiles : sms;
  Key8 seed; U256w bound; for (int i = 0; i < 8; ++i) seed.w[i] = 0x9e3779b9u * (i + 1), bound.w[i] = i == 7 ? 0x000fffffu : 0xffffffffu;
  auto launch = [&](unsigned* out) { cudaMemsetAsync(wave, 0, 4); KERN<<<grid, H_THREADS, H_SMEM>>>(A, B, out, M, N, K, seed, bound, found, MAXF, H_WAVE, scratch); };
  cudaMemset(o, 0, so); launch(o);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("%s: launch failed: %s\n", label, cudaGetErrorString(cudaGetLastError())); return 1; }
  std::vector<unsigned> r(so / 4); cudaMemcpy(r.data(), o, so, cudaMemcpyDeviceToHost);
  unsigned long long tsum = 0; for (size_t i = 0; i < r.size(); ++i) tsum = tsum * 1099511628211ull + r[i];
  cudaMemset(found, 0, fb); launch(nullptr); cudaDeviceSynchronize();
  std::vector<uint32_t> f(fb / 4); cudaMemcpy(f.data(), found, fb, cudaMemcpyDeviceToHost);
  uint32_t nh = std::min<uint32_t>(f[0], MAXF); std::vector<unsigned long long> recs;
  for (uint32_t h = 0; h < nh; ++h) { unsigned long long x = 0; for (int w = 0; w < 18; ++w) x = x * 1099511628211ull + f[1 + h * 18 + w]; recs.push_back(x); }
  std::sort(recs.begin(), recs.end()); unsigned long long fsum = 0; for (auto x : recs) fsum = fsum * 31 + x;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 20; ++i) launch(nullptr);
  cudaDeviceSynchronize(); cudaEventRecord(e0);
  for (int i = 0; i < reps; ++i) {
#if H_DUMMY == 1
    dummy_small<<<1, 64>>>(scratch);
#elif H_DUMMY == 2
    dummy_small<<<2304, 256>>>(scratch);
#elif H_DUMMY == 3
    dummy_wide<<<36, 256, 98304>>>(scratch);
#elif H_DUMMY == 4
    dummy_wide<<<36, 256, 1024>>>(scratch);
#endif
    launch(nullptr);
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double rate = (double)M * N * K * reps / (ms * 1e-3), peak = sms * 1024.0 * mhz * 1e6;
  printf("%-22s %dx%dx%d: %6.2f TMAC/s = %5.1f%% of peak | transcripts %016llx | hits %u (raw %u) %016llx\n", label, M, N, K, rate / 1e12, 100 * rate / peak, tsum, nh, f[0], fsum);
#ifdef XM_DIAG_WAITS
  cudaMemset(scratch, 0, 2 << 20); launch(nullptr); cudaDeviceSynchronize();
  unsigned dw[5]; cudaMemcpy(dw, scratch + 400000, 20, cudaMemcpyDeviceToHost);
  printf("    waits for data (full): %.1f%% of warp-stages, %.1f polls each | waits for buffer release (empty): %.1f%%, %.1f polls each\n", 100.0 * dw[1] / dw[2], dw[1] ? (double)dw[0] / dw[1] : 0.0, 100.0 * dw[4] / dw[2], dw[4] ? (double)dw[3] / dw[4] : 0.0);
#endif
  return 0;
}
