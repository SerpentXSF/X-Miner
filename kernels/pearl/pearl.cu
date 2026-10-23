// X-Miner Pearl kernels for NVIDIA Ampere and newer (sm_80+): shared Pearl device code + an int8 tensor-core GEMM
// with the Pearl transcript and a fused jackpot check. Written for X-Miner from the PTX ISA documentation.
//
// GEMM: C = A' * B'^T with A' (M x K) and B'^T (N x K) both row-major int8 ("K-major"), int32 accumulation.
//   Block 128x256, 8 warps in 2x4, warp tile 64x64 = 4 (m16) x 8 (n8) mma.sync.m16n8k32 accumulators.
//   K advances BK (128) per stage through a cp.async 2-stage shared-memory pipeline; fragments come from
//   ldmatrix.x4 on XOR-swizzled BK-byte rows (conflict-free).
// Hash tiles follow the accumulator layout: rows {0, 8, ..., 56} x cols {0, 1, 8, 9, ..., 56, 57} (h = 8, w = 16;
//   the host's NVIDIA mining configuration), i.e. exactly the 128 accumulators one lane owns in its warp tile.
//   Every 128 k each lane XORs its 128 running sums into a word x and folds t[c % 16] = rotl(t[c % 16], 13) ^ x,
//   with no cross-lane traffic. After the last fold each lane hashes its transcript (keyed BLAKE3 with a_noise_seed)
//   and tiles at or under the share bound are appended to found[] as (t_rows, t_cols, 16 transcript words).
#define XM_PEARL_ROWMAJOR 1
#include "pearl/pearl_common.h"

namespace {

#ifndef XM_PEARL_BK
#define XM_PEARL_BK 128
#endif
#ifndef XM_PEARL_BM
#define XM_PEARL_BM 128
#endif
#ifndef XM_PEARL_WM
#define XM_PEARL_WM 64  // warp tile height: 64 (8 warps, ~220 registers) or 32 (16 warps, <= 128 registers)
#endif
constexpr int BM = XM_PEARL_BM, BN = 384 - XM_PEARL_BM, BK = XM_PEARL_BK;  // block 128x256 or 256x128  // BK: k bytes per pipeline stage (64 or 128)
constexpr int CH = BK / 16;     // 16-byte chunks per stage row
constexpr int KS = BK / 32;     // mma k32 steps per stage
constexpr int SPC = 128 / BK;   // stages per 128-k Pearl chunk
constexpr int WM = XM_PEARL_WM, WN = 64;
constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
constexpr int MT = WM / 16;      // m16 fragments per warp
constexpr int STAGE_A = BM * BK, STAGE_B = BN * BK, STAGE_BYTES = STAGE_A + STAGE_B;  // 24 KB per 64 k
// Tuned on the RTX 3060 (TH/s at 127 W): BK 64 x 3 stages 39.5 / 64 x 4 38.4 / 128 x 2 41.8; at 170 W:
// warp tiles 64x64 (8 warps) 46.3 vs 32x64 (16 warps, 127 registers) 43.5.
// group rows 4 39.1 / 8 39.5 / 16 38.4 (at BK 64).
// 12 warps (block 192x256, BK 64 x 3 stages, 168-register cap): 18 TH/s - ~45 registers spill, the spills miss L1
// (84 KB of shared memory) and double DRAM traffic. Group rows 4/8/16 at BK 128: equal within noise.
#ifndef XM_PEARL_STAGES
#define XM_PEARL_STAGES 2
#endif
#ifndef XM_PEARL_GROUP
#define XM_PEARL_GROUP 8
#endif
constexpr int STAGES = XM_PEARL_STAGES;  // cp.async pipeline depth: STAGES x STAGE_BYTES of dynamic shared memory

// Byte offset of (row, 16-byte chunk) inside a BK-byte-row stage, with an XOR swizzle so the 8 row addresses of
// an ldmatrix hit 32 distinct banks.
__device__ __forceinline__ unsigned swz(int row, int chunk) {
  if constexpr (BK == 64) return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
  else return row * BK + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  // Not volatile: a pure register-to-register operation, so the compiler may interleave it with the fold work.
  asm(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Out-of-place variant: d = a * b + c with separate input and output registers.
__device__ __forceinline__ void mma_s8_oop(int* d, const int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

}  // namespace

// Grid (N/256, M/128), block 256, STAGES x 48 KB = 96 KB of dynamic shared memory (BK 128). Requirements: M % 128 == 0, N % 256 == 0,
// K % 2048 == 0.
// out != nullptr: debug path writing 16 transcript words per lane tile instead of the check, at index
//   ((t_rows / 64) * 8 + t_rows % 8) * (N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2.
extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm(const int8_t* __restrict__ A,
                                                                    const int8_t* __restrict__ B, unsigned* __restrict__ out,
                                                                    int M, int N, int K, Key8 a_seed, U256w bound,
                                                                    uint32_t* __restrict__ found, uint32_t max_found) {
  extern __shared__ __align__(128) int8_t smem[];  // [stage][A (BM x 64) | B (BN x 64)]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int WARPS_N = BN / 64;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;

  // Grouped rasterization (8 row-blocks per group) so concurrent blocks share A/B tiles in L2.
  const int gx = gridDim.x, gy = gridDim.y, pid = blockIdx.y * gx + blockIdx.x;
  const int G = XM_PEARL_GROUP, per = G * gx, group = pid / per, first = group * G;
  const int rows = min(gy - first, G), in = pid % per;
  const int by = first + in % rows, bx = in / rows;

  // A' and B'^T are tiled (pearl_common.h tiled_offset): row block rb, k-slab s at (rb * K/128 + s) * 16 KB.
  static_assert(BK == 128 && BM % 128 == 0 && BN % 128 == 0, "tiled layout assumes 128-k stages and 128-row blocks");
  const size_t SLAB = 16384, RB_BYTES = (size_t)(K / 128) * SLAB;
  const int8_t* gA = A + (size_t)by * (BM / 128) * RB_BYTES;
  const int8_t* gB = B + (size_t)bx * (BN / 128) * RB_BYTES;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0]);

  // Each thread copies 16-byte chunks: chunk q = tid + 256*i -> row q/CH (+256/CH rows per i), chunk q%CH. The
  // swizzle term does not change with +256/CH rows, so the per-lane shared and global offsets are computed once.
  constexpr int RPP = THREADS / CH;  // rows per copy pass
  const int cp_row = tid / CH, cp_ch = tid % CH;
  const unsigned cp_soff = swz(cp_row, cp_ch);
  const size_t cp_goff = (size_t)cp_row * 128 + cp_ch * 16;
  // Row r of a block lives in row block r / 128 at slab row r % 128.
  auto row_off = [&](int r) { return (size_t)(r >> 7) * RB_BYTES + (size_t)(r & 127) * 128; };
  auto load_stage = [&](int stage, int kt) {
    const unsigned sa = sbase + stage * STAGE_BYTES + cp_soff, sb = sa + STAGE_A;
    const int8_t* ga = gA + cp_goff + (size_t)kt * SLAB;
    const int8_t* gb = gB + cp_goff + (size_t)kt * SLAB;
#pragma unroll
    for (int i = 0; i < BM / RPP; ++i) cp_async16(sa + i * RPP * BK, ga + row_off(i * RPP));
#pragma unroll
    for (int i = 0; i < BN / RPP; ++i) cp_async16(sb + i * RPP * BK, gb + row_off(i * RPP));
    cp_async_commit();
  };

  // ldmatrix addresses: per-lane base (row within the 16/8-row group and swizzled chunk) for each k32 step ks of
  // a stage (k bytes 32ks..32ks+31 = chunks 2ks, 2ks+1), plus constants per fragment.
  const int a_row = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_row = wn * WN + (lane & 7) + (lane >> 4) * 8;
  unsigned a_off[KS], b_off[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    a_off[ks] = swz(a_row, 2 * ks + (lane >> 4));
    b_off[ks] = swz(b_row, 2 * ks + ((lane >> 3) & 1));
  }

  // Two accumulator sets: the running sums live in acc, except across a chunk boundary, where the last k-step
  // writes them into snap (out of place). The fold reads snap while the next chunk's first k-step reads snap and
  // writes acc, so the fold never has to finish before the tensor cores may overwrite what it reads.
  int acc[MT][8][4], snap[MT][8][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int v2 = 0; v2 < 4; ++v2) snap[i][j][v2] = 0;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][j][v] = 0;
  // This lane's tile transcript. Chunk c updates only word c % 16, so R lives in (L1-cached) local memory with a
  // dynamic index: one load and one store per 128 k instead of 16 registers and a 16-move rotation.
  unsigned R[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) R[w] = 0;
  int chunk = 0;

  const int nk = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) load_stage(s, s);
  unsigned x = 0;  // XOR of this lane's running sums, produced during the last k-step of each chunk
  auto stage = [&](int kt, int half) {  // half: position of this stage inside the 128-k chunk (0..SPC-1)
    const int st = kt % STAGES;
    cp_async_wait<STAGES - 2>();           // stage kt has landed (one newer group may still be in flight)
    __syncthreads();                       // stage kt visible; everyone is done with stage kt-1
    if (kt + STAGES - 1 < nk) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    else cp_async_commit();                // keep the group count uniform for wait_group
    const unsigned sA = sbase + st * STAGE_BYTES, sB = sA + STAGE_A;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {      // k32 steps of this stage
      const int pos = half * KS + ks;      // k32 step inside the 128-k chunk (0..3)
      unsigned af[MT][4], bf[8][2];
      // A: matrix q = lane/8 -> rows +0/+8 (q&1), k-bytes +0/+16 (q>>1) = registers a0..a3 of the m16k32 fragment.
      const unsigned pa = sA + a_off[ks];
#pragma unroll
      for (int i = 0; i < MT; ++i) ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], pa + i * 16 * BK);
      // B: matrices (n-tile j, k 0-15), (n-tile j, k 16-31), (n-tile j+1, k 0-15), (n-tile j+1, k 16-31).
      const unsigned pb = sB + b_off[ks];
#pragma unroll
      for (int j = 0; j < 8; j += 2) ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], pb + j * 8 * BK);
      if (pos == 0) {
        // First k-step of a chunk: read the snapshot, write the running sums back to acc.
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]);
      } else if (pos == 3) {
        // Last k-step: write the chunk's sums to snap, and XOR each row as soon as the next row has been issued.
        auto xor_row = [&](int i) {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            x ^= (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
        };
        x = 0;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_s8_oop(snap[i][j], acc[i][j], af[i], bf[j][0], bf[j][1]);
          if (i > 0) xor_row(i - 1);
        }
        xor_row(MT - 1);
      } else {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
      }
    }
  };
  auto fold = [&]() {  // 128 k done: x (filled during the last k-step) is this lane's tile XOR
      const unsigned r = R[chunk & 15];
      R[chunk & 15] = ((r << 13) | (r >> 19)) ^ x;
      ++chunk;
    };
  // SPC stages per iteration = one 128-k Pearl chunk; the fold runs unconditionally at the end of it.
  for (int kt = 0; kt < nk; kt += SPC) {
#pragma unroll
    for (int h = 0; h < SPC; ++h) stage(kt + h, h);
    fold();
  }

  // Epilogue: every lane holds its tile's complete transcript.
  const uint32_t t_rows = (uint32_t)(by * BM + wm * WM + (lane >> 2));
  const uint32_t t_cols = (uint32_t)(bx * BN + wn * WN + 2 * (lane & 3));
  if (out) {
    const size_t idx = ((size_t)(t_rows / WM) * 8 + t_rows % 8) * (size_t)(N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2;
#pragma unroll
    for (int w = 0; w < 16; ++w) out[idx * 16 + w] = R[w];
    return;
  }
  uint32_t words[16];  // separate copy: taking R's address would push it to local memory for the whole kernel
#pragma unroll
  for (int w = 0; w < 16; ++w) words[w] = R[w];
  uint32_t h[8];
  b3_keyed64(a_seed, words, h);
  if (u256_le(h, bound)) {
    uint32_t slot = atomicAdd(&found[0], 1u);
    if (slot < max_found) {
      uint32_t* rec = found + 1 + slot * 18;
      rec[0] = t_rows;
      rec[1] = t_cols;
#pragma unroll
      for (int w = 0; w < 16; ++w) rec[2 + w] = words[w];
    }
  }
}
