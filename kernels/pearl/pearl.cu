// X-Miner Pearl kernels for NVIDIA Ampere and newer (sm_80+): shared Pearl device code + an int8 tensor-core GEMM
// with the Pearl transcript and a fused jackpot check. Written for X-Miner from the PTX ISA documentation.
//
// GEMM: C = A' * B'^T with A' (M x K) and B'^T (N x K) both row-major int8 ("K-major"), int32 accumulation.
//   Block 128x128, 4 warps in 2x2, warp tile 64x64 = 4 (m16) x 8 (n8) mma.sync.m16n8k32 accumulators.
//   K advances 64 per stage through a cp.async 3-stage shared-memory pipeline; fragments come from ldmatrix.x4 on
//   XOR-swizzled 64-byte rows (conflict-free).
// Hash tiles follow the accumulator layout: rows {0, 8, ..., 56} x cols {0, 1, 8, 9, ..., 56, 57} (h = 8, w = 16;
//   the host's NVIDIA mining configuration), i.e. exactly the 128 accumulators one lane owns in its warp tile.
//   Every 128 k each lane XORs its 128 running sums into a word x and folds t[c % 16] = rotl(t[c % 16], 13) ^ x,
//   with no cross-lane traffic. After the last fold each lane hashes its transcript (keyed BLAKE3 with a_noise_seed)
//   and tiles at or under the share bound are appended to found[] as (t_rows, t_cols, 16 transcript words).
#define XM_PEARL_ROWMAJOR 1
#include "pearl/pearl_common.h"

namespace {

constexpr int BM = 128, BN = 128, BK = 64, THREADS = 128;
constexpr int WM = 64, WN = 64;  // 4 warps: 2 (m) x 2 (n)
constexpr int MT = WM / 16;      // m16 fragments per warp
constexpr int STAGE_BYTES = BM * BK;  // per operand (8 KB)
constexpr int STAGES = 3;              // cp.async pipeline depth (3 x 16 KB = 48 KB static shared memory)

// Byte offset of (row, 16-byte chunk) inside a 64-byte-row stage, with an XOR swizzle so the 8 row addresses of
// an ldmatrix hit 32 distinct banks.
__device__ __forceinline__ unsigned swz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

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

// Grid (N/128, M/128), block 128. Requirements: M % 128 == 0, N % 128 == 0, K % 2048 == 0.
// out != nullptr: debug path writing 16 transcript words per lane tile instead of the check, at index
//   ((t_rows / 64) * 8 + t_rows % 8) * (N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2.
extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm(const int8_t* __restrict__ A,
                                                                    const int8_t* __restrict__ B, unsigned* __restrict__ out,
                                                                    int M, int N, int K, Key8 a_seed, U256w bound,
                                                                    uint32_t* __restrict__ found, uint32_t max_found) {
  __shared__ __align__(128) int8_t smem[STAGES][2][STAGE_BYTES];  // [stage][A|B]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;  // wm 0..1, wn 0..1

  // Grouped rasterization (8 row-blocks per group) so concurrent blocks share A/B tiles in L2.
  const int gx = gridDim.x, gy = gridDim.y, pid = blockIdx.y * gx + blockIdx.x;
  const int G = 8, per = G * gx, group = pid / per, first = group * G;
  const int rows = min(gy - first, G), in = pid % per;
  const int by = first + in % rows, bx = in / rows;

  const int8_t* gA = A + (size_t)by * BM * K;
  const int8_t* gB = B + (size_t)bx * BN * K;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0][0][0]);

  // Each thread copies 4 x 16 bytes of A and of B per stage: chunk q = tid + 128*i -> row q/4 (+32 per i),
  // 16-byte chunk q%4. The swizzle depends on (row >> 1) & 3, which +32 rows does not change, so the per-lane
  // shared and global offsets are computed once.
  const int cp_row = tid >> 2, cp_ch = tid & 3;
  const unsigned cp_soff = swz(cp_row, cp_ch);
  const size_t cp_goff = (size_t)cp_row * K + cp_ch * 16, cp_gstep = (size_t)32 * K;
  auto load_stage = [&](int stage, int kt) {
    const unsigned sa = sbase + (stage * 2 + 0) * STAGE_BYTES + cp_soff, sb = sa + STAGE_BYTES;
    const int8_t* ga = gA + cp_goff + (size_t)kt * BK;
    const int8_t* gb = gB + cp_goff + (size_t)kt * BK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      cp_async16(sa + i * 32 * 64, ga + i * cp_gstep);
      cp_async16(sb + i * 32 * 64, gb + i * cp_gstep);
    }
    cp_async_commit();
  };

  // ldmatrix addresses: per-lane base (row within the 16/8-row group and swizzled chunk), plus constants per
  // fragment. ks (0/1) selects k-bytes 0-31 or 32-63 of the 64-byte row: chunk index XOR 2.
  const int a_row = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8;
  const unsigned a_off0 = swz(a_row, lane >> 4), a_off1 = swz(a_row, 2 + (lane >> 4));
  const int b_row = wn * WN + (lane & 7) + (lane >> 4) * 8;
  const unsigned b_off0 = swz(b_row, (lane >> 3) & 1), b_off1 = swz(b_row, 2 + ((lane >> 3) & 1));

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
  // This lane's tile transcript. R is kept rotated so the word updated by the current 128-k chunk is always R[0]:
  // after N chunks, word w is R[(w - N) mod 16].
  unsigned R[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) R[w] = 0;

  const int nk = K / BK;
  load_stage(0, 0);
  load_stage(1, 1);
  unsigned x = 0;  // XOR of this lane's running sums, produced during the last k-step of each chunk
  auto stage = [&](int kt, bool chunk_end) {  // chunk_end: 2nd stage of the 128-k chunk; else the 1st
    const int st = kt % STAGES;
    cp_async_wait<STAGES - 2>();           // stage kt has landed (one newer group may still be in flight)
    __syncthreads();                       // stage kt visible; everyone is done with stage kt-1
    if (kt + 2 < nk) load_stage((kt + 2) % STAGES, kt + 2);
    else cp_async_commit();                // keep the group count uniform for wait_group
    const unsigned sA = sbase + (st * 2 + 0) * STAGE_BYTES, sB = sbase + (st * 2 + 1) * STAGE_BYTES;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {       // two k32 steps per 64-k stage
      unsigned af[MT][4], bf[8][2];
      // A: matrix q = lane/8 -> rows +0/+8 (q&1), k-bytes +0/+16 (q>>1) = registers a0..a3 of the m16k32 fragment.
      const unsigned pa = sA + (ks ? a_off1 : a_off0);
#pragma unroll
      for (int i = 0; i < MT; ++i) ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], pa + i * 16 * 64);
      // B: matrices (n-tile j, k 0-15), (n-tile j, k 16-31), (n-tile j+1, k 0-15), (n-tile j+1, k 16-31).
      const unsigned pb = sB + (ks ? b_off1 : b_off0);
#pragma unroll
      for (int j = 0; j < 8; j += 2) ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], pb + j * 8 * 64);
      if (!chunk_end && ks == 0) {
        // First k-step of a chunk: read the snapshot, write the running sums back to acc.
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]);
      } else if (chunk_end && ks == 1) {
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
      unsigned r0 = ((R[0] << 13) | (R[0] >> 19)) ^ x;
#pragma unroll
      for (int w = 0; w < 15; ++w) R[w] = R[w + 1];
      R[15] = r0;
    };
  // Two 64-k stages per iteration = one 128-k Pearl chunk; the fold runs unconditionally at the end of it.
  for (int kt = 0; kt < nk; kt += 2) {
    stage(kt, false);
    stage(kt + 1, true);
    fold();
  }

  // Epilogue: every lane holds its tile's complete transcript.
  // K / 128 is a multiple of 16 (K % 2048 == 0), so after the last chunk R[w] is word w again.
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
