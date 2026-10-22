// X-Miner Pearl kernels for NVIDIA Ampere and newer (sm_80+): shared Pearl device code + an int8 tensor-core GEMM
// with the Pearl transcript and a fused jackpot check. Written for X-Miner from the PTX ISA documentation.
//
// GEMM: C = A' * B'^T with A' (M x K) and B'^T (N x K) both row-major int8 ("K-major"), int32 accumulation.
//   Block 128x128, 4 warps in 2x2, warp tile 64x64 = 4 (m16) x 8 (n8) mma.sync.m16n8k32 accumulators.
//   K advances 64 per stage through a cp.async double-buffered shared-memory pipeline; fragments come from
//   ldmatrix.x4 on XOR-swizzled 64-byte rows (conflict-free).
// Pearl transcript: every 128 k, each 16x16 hash tile's 256 running sums are XOR-reduced to a word x and folded
//   t[c % 16] = rotl(t[c % 16], 13) ^ x. After the last fold each tile's transcript is hashed (keyed BLAKE3 with
//   a_noise_seed) and tiles at or under the share bound are appended to found[].
#define XM_PEARL_ROWMAJOR 1
#include "pearl/pearl_common.h"

namespace {

constexpr int BM = 128, BN = 128, BK = 64, THREADS = 128;
constexpr int WM = 64, WN = 64;
constexpr int STAGE_BYTES = BM * BK;  // per operand (8 KB)

// Byte offset of (row, 16-byte chunk) inside a 64-byte-row stage, with an XOR swizzle so the 8 row addresses of
// an ldmatrix hit 32 distinct banks.
__device__ __forceinline__ unsigned swz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace

// Grid (N/128, M/128), block 128. Requirements: M % 128 == 0, N % 128 == 0, K % 128 == 0.
// out != nullptr: debug path writing 16 transcript words per 16x16 tile (tile-major) instead of the check.
extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm(const int8_t* __restrict__ A,
                                                                    const int8_t* __restrict__ B, unsigned* __restrict__ out,
                                                                    int M, int N, int K, Key8 a_seed, U256w bound,
                                                                    uint32_t* __restrict__ found, uint32_t max_found) {
  __shared__ __align__(128) int8_t smem[2][2][STAGE_BYTES];  // [stage][A|B]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;

  // Grouped rasterization (8 row-blocks per group) so concurrent blocks share A/B tiles in L2.
  const int gx = gridDim.x, gy = gridDim.y, pid = blockIdx.y * gx + blockIdx.x;
  const int G = 8, per = G * gx, group = pid / per, first = group * G;
  const int rows = min(gy - first, G), in = pid % per;
  const int by = first + in % rows, bx = in / rows;

  const int8_t* gA = A + (size_t)by * BM * K;
  const int8_t* gB = B + (size_t)bx * BN * K;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0][0][0]);

  // Each thread copies 4 x 16 bytes of A and of B per stage: chunk q = tid + 128*i -> row q/4, 16-byte chunk q%4.
  auto load_stage = [&](int stage, int kt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int q = tid + THREADS * i, row = q >> 2, ch = q & 3;
      size_t g = (size_t)row * K + (size_t)kt * BK + ch * 16;
      cp_async16(sbase + (stage * 2 + 0) * STAGE_BYTES + swz(row, ch), gA + g);
      cp_async16(sbase + (stage * 2 + 1) * STAGE_BYTES + swz(row, ch), gB + g);
    }
    cp_async_commit();
  };

  int acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][j][v] = 0;
  unsigned tr[4][4];  // per 16x16 tile: this lane's copy of transcript word (lane % 16)
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) tr[i][j] = 0;

  const int nk = K / BK;
  load_stage(0, 0);
  for (int kt = 0; kt < nk; ++kt) {
    const int st = kt & 1;
    cp_async_wait_all();
    __syncthreads();                       // stage kt visible; everyone is done with stage kt-1
    if (kt + 1 < nk) load_stage(st ^ 1, kt + 1);
    const unsigned sA = sbase + (st * 2 + 0) * STAGE_BYTES, sB = sbase + (st * 2 + 1) * STAGE_BYTES;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {       // two k32 steps per 64-k stage
      unsigned af[4][4], bf[8][2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // matrix q = lane/8: rows +0/+8 (q&1), k-bytes +0/+16 (q>>1) -> registers a0..a3 of the m16k32 fragment
        int r = wm * WM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int ch = ks * 2 + (lane >> 4);
        ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], sA + swz(r, ch));
      }
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        // matrices: (n-tile j, k 0-15), (n-tile j, k 16-31), (n-tile j+1, k 0-15), (n-tile j+1, k 16-31)
        int r = wn * WN + j * 8 + (lane & 7) + (lane >> 4) * 8;
        int ch = ks * 2 + ((lane >> 3) & 1);
        ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], sB + swz(r, ch));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
    if ((kt & 1) == 1) {                   // 128 k done: fold every 16x16 tile
      const int chunk = (kt >> 1) & 15;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int t = 0; t < 4; ++t) {      // hash tile t covers n8 tiles 2t and 2t+1
          unsigned x = 0;
#pragma unroll
          for (int v = 0; v < 4; ++v) x ^= (unsigned)acc[i][2 * t][v] ^ (unsigned)acc[i][2 * t + 1][v];
#pragma unroll
          for (int o = 16; o >= 1; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
          unsigned rot = (tr[i][t] << 13) | (tr[i][t] >> 19);
          tr[i][t] = ((lane & 15) == chunk) ? (rot ^ x) : tr[i][t];
        }
    }
  }

  // Epilogue: lane w (w < 16) holds word w of every tile. Gather tile t's 16 words into lane t through shared
  // memory (free now), then lanes 0..15 hash one tile each.
  __syncthreads();
  unsigned* scratch = reinterpret_cast<unsigned*>(&smem[0][0][0]) + warp * 16 * 16;
  if (lane < 16) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int t = 0; t < 4; ++t) scratch[(i * 4 + t) * 16 + lane] = tr[i][t];
  }
  __syncwarp();
  const int tiles_n = N / 16;
  if (lane < 16) {
    const int i = lane >> 2, t = lane & 3;
    uint32_t words[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) words[w] = scratch[lane * 16 + w];
    uint32_t tile = (uint32_t)((by * BM / 16 + wm * 4 + i) * tiles_n + (bx * BN / 16 + wn * 4 + t));
    if (out) {
#pragma unroll
      for (int w = 0; w < 16; ++w) out[(size_t)tile * 16 + w] = words[w];
      return;
    }
    uint32_t h[8];
    b3_keyed64(a_seed, words, h);
    if (u256_le(h, bound)) {
      uint32_t slot = atomicAdd(&found[0], 1u);
      if (slot < max_found) {
        uint32_t* rec = found + 1 + slot * 17;
        rec[0] = tile;
#pragma unroll
        for (int w = 0; w < 16; ++w) rec[1 + w] = words[w];
      }
    }
  }
}
