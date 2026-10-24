// X-Miner Pearl kernels for NVIDIA Ampere and newer (sm_80+): shared Pearl device code + an int8 tensor-core GEMM
// with the Pearl transcript and a fused jackpot check. Written for X-Miner from the PTX ISA documentation.
//
// GEMM: C = A' * B'^T with A' (M x K) and B'^T (N x K) both int8 in the tiled K-major layout (pearl_common.h),
//   int32 accumulation. Tiles of 128x256: 8 math warps in 2x4, warp tile 64x64 = 4 (m16) x 8 (n8)
//   mma.sync.m16n8k32 accumulators; fragments come from ldmatrix.x4 on XOR-swizzled 128-byte rows (conflict-free).
//   Persistent blocks (one per SM) walk the tiles in grouped order. A 9th warp issues every global -> shared copy
//   (cp.async, 2 stages of 128 k) up to two stages ahead, across tile boundaries, so the next tile loads during the
//   previous tile's epilogue; mbarriers replace __syncthreads. Copy warps start each wave of tiles together, so the
//   blocks sharing A/B slabs stay at the same k and DRAM traffic stays at the one-tile-per-SM minimum.
// Hash tiles follow the accumulator layout: rows {0, 8, ..., 56} x cols {0, 1, 8, 9, ..., 56, 57} (h = 8, w = 16;
//   the host's NVIDIA mining configuration), i.e. exactly the 128 accumulators one lane owns in its warp tile.
//   Every 128 k each lane XORs its 128 running sums into a word x and folds t[c % 16] = rotl(t[c % 16], 13) ^ x,
//   with no cross-lane traffic. After the last fold each lane hashes its transcript (keyed BLAKE3 with a_noise_seed)
//   and tiles at or under the share bound are appended to found[] as (t_rows, t_cols, 16 transcript words).
#define XM_PEARL_ROWMAJOR 1
#include "pearl/pearl_common.h"

namespace {

constexpr int BM = 128, BN = 256, BK = 128;    // tile; k bytes per stage (one Pearl chunk)
constexpr int CH = BK / 16;                    // 16-byte chunks per stage row
constexpr int KS = BK / 32;                    // mma k32 steps per stage
constexpr int WM = 64, WN = 64;                // warp tile
constexpr int CWARPS = (BM / WM) * (BN / WN);  // math warps
constexpr int CTHREADS = CWARPS * 32;
constexpr int THREADS = CTHREADS + 32;  // + the copy warp; 9 warps cap registers at 168 (3 warps on one SMSP)
constexpr int MT = WM / 16;             // m16 fragments per warp
constexpr int STAGE_A = BM * BK, STAGE_B = BN * BK, STAGE_BYTES = STAGE_A + STAGE_B;  // 48 KB per stage
#ifndef XM_PEARL_STAGES
#define XM_PEARL_STAGES 2
#endif
constexpr int STAGES = XM_PEARL_STAGES;  // STAGES x STAGE_BYTES of dynamic shared memory
#ifndef XM_PEARL_XCH
#define XM_PEARL_XCH 4  // independent XOR chains in the fold
#endif
#ifndef XM_PEARL_GROUP
#define XM_PEARL_GROUP 8  // row blocks per grouped-rasterization sweep
#endif
#ifndef XM_PEARL_PSLEEP
#define XM_PEARL_PSLEEP 0  // copy-warp back-off between empty-barrier polls, ns
#endif
#ifndef XM_PEARL_DIAG_L2
#define XM_PEARL_DIAG_L2 0  // developer power diagnostic: every tile reads the first A and B tiles (L2-resident)
#endif
// RTX 3060 notes (2026-10-23; bare harness with CUDA-event timing on Pearl-like data, which draws ~45 W more than
// all-zero matrices at 1500 MHz; "cap" = stock 170 W, where energy per MAC sets the clock and so the rate):
//  - Previous design (grid of tiles, the 8 math warps copying their own stages, __syncthreads per stage): 36.3
//    TMAC/s at a locked 1500 MHz (84% of tensor peak), 43.6 at the cap. Its DRAM traffic drifted between 5 and
//    10 GB per 16384x32768x4096 pass as blocks lost step (DRAM costs ~18 W at 1500 MHz). This design: 38.1 at
//    1500 MHz and 45.3 at the cap with 5.0 GB (L2 hit rate 81%, the minimum for one tile per SM); the miner went
//    from 41.5 to 43.9 TH/s with the XOR chains. Without the wave sync the copy warp's bursts drive DRAM traffic
//    to 12 GB and the gain at the cap disappears.
//  - Fold XOR chains 1/2/4/8: 4 best (+1.6% at the cap). Group rows 4/16 and copy-warp sleeps of 64-2000 ns: slower.
//    Lane-0 arrive predicated inside the asm instead of __syncwarp + if: +0.7%. Releasing the buffer right after
//    the last ldmatrix, or XORing the snapshot during the next chunk's first k-step: spill under the 168 cap.
//  - Previous design: BK 64 x 3/4 stages -10%; L2 prefetch, deferred and ordered-XOR folds, 12 warps (spills to
//    DRAM): all slower; 16 warps of 32x64 -6% (more ldmatrix traffic per mma).
//  - Ceiling at the cap: tensor cores + ldmatrix alone on Pearl-like data 55.8 TMAC/s at 1942 MHz; + fold 51.6.

// Byte offset of (row, 16-byte chunk) inside a 128-byte-row stage, with an XOR swizzle so the 8 row addresses of
// an ldmatrix hit 32 distinct banks.
__device__ __forceinline__ unsigned swz(int row, int chunk) { return row * BK + ((chunk ^ (row & 7)) << 4); }

__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}

// mbarriers (sm_80): the copy warp's copies arrive on full[s] as they land; each math warp arrives on empty[s] once
// it has read stage buffer s.
__device__ __forceinline__ void mbar_init(unsigned bar, unsigned count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
// Arrive from lane 0 only, predicated inside the asm: a C++ branch would split the math loop into basic blocks
// that the scheduler does not move the fold's XORs across.
__device__ __forceinline__ void mbar_arrive_lane0(unsigned bar, int lane) {
  asm volatile("{\n .reg .pred p;\n .reg .b64 st;\n setp.eq.s32 p, %1, 0;\n @p mbarrier.arrive.shared.b64 st, [%0];\n}\n"
               ::"r"(bar), "r"(lane) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cp_async(unsigned bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar) : "memory");
}
template <int SLEEP_NS = 0>  // > 0: back off between polls
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
  unsigned done;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done)
                 : "r"(bar), "r"(parity)
                 : "memory");
    if (done) break;
    if (SLEEP_NS) __nanosleep(SLEEP_NS);
  }
}

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

// Grid: any number of blocks (one per SM is best), 288 threads, STAGES x 48 KB = 96 KB of dynamic shared memory.
// Block b walks tiles t = b + w * gridDim.x (wave w). Requirements: M % 128 == 0, N % 256 == 0, K % 128 == 0.
// wave_ctr: a zeroed word (copy warps start each wave together), or nullptr.
// out != nullptr: debug path writing 16 transcript words per lane tile instead of the check, at index
//   ((t_rows / 64) * 8 + t_rows % 8) * (N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2.
extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr) {
  extern __shared__ __align__(128) int8_t smem[];
  __shared__ __align__(8) unsigned long long bars[2 * STAGES];  // full[0..STAGES), empty[0..STAGES)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int WARPS_N = BN / 64;
  const int gx = N / BN, gy = M / BM, T = gx * gy, G = gridDim.x;
  auto tile_xy = [&](int t, int& bx, int& by) {  // grouped rasterization: GROUP row blocks per column sweep
    const int per = XM_PEARL_GROUP * gx, first = (t / per) * XM_PEARL_GROUP, rows = min(gy - first, XM_PEARL_GROUP);
    by = first + (t % per) % rows;
    bx = (t % per) / rows;
  };
  const size_t SLAB = 16384, RB_BYTES = (size_t)(K / 128) * SLAB;
  const int nk = K / BK;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0]);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  if (tid == 0)
    for (int s = 0; s < STAGES; ++s) mbar_init(bar0 + 8 * s, 32), mbar_init(bar0 + 8 * (STAGES + s), CWARPS);
  __syncthreads();

  if (warp == CWARPS) {
    // Copy warp: lane = (row within a 4-row pass, 16-byte chunk); 96 copies per lane per stage, then one arrive on
    // full[s] that fires when they have all landed. Stage g reuses buffer g % STAGES once every math warp has
    // finished stage g - STAGES.
    constexpr int PR = 32 / CH;  // rows per copy instruction (4 rows x 8 chunks of 16 bytes)
    const int prow = lane / CH, pch = lane % CH;
    unsigned g = 0, need = 0;  // stages issued so far (all tiles); wave starts this block waits for
    for (int t = blockIdx.x, w = 0; t < T; t += G, ++w) {
      int bx, by;
      tile_xy(t, bx, by);
      if (wave_ctr && w > 0) {  // count this block's start of wave w, then wait for every block's
        need += (unsigned)min(G, T - w * G);
        if (lane == 0) {
          atomicAdd(wave_ctr, 1u);
          while (*(volatile unsigned*)wave_ctr < need) __nanosleep(32);
        }
        __syncwarp();
      }
      const int8_t* pa = A + (size_t)(XM_PEARL_DIAG_L2 ? 0 : by) * RB_BYTES + prow * 128 + pch * 16;
      const int8_t* pb = B + (size_t)(XM_PEARL_DIAG_L2 ? 0 : bx) * (BN / 128) * RB_BYTES + prow * 128 + pch * 16;
      for (int kt = 0; kt < nk; ++kt, ++g) {
        const int s = g % STAGES;
        if (g >= STAGES) mbar_wait<XM_PEARL_PSLEEP>(bar0 + 8 * (STAGES + s), (g / STAGES - 1) & 1);
        const unsigned sa = sbase + s * STAGE_BYTES, sb = sa + STAGE_A;
        const int8_t* ga = pa + (size_t)kt * SLAB;
        const int8_t* gb = pb + (size_t)kt * SLAB;
#pragma unroll
        for (int i = 0; i < BM / PR; ++i) cp_async16(sa + swz(prow + PR * i, pch), ga + (PR * i) * 128);
#pragma unroll
        for (int i = 0; i < BN / PR; ++i)
          cp_async16(sb + swz(prow + PR * i, pch), gb + (size_t)((PR * i) >> 7) * RB_BYTES + ((PR * i) & 127) * 128);
        mbar_arrive_cp_async(bar0 + 8 * s);
      }
    }
    return;
  }

  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int a_row = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_row = wn * WN + (lane & 7) + (lane >> 4) * 8;
  // swz(row, 2ks + c) = swz(row, c) ^ (ks << 5) for c in {0, 1}: one base register each instead of KS offsets.
  const unsigned a_off0 = swz(a_row, lane >> 4), b_off0 = swz(b_row, (lane >> 3) & 1);
  unsigned g = 0;  // stages consumed by this block so far
  for (int t = blockIdx.x; t < T; t += G) {
    int bx, by;
    tile_xy(t, bx, by);
    // Two accumulator sets: the running sums live in acc, except across a chunk boundary, where the last k-step
    // writes them into snap (out of place). The fold reads snap while the next chunk's first k-step reads snap and
    // writes acc, so the fold never has to finish before the tensor cores may overwrite what it reads.
    int acc[MT][8][4], snap[MT][8][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) acc[i][j][v] = snap[i][j][v] = 0;
    // This lane's tile transcript. Chunk c updates only word c % 16, so R lives in (L1-cached) local memory with
    // a dynamic index: one load and one store per 128 k instead of 16 registers and a 16-move rotation.
    unsigned R[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) R[w] = 0;
    for (int kt = 0; kt < nk; ++kt, ++g) {  // one 128-k stage = one Pearl chunk
      const int st = g % STAGES;
      mbar_wait(bar0 + 8 * st, (g / STAGES) & 1);
      const unsigned sA = sbase + st * STAGE_BYTES, sB = sA + STAGE_A;
      unsigned x = 0;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        unsigned af[MT][4], bf[8][2];
        const unsigned pa = sA + (a_off0 ^ (ks << 5));
#pragma unroll
        for (int i = 0; i < MT; ++i) ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], pa + i * 16 * BK);
        const unsigned pb = sB + (b_off0 ^ (ks << 5));
#pragma unroll
        for (int j = 0; j < 8; j += 2) ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], pb + j * 8 * BK);
        if (ks == 0) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]);
        } else if (ks == KS - 1) {
          // Last k-step: write the chunk's sums to snap and XOR each row once the next row has been issued, into
          // XCH independent chains (a single chain is 64 dependent LOP3s).
          constexpr int XCH = XM_PEARL_XCH;
          unsigned xs[XCH];
#pragma unroll
          for (int q = 0; q < XCH; ++q) xs[q] = 0;
          auto xor_row = [&](int i) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
              xs[(i * 8 + j) % XCH] ^=
                  (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
          };
#pragma unroll
          for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(snap[i][j], acc[i][j], af[i], bf[j][0], bf[j][1]);
            if (i > 0) xor_row(i - 1);
          }
          xor_row(MT - 1);
#pragma unroll
          for (int q = 0; q < XCH; ++q) x ^= xs[q];
        } else {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
        }
      }
      mbar_arrive_lane0(bar0 + 8 * (STAGES + st), lane);  // done with buffer st
      const unsigned r = R[kt & 15];
      R[kt & 15] = ((r << 13) | (r >> 19)) ^ x;
    }

    const uint32_t t_rows = (uint32_t)(by * BM + wm * WM + (lane >> 2));
    const uint32_t t_cols = (uint32_t)(bx * BN + wn * WN + 2 * (lane & 3));
    if (out) {
      const size_t idx = ((size_t)(t_rows / WM) * 8 + t_rows % 8) * (size_t)(N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2;
#pragma unroll
      for (int w = 0; w < 16; ++w) out[idx * 16 + w] = R[w];
      continue;
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
}
