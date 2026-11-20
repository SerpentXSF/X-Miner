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

// Pascal, Volta and Turing (sm_61-75) have no m16n8k32 int8 mma, cp.async or mbarriers: they run the grid-of-tiles
// kernel at the end of this file, with dp4a (SIMT) or, on Turing, int8 mma.m8n8k16. XM_PEARL_SIMT=1 forces the dp4a
// variant and XM_PEARL_TURING=1 the m8n8k16 one on newer GPUs (developer checks through --selftest; the host must
// launch them the same way, see src/algo/pearl/pearl_gpu.cpp).
#if defined(XM_PEARL_SIMT) || defined(XM_PEARL_TURING) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
#define XM_PEARL_USE_SIMT 1
#else
#define XM_PEARL_USE_SIMT 0
#endif
#if !defined(XM_PEARL_SIMT) && (defined(XM_PEARL_TURING) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750))
#define XM_PEARL_LEGACY_TC 1
#else
#define XM_PEARL_LEGACY_TC 0
#endif

#if !XM_PEARL_USE_SIMT
#ifdef XM_DIAG_VOLATILE
#define XM_DIAG_VOL volatile
#else
#define XM_DIAG_VOL
#endif
namespace {

// Two stage sizes, one entry point each (the host times both at start-up and mines with the faster one):
//  - xm_pearl_gemm: 2 stages of 128 k (one Pearl chunk each); the copy warp runs one chunk ahead.
//  - xm_pearl_gemm_bk64: 4 stages of 64 k (half a chunk each) in the same 96 KB; the copy warp runs 1.5 chunks
//    ahead, which hides the longer DRAM latency of a locked low memory clock. RTX 3060 at 100 W: memory at stock
//    37.9 vs 35.9 TH/s, memory locked at 5001 MHz 31.1 vs 35.9.
constexpr int BM = 128, BN = 256;              // tile
constexpr int WM = 64, WN = 64;                // warp tile
constexpr int CWARPS = (BM / WM) * (BN / WN);  // math warps
constexpr int CTHREADS = CWARPS * 32;
#ifndef XM_PEARL_CPW
#define XM_PEARL_CPW 1  // copy warps (1, 2 or 4): they share each stage's copies; up to 12 warps keep the 168-register cap
#endif
constexpr int CPW = XM_PEARL_CPW;
#ifndef XM_PEARL_HW
#define XM_PEARL_HW 0  // 1: a hash warp does the per-tile BLAKE3 + bound check, so math warps never pause for it
#endif
constexpr int HW = XM_PEARL_HW;
constexpr int THREADS = CTHREADS + 32 * CPW + 32 * HW;  // + the copy warps; 9-12 warps cap registers at 168 (3 warps on one SMSP)
constexpr int MT = WM / 16;             // m16 fragments per warp
constexpr int SMEM_BYTES = 2 * 128 * (BM + BN);  // 96 KB of dynamic shared memory, either stage size
#ifndef XM_PEARL_XCH
#define XM_PEARL_XCH 4  // independent XOR chains in the fold
#endif
#ifndef XM_PEARL_FOLD_LATE
#define XM_PEARL_FOLD_LATE 0  // 1: fold a chunk's sums during the next chunk's first k-step (no wait on the last MMAs)
#endif
#ifndef XM_PEARL_PFB
#define XM_PEARL_PFB 0  // > 0: prefetch B' this many tile columns ahead into L2 (cache-blocked passes)
#endif
#ifndef XM_PEARL_APPLY_SMEM
#define XM_PEARL_APPLY_SMEM 0  // fused mining: stage the noise inputs through shared memory (cp.async); no gain measured
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
//  - 2026-10-25, miner at the cap (44.3 TH/s at ~1757 MHz; SRB 3.7.x 48.0 at ~1796 MHz, i.e. 93% vs our 88% of
//    tensor peak per clock): BK 64 with 3 or 4 stages and this copy warp -4.5% / -4.0%; copy-warp sleeps of 16 and
//    32 ns -0.2% / -1.9%; 2 or 8 XOR chains -0.5%; group rows 4 / 16 -0.5% / -1.2%. Zero DRAM traffic (DIAG_L2,
//    wrong results) is only +2%, so DRAM is not the lever. Larger K amortizes the per-tile epilogue: XM_PEARL_K
//    8192 +1.1%, 16384 +1.6% (share proofs grow with K).
//  - 2026-10-25 night (K 8192, 44.7 TH/s at the cap): Nsight source sampling puts ~4.5% of math-warp time in the
//    full-barrier wait and ~8% of IMMA samples on ldmatrix results (short scoreboard); the copy warp waits on empty
//    barriers >50% of its time. Tried: starting warps 4..7 up to 2 us late (phase skew between the two math warps
//    of a scheduler) +0.1%; no wave sync -4.6%; per-k32-slice full/empty barriers: waits per slice or a release
//    right after the slice's ldmatrix spill (17.6 / similar), a release after the slice's MMAs 39.6 (32-byte copy
//    rows). All reverted.
//  - 2026-11-03 (45.3 TH/s at the cap): Nsight at base clock: tensor pipe 90.8% busy, issue 17%, stalls wait 3.1,
//    short scoreboard 1.3, branch resolving 1.0 per issue. Unrolling the stage loop by 2 / 4 (fewer loop branches)
//    spills under the 168 cap: 35.4 / 32.4 TH/s. Prefetching the next k-step's fragments needs ~32 more registers,
//    which only a design without the copy warp (8 warps, 255 cap) would have; that design measured slower above.

// Byte offset of (row, 16-byte chunk) inside a 128-byte-row stage, with an XOR swizzle so the 8 row addresses of
// an ldmatrix hit 32 distinct banks.
// 64-byte rows (BK 64): two rows per 128-byte line, so the swizzle uses (row / 2) % 4 and the 8 rows of an ldmatrix
// still land in 8 distinct 16-byte bank groups.
template <int BK>
__device__ __forceinline__ unsigned swz(int row, int chunk) {
  return BK == 128 ? row * BK + ((chunk ^ (row & 7)) << 4) : row * BK + ((chunk ^ ((row >> 1) & 3)) << 4);
}

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

// Developer diagnostic: like mbar_wait, returns the number of failed polls.
__device__ __forceinline__ unsigned mbar_wait_count(unsigned bar, unsigned parity) {
  unsigned done, n = 0;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done)
                 : "r"(bar), "r"(parity)
                 : "memory");
    if (done) break;
    ++n;
  }
  return n;
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int* c, const unsigned* a, unsigned b0, unsigned b1) {
  // Not volatile: a pure register-to-register operation, so the compiler may interleave it with the fold work.
  asm XM_DIAG_VOL(
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

// Grid: any number of blocks (one per SM is best), 288 threads, 96 KB of dynamic shared memory.
// Block b walks tiles t = b + w * gridDim.x (wave w). Requirements: M % 128 == 0, N % 256 == 0, K % 128 == 0.
// wave_ctr: a zeroed word (copy warps start each wave together), or nullptr.
// out != nullptr: debug path writing 16 transcript words per lane tile instead of the check, at index
//   ((t_rows / 64) * 8 + t_rows % 8) * (N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2.
template <int BK>  // k bytes per shared-memory stage: 128 (one Pearl chunk) or 64 (half a chunk)
__device__ __forceinline__ void pearl_gemm_body(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    const Key8& a_seed, const U256w& bound, uint32_t* __restrict__ found, uint32_t max_found,
    unsigned* __restrict__ wave_ctr, unsigned* __restrict__ scratch) {
  static_assert(BK == 128 || BK == 64, "stage size");
  constexpr int CS = 128 / BK;  // stages per Pearl chunk
  constexpr int CH = BK / 16;   // 16-byte chunks per stage row
  constexpr int KS = BK / 32;   // mma k32 steps per stage
  constexpr int STAGE_A = BM * BK, STAGE_B = BN * BK, STAGE_BYTES = STAGE_A + STAGE_B;
  constexpr int STAGES = SMEM_BYTES / STAGE_BYTES;  // 2 x 128 k or 4 x 64 k
  extern __shared__ __align__(128) int8_t smem[];
  __shared__ __align__(8) unsigned long long bars[2 * STAGES + 2];  // full[0..STAGES), empty[0..STAGES), transcripts[0..2)
  __shared__ volatile unsigned hdone;  // tiles of this block the hash warp has finished
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
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(bar0 + 8 * s, 32 * CPW), mbar_init(bar0 + 8 * (STAGES + s), CWARPS);
    mbar_init(bar0 + 8 * (2 * STAGES), CWARPS), mbar_init(bar0 + 8 * (2 * STAGES + 1), CWARPS);
    hdone = 0;
  }
  __syncthreads();

  // Scratch (global): per block two buffers of CWARPS * 32 lanes x 16 transcript words (64 bytes per lane).
  const size_t HBUF = (size_t)CWARPS * 32 * 4;  // uint4 per buffer
  if (HW && warp == CWARPS + CPW) {
    // Hash warp: tile n of this block lands in buffer n & 1; when all math warps have arrived, hash each warp's 32
    // lane tiles (keyed BLAKE3 with a_noise_seed), record the ones under the bound, then release the buffer.
    if (out) return;
    unsigned n = 0;
    for (int t = blockIdx.x; t < T; t += G, ++n) {
      int bx, by;
      tile_xy(t, bx, by);
      const int p = n & 1;
      mbar_wait<64>(bar0 + 8 * (2 * STAGES + p), (n >> 1) & 1);
      const uint4* src = reinterpret_cast<const uint4*>(scratch) + ((size_t)blockIdx.x * 2 + p) * HBUF + lane * 4;
      for (int w = 0; w < CWARPS; ++w) {
        uint32_t words[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 q = src[(size_t)w * 128 + i];
          words[4 * i] = q.x, words[4 * i + 1] = q.y, words[4 * i + 2] = q.z, words[4 * i + 3] = q.w;
        }
        uint32_t h[8];
        b3_keyed64(a_seed, words, h);
        if (u256_le(h, bound)) {
          uint32_t slot = atomicAdd(&found[0], 1u);
          if (slot < max_found) {
            uint32_t* rec = found + 1 + slot * 18;
            rec[0] = (uint32_t)(by * BM + (w / WARPS_N) * WM + (lane >> 2));
            rec[1] = (uint32_t)(bx * BN + (w % WARPS_N) * WN + 2 * (lane & 3));
#pragma unroll
            for (int i = 0; i < 16; ++i) rec[2 + i] = words[i];
          }
        }
      }
      hdone = n + 1;
    }
    return;
  }
  if (warp >= CWARPS) {
    const int cw = warp - CWARPS;  // copy warp cw issues every CPW-th row group
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
          if (cw == 0) atomicAdd(wave_ctr, 1u);
          while (*(volatile unsigned*)wave_ctr < need) __nanosleep(32);
        }
        __syncwarp();
      }
#if XM_PEARL_PFB
      // Cache-blocked passes (A' resident in L2, B' streamed once per pass): every tile of column bx pulls its share
      // of column bx + XM_PEARL_PFB's B' block (2 * RB_BYTES) from DRAM into L2, so the first tiles of that column do
      // not stall on it. The column's rows-in-group tiles cover the block between them.
      if (cw == 0 && bx + XM_PEARL_PFB < gx) {
        const int lines = (int)(2 * RB_BYTES / 128), per_tile = (lines + XM_PEARL_GROUP - 1) / XM_PEARL_GROUP;
        const int8_t* q = B + (size_t)(bx + XM_PEARL_PFB) * (BN / 128) * RB_BYTES;
        const int l0 = (by % XM_PEARL_GROUP) * per_tile;
        for (int l = lane; l < per_tile; l += 32)
          if (l0 + l < lines) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(q + (size_t)(l0 + l) * 128));
      }
#endif
      const int8_t* pa = A + (size_t)(XM_PEARL_DIAG_L2 ? 0 : by) * RB_BYTES + prow * 128 + pch * 16;
      const int8_t* pb = B + (size_t)(XM_PEARL_DIAG_L2 ? 0 : bx) * (BN / 128) * RB_BYTES + prow * 128 + pch * 16;
      for (int kt = 0; kt < nk; ++kt, ++g) {
        const int s = g % STAGES;
        if (g >= STAGES) mbar_wait<XM_PEARL_PSLEEP>(bar0 + 8 * (STAGES + s), (g / STAGES - 1) & 1);
        const unsigned sa = sbase + s * STAGE_BYTES, sb = sa + STAGE_A;
        // Stage kt: bytes (kt % CS) * BK.. of each 128-byte row of slab kt / CS.
#ifdef XM_DIAG_NOBAR
        if (g >= STAGES) return;
#endif
#ifdef XM_DIAG_NOCOPY  // developer timing: stage buffers are filled once per block, later stages only signal
        if (g >= STAGES) {
          asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar0 + 8 * s) : "memory");
          continue;
        }
#endif
        const size_t koff = (size_t)(kt / CS) * SLAB + (kt % CS) * BK;
        const int8_t* ga = pa + koff;
        const int8_t* gb = pb + koff;
#pragma unroll
        for (int i = 0; i < 0; ++i) {}
        // One specialised copy list per copy warp: every address offset is a compile-time constant (a run-time warp
        // index in the row computation slowed the single copy warp by 17% on an RTX 5060 Ti).
        auto copies = [&](const int CW) {
#pragma unroll
          for (int i = CW; i < BM / PR; i += CPW) cp_async16(sa + swz<BK>(prow + PR * i, pch), ga + (PR * i) * 128);
#pragma unroll
          for (int i = CW; i < BN / PR; i += CPW)
            cp_async16(sb + swz<BK>(prow + PR * i, pch), gb + (size_t)((PR * i) >> 7) * RB_BYTES + ((PR * i) & 127) * 128);
        };
        if (CPW == 1 || cw == 0) copies(0);
        else if (cw == 1) copies(1);
        else if (cw == 2) copies(2);
        else copies(3);
        mbar_arrive_cp_async(bar0 + 8 * s);
      }
    }
    return;
  }

  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int a_row = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_row = wn * WN + (lane & 7) + (lane >> 4) * 8;
  // swz(row, 2ks + c) = swz(row, c) ^ (ks << 5) for c in {0, 1}: one base register each instead of KS offsets.
  const unsigned a_off0 = swz<BK>(a_row, lane >> 4), b_off0 = swz<BK>(b_row, (lane >> 3) & 1);
  unsigned dw_polls = 0, dw_late = 0;  // XM_DIAG_WAITS
  unsigned g = 0;  // stages consumed by this block so far
  unsigned nt = 0;  // tiles finished by this block so far
  for (int t = blockIdx.x; t < T; t += G, ++nt) {
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
    for (int kc = 0; kc < nk; kc += CS) {  // one Pearl chunk (128 k) = CS stages
     unsigned x = 0;
#pragma unroll
     for (int cs = 0; cs < CS; ++cs, ++g) {
      const int st = g % STAGES;
#ifdef XM_DIAG_WAITS
      { const unsigned nw = mbar_wait_count(bar0 + 8 * st, (g / STAGES) & 1); dw_polls += nw; dw_late += nw != 0; }
#elif !defined(XM_DIAG_NOBAR)
      mbar_wait(bar0 + 8 * st, (g / STAGES) & 1);
#else
      if (g < STAGES) mbar_wait(bar0 + 8 * st, 0);
#endif
      const unsigned sA = sbase + st * STAGE_BYTES, sB = sA + STAGE_A;
      unsigned af[MT][4], bf[8][2];
#pragma unroll
      for (int ks0 = 0; ks0 < KS; ++ks0) {
        const int ks = cs * KS + ks0;  // k-step within the chunk (0..3); ks0 within this stage
        constexpr int KC = 128 / 32;   // k-steps per chunk
#ifdef XM_DIAG_NOLD
        static_assert(BK == 128 || BK == 64, "");
        if (ks0 == 0) {
#endif
        const unsigned pa = sA + (a_off0 ^ (ks0 << 5));
#pragma unroll
        for (int i = 0; i < MT; ++i) ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], pa + i * 16 * BK);
        const unsigned pb = sB + (b_off0 ^ (ks0 << 5));
#pragma unroll
        for (int j = 0; j < 8; j += 2) ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], pb + j * 8 * BK);
#ifdef XM_DIAG_NOLD
        }
#endif
        if (ks == 0) {
#if XM_PEARL_FOLD_LATE
          // Late fold: XOR the previous chunk's sums (snap, written a whole k-step ago, so never a wait for the tensor
          // cores) right before the MMAs that consume them, then fold them into the transcript.
          unsigned xs[XM_PEARL_XCH];
#pragma unroll
          for (int q = 0; q < XM_PEARL_XCH; ++q) xs[q] = 0;
#pragma unroll
          for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
              xs[(i * 8 + j) % XM_PEARL_XCH] ^=
                  (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]);
          }
          unsigned xp = 0;
#pragma unroll
          for (int q = 0; q < XM_PEARL_XCH; ++q) xp ^= xs[q];
          {
            const int cp = (kc / CS - 1) & 15;  // previous chunk's transcript word (no-op for the first chunk)
            const unsigned r = R[cp], nr = ((r << 13) | (r >> 19)) ^ xp;
            R[cp] = kc > 0 ? nr : r;
          }
#else
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]);
#endif
#if XM_PEARL_FOLD_LATE
        } else if (ks == KC - 1) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(snap[i][j], acc[i][j], af[i], bf[j][0], bf[j][1]);
#endif
        } else if (ks == KC - 1) {
          // Last k-step: write the chunk's sums to snap and XOR each row once the next row has been issued, into
          // XCH independent chains (a single chain is 64 dependent LOP3s).
          constexpr int XCH = XM_PEARL_XCH;
          unsigned xs[XCH];
#pragma unroll
          for (int q = 0; q < XCH; ++q) xs[q] = 0;
          auto xor_row = [&](int i) {
#ifdef XM_DIAG_FOLDMIN  // developer timing: fold one accumulator per row instead of 32
            xs[i % XCH] ^= (unsigned)snap[i][0][0] ^ (unsigned)snap[i][7][3];
            return;
#endif
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
#ifndef XM_DIAG_NOBAR
      mbar_arrive_lane0(bar0 + 8 * (STAGES + st), lane);  // done with buffer st
#endif
     }
#if !XM_PEARL_FOLD_LATE
      const int c = kc / CS;  // chunk index
      const unsigned r = R[c & 15];
      R[c & 15] = ((r << 13) | (r >> 19)) ^ x;
#endif
    }
#if XM_PEARL_FOLD_LATE
    {  // the last chunk's fold (once per tile)
      unsigned xl = 0;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          xl ^= (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
      const int cl = (nk / CS - 1) & 15;
      const unsigned r = R[cl];
      R[cl] = ((r << 13) | (r >> 19)) ^ xl;
    }
#endif

    const uint32_t t_rows = (uint32_t)(by * BM + wm * WM + (lane >> 2));
    const uint32_t t_cols = (uint32_t)(bx * BN + wn * WN + 2 * (lane & 3));
    if (out) {
      const size_t idx = ((size_t)(t_rows / WM) * 8 + t_rows % 8) * (size_t)(N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2;
#pragma unroll
      for (int w = 0; w < 16; ++w) out[idx * 16 + w] = R[w];
      continue;
    }
    if (HW) {
      // Hand the transcript to the hash warp: buffer nt & 1, free once the hash warp has finished tile nt - 2.
      while ((int)hdone < (int)nt - 1) {}
      uint4* dst = reinterpret_cast<uint4*>(scratch) + ((size_t)blockIdx.x * 2 + (nt & 1)) * HBUF + (warp * 32 + lane) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = make_uint4(R[4 * i], R[4 * i + 1], R[4 * i + 2], R[4 * i + 3]);
      mbar_arrive_lane0(bar0 + 8 * (2 * STAGES + (nt & 1)), lane);
      continue;
    }
    uint32_t words[16];  // separate copy: taking R's address would push it to local memory for the whole kernel
#pragma unroll
    for (int w = 0; w < 16; ++w) words[w] = R[w];
    uint32_t h[8];
#ifdef XM_DIAG_NOHASH
    unsigned xx = 0;
#pragma unroll
    for (int w = 0; w < 16; ++w) xx += words[w] * (2 * w + 1);
#pragma unroll
    for (int w = 0; w < 8; ++w) h[w] = xx == 0x12345u ? 0u : 0xffffffffu;
#else
    b3_keyed64(a_seed, words, h);
#endif
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
#ifdef XM_DIAG_WAITS
  if (lane == 0 && scratch) atomicAdd(scratch + 400000, dw_polls), atomicAdd(scratch + 400001, dw_late), atomicAdd(scratch + 400002, g);
#endif
}

extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr,
    unsigned* __restrict__ scratch) {
  pearl_gemm_body<128>(A, B, out, M, N, K, a_seed, bound, found, max_found, wave_ctr, scratch);
}

extern "C" __global__ void __launch_bounds__(THREADS) xm_pearl_gemm_bk64(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr,
    unsigned* __restrict__ scratch) {
  pearl_gemm_body<64>(A, B, out, M, N, K, a_seed, bound, found, max_found, wave_ctr, scratch);
}


// ---------------------------------------------------------------------------------------------------------------
// xm_pearl_gemm_w8: the same GEMM without a copy warp. 8 math warps (255 registers instead of the 168 that 9 warps
// allow): each issues one eighth of the next stage's cp.async copies itself, part-way into the current stage, once
// every warp has released the buffer it refills. Late fold: a chunk's sums are XORed during the next chunk's first
// k-step, right before the MMAs that consume them, so the fold never waits for the tensor cores.
// Needs the pre-swizzled tiled layout (XM_PEARL_PRESWZ: each row's 16-byte chunks stored in the shared-memory swizzle
// order), which makes a stage a straight 48 KB copy. RTX 5060 Ti, 2304x8192x2048 in L2, locked 2745 MHz, bare harness:
// 89.8% of the tensor-core limit against 87.4% for xm_pearl_gemm.
#define V5_PRESWZ 1
#ifndef V5_COPY_AT
#define V5_COPY_AT 1
#endif
#ifndef V5_COPY_ROW
#define V5_COPY_ROW 3
#endif
namespace v5 {
constexpr int STAGE_A = 128 * 128, STAGE_B = 256 * 128, STAGE_BYTES = STAGE_A + STAGE_B, STAGES = 2;
}

// One pass over all tiles; every thread of the block calls it (it starts with a block-wide barrier).
#ifndef XM_PEARL_HASH_INLINE
#define XM_PEARL_HASH_INLINE 0  // G functions between the next tile's MMAs: slower (adds and rotates hold the MMAs back)
#endif
#ifndef XM_PEARL_HASH_STAGGER
#define XM_PEARL_HASH_STAGGER 0  // slower: a warp that hashes mid-tile holds the whole block back at the stage barriers
#endif
#ifndef W8_SLEEP_DATA
#define W8_SLEEP_DATA 0  // ns a math warp sleeps between polls for a stage's data
#endif
#ifndef W8_SLEEP_FREE
#define W8_SLEEP_FREE 0  // ... and for the release of the buffer it refills
#endif
#ifndef W8_EARLY_RELEASE
#define W8_EARLY_RELEASE 0  // release a buffer after its last ldmatrix instead of at the next stage: no gain measured
#endif
#ifndef W8_FRAG_AHEAD
#define W8_FRAG_AHEAD 0  // +1.5% in the bare harness and +0.9% memory-unlocked, -1.8% in the fused path: opt-in
#endif
#ifndef W8_PIN_LOADS
#define W8_PIN_LOADS 0  // slower: the extra block boundary breaks the fold interleaving
#endif
#ifndef W8_DEEP
#define W8_DEEP 0  // refill two stages ahead: the buffer-release wait at the last k-step dominates (80%)
#endif
#ifndef W8_RELEASE_KS2
#define W8_RELEASE_KS2 0  // release at k-step 2 (needs W8_FRAG_AHEAD): fewer waits yet slower
#endif
#ifndef W8_PRELOAD
#define W8_PRELOAD 0  // also load the next stage's first fragments at the last k-step (spills: slower)
#endif
#ifndef W8_DIAG_NOCOPY
#define W8_DIAG_NOCOPY 0
#endif
#ifndef W8_LATE
#define W8_LATE(w) ((w) >= 4)  // which warp of each scheduler pair hashes mid-tile
#endif
#ifndef W8_DIAG_NOHSTEP
#define W8_DIAG_NOHSTEP 0
#endif
struct W8NoHash { char c; };
struct W8Hash { char c[2]; };
// BLAKE3 round structure for the inline hash: G q of a round works on state words (a, b, c, d) and message words
// sched(r, 2q), sched(r, 2q + 1), where the schedule of round r is the permutation applied r times.
__device__ constexpr int w8_ga(int q) { return q & 3; }
__device__ constexpr int w8_gb(int q) { return q < 4 ? 4 + q : 4 + ((q - 4 + 1) & 3); }
__device__ constexpr int w8_gc(int q) { return q < 4 ? 8 + q : 8 + ((q - 4 + 2) & 3); }
__device__ constexpr int w8_gd(int q) { return q < 4 ? 12 + q : 12 + ((q - 4 + 3) & 3); }
// kW8Sched[r][i]: message word used at position i of round r (BLAKE3's permutation applied r times). A literal table:
// a recursive constexpr function is evaluated at run time in device code.
__device__ __forceinline__ int w8_sched(int r, int i) {
  constexpr unsigned char t[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};
  return t[r][i];
}

__device__ __forceinline__ void w8_pass(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    const Key8& a_seed, const U256w& bound, uint32_t* __restrict__ found, uint32_t max_found,
    unsigned* __restrict__ scratch) {
  using namespace v5;
  extern __shared__ __align__(128) int8_t smem[];
  __shared__ __align__(8) unsigned long long bars[2 * STAGES];  // full[0..2), empty[0..2)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int gx = N / 256, gy = M / 128, T = gx * gy, G = gridDim.x, nk = K / 128;
  auto tile_xy = [&](int t, int& bx, int& by) {
    const int per = XM_PEARL_GROUP * gx, first = (t / per) * XM_PEARL_GROUP, rows = min(gy - first, XM_PEARL_GROUP);
    by = first + (t % per) % rows;
    bx = (t % per) / rows;
  };
  const size_t SLAB = 16384, RB = (size_t)(K / 128) * SLAB;
  const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0]);
  const unsigned bar0 = (unsigned)__cvta_generic_to_shared(&bars[0]);
  if (tid == 0)
    for (int s = 0; s < STAGES; ++s) mbar_init(bar0 + 8 * s, 256), mbar_init(bar0 + 8 * (STAGES + s), 8);
  __syncthreads();

  // This warp's share of one stage's copies: instruction i covers 4 rows x 128 k (lane = row within the 4, 16-byte
  // chunk); warp w issues the instructions i = w (mod 8): 4 of A's 32 and 8 of B's 64.
  const int prow = lane >> 3, pch = lane & 7;
  auto issue_w = [&](const int W, unsigned sa, const int8_t* ga, const int8_t* gb) {
#if V5_PRESWZ  // inputs already hold each row's 16-byte chunks in swizzled order: a stage is a straight copy
#pragma unroll
    for (int i = W; i < 32; i += 8) cp_async16(sa + i * 512, ga + i * 512);
#pragma unroll
    for (int i = W; i < 64; i += 8) cp_async16(sa + STAGE_A + i * 512, gb + (size_t)(i >> 5) * RB + (i & 31) * 512);
#else
#pragma unroll
    for (int i = W; i < 32; i += 8) cp_async16(sa + swz<128>(prow + 4 * i, pch), ga + (4 * i) * 128);
#pragma unroll
    for (int i = W; i < 64; i += 8)
      cp_async16(sa + STAGE_A + swz<128>(prow + 4 * i, pch), gb + (size_t)((4 * i) >> 7) * RB + ((4 * i) & 127) * 128);
#endif
  };
  bool diag_copy = true;
  auto issue = [&](int s, int bx, int by, int kt) {  // stage kt of tile (bx, by) into buffer s, then arrive on full[s]
    const unsigned sa = sbase + s * STAGE_BYTES + (V5_PRESWZ ? lane * 16 : 0);
    const int8_t* ga = A + (size_t)by * RB + (size_t)kt * SLAB + (V5_PRESWZ ? lane * 16 : prow * 128 + pch * 16);
    const int8_t* gb = B + (size_t)bx * 2 * RB + (size_t)kt * SLAB + (V5_PRESWZ ? lane * 16 : prow * 128 + pch * 16);
#if W8_DIAG_NOCOPY  // developer power/timing probe: only the first two stages are really copied
    if (!diag_copy) { mbar_arrive_cp_async(bar0 + 8 * s); return; }
#endif
    switch (warp) {
      case 0: issue_w(0, sa, ga, gb); break;
      case 1: issue_w(1, sa, ga, gb); break;
      case 2: issue_w(2, sa, ga, gb); break;
      case 3: issue_w(3, sa, ga, gb); break;
      case 4: issue_w(4, sa, ga, gb); break;
      case 5: issue_w(5, sa, ga, gb); break;
      case 6: issue_w(6, sa, ga, gb); break;
      default: issue_w(7, sa, ga, gb); break;
    }
    mbar_arrive_cp_async(bar0 + 8 * s);
  };

  const int a_row = wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int b_row = wn * 64 + (lane & 7) + (lane >> 4) * 8;
  const unsigned a_off0 = swz<128>(a_row, lane >> 4), b_off0 = swz<128>(b_row, (lane >> 3) & 1);

  unsigned dwf = 0, dwfl = 0, dwe = 0, dwel = 0;  // XM_DIAG_WAITS: failed polls / late stages, full and empty
  unsigned g = 0;  // stages started by this warp
  {  // prologue: the first tile's stages 0 and 1
    int bx, by;
    tile_xy(blockIdx.x, bx, by);
    issue(0, bx, by, 0);
    if (nk > 1) issue(1, bx, by, 1);
  }
#if W8_DIAG_NOCOPY
  diag_copy = false;
#endif
  // Inline hash (XM_PEARL_HASH_INLINE): a tile's transcript P is hashed while the next tile's first chunk runs, one
  // BLAKE3 G function after every second MMA, so the tensor cores never idle for the ~1000 instructions of a hash.
  // P stays in local memory (volatile): 16 more live registers during a tile's first chunk make ptxas spill accumulators.
  volatile uint32_t P[16];
  uint32_t pv[16], p_rows = 0, p_cols = 0, hm0 = 0, hm1 = 0;
  bool has_prev = false;
#pragma unroll
  for (int w = 0; w < 16; ++w) P[w] = pv[w] = 0;
  // Staggered hash (XM_PEARL_HASH_STAGGER): a tile's transcript is hashed during the next tile, right after this warp
  // has issued its copies, and the two warps that share a scheduler do it in different chunks. One warp alone can
  // keep a scheduler's tensor unit busy, so while one hashes (~1000 adds and rotates, which would slow the MMAs if
  // interleaved with them) its partner takes the whole unit, instead of all eight warps pausing at the tile boundary.
#if W8_DEEP
  // Deep pipeline: with fragments loaded a k-step ahead, a warp has read everything it needs from a stage's buffer
  // half-way through that stage. It releases the buffer there, and at the last k-step refills it with the stage two
  // ahead (1.5 stages for the copy to land instead of barely one: Nsight showed ~7 of the 10 idle tensor points were
  // warps polling for a stage that was still in flight). The next stage's first fragments are loaded at the last
  // k-step too, so a stage no longer starts with a wait for ldmatrix.
  unsigned naf[4][4], nbf[8][2];
  bool pre = false;  // naf / nbf already hold the next stage's first fragments
  auto load_frags = [&](const int buf, const int k1) {
    const unsigned fa = sbase + buf * STAGE_BYTES, fb = fa + STAGE_A;
    const unsigned pa = fa + (a_off0 ^ (k1 << 5));
#pragma unroll
    for (int i = 0; i < 4; ++i) ldmatrix_x4(naf[i][0], naf[i][1], naf[i][2], naf[i][3], pa + i * 16 * 128);
    const unsigned pb = fb + (b_off0 ^ (k1 << 5));
#pragma unroll
    for (int j = 0; j < 8; j += 2) ldmatrix_x4(nbf[j][0], nbf[j][1], nbf[j + 1][0], nbf[j + 1][1], pb + j * 8 * 128);
  };
#endif
  const int hash_kc = W8_LATE(warp) ? nk / 2 : 1;
  auto pending_hash = [&]() {
    uint32_t pw[16], h[8];
#pragma unroll
    for (int q = 0; q < 16; ++q) pw[q] = P[q];
    b3_keyed64(a_seed, pw, h);
    if (u256_le(h, bound)) {
      uint32_t slot = atomicAdd(&found[0], 1u);
      if (slot < max_found) {
        uint32_t* rec = found + 1 + slot * 18;
        rec[0] = p_rows;
        rec[1] = p_cols;
#pragma unroll
        for (int q = 0; q < 16; ++q) rec[2 + q] = pw[q];
      }
    }
  };
  for (int t = blockIdx.x; t < T; t += G) {
    int bx, by, nbx = 0, nby = 0;
    tile_xy(t, bx, by);
    const bool has_next = t + G < T;
    if (has_next) tile_xy(t + G, nbx, nby);
    int acc[4][8][4], snap[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) acc[i][j][v] = snap[i][j][v] = 0;
    unsigned R[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) R[w] = 0;
    auto chunk = [&](const int kc, auto hash_tag) __attribute__((always_inline)) {
      constexpr bool HASH = sizeof(hash_tag) > 1;
      // Micro-step s of the previous tile's hash after MMA (ks, i, j), j odd: 7 rounds x 8 G = 56 steps in 64 slots.
      auto hstep = [&](const int ks, const int i, const int j) {
        if (!HASH || !(j & 1) || W8_DIAG_NOHSTEP) return;
        const int s = ks * 16 + i * 4 + (j >> 1);
        if (s == 0) {
#pragma unroll
          for (int q = 0; q < 8; ++q) pv[q] = a_seed.w[q];
#pragma unroll
          for (int q = 0; q < 4; ++q) pv[8 + q] = kB3IV[q];
          pv[12] = 0, pv[13] = 0, pv[14] = 64, pv[15] = B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED;
        }
        if (s < 56) {
          // The two message words were loaded one step ago (hm0, hm1): a load issued here would hold this warp's
          // next MMA back until it returns. Load the next step's words now.
          const int r = s >> 3, q = s & 7;
          const uint32_t m0 = hm0, m1 = hm1;
          if (s + 1 < 56) {
            const int r1 = (s + 1) >> 3, q1 = (s + 1) & 7;
            hm0 = P[w8_sched(r1, 2 * q1)], hm1 = P[w8_sched(r1, 2 * q1 + 1)];
          }
          b3_g(pv, w8_ga(q), w8_gb(q), w8_gc(q), w8_gd(q), m0, m1);
        }
      };
      if (HASH) hm0 = P[0], hm1 = P[1];  // step 0's message words
      const int st = g & 1;
#if W8_DEEP
      if (!pre) {
        mbar_wait(bar0 + 8 * st, (g >> 1) & 1);
        load_frags(st, 0);
      }
#else
#ifdef XM_DIAG_WAITS
      { const unsigned nw = mbar_wait_count(bar0 + 8 * st, (g >> 1) & 1); dwf += nw; dwfl += nw != 0; }
#else
      mbar_wait<W8_SLEEP_DATA>(bar0 + 8 * st, (g >> 1) & 1);
#endif
#if !W8_EARLY_RELEASE && !W8_RELEASE_KS2
      if (g > 0) mbar_arrive_lane0(bar0 + 8 * (STAGES + (st ^ 1)), lane);  // this warp is done with stage g - 1's buffer
#endif
      const unsigned sA = sbase + st * STAGE_BYTES, sB = sA + STAGE_A;
#if W8_FRAG_AHEAD
      unsigned naf[4][4], nbf[8][2];
      auto load_frags = [&](const int k1) {
        const unsigned pa = sA + (a_off0 ^ (k1 << 5));
#pragma unroll
        for (int i = 0; i < 4; ++i) ldmatrix_x4(naf[i][0], naf[i][1], naf[i][2], naf[i][3], pa + i * 16 * 128);
        const unsigned pb = sB + (b_off0 ^ (k1 << 5));
#pragma unroll
        for (int j = 0; j < 8; j += 2) ldmatrix_x4(nbf[j][0], nbf[j][1], nbf[j + 1][0], nbf[j + 1][1], pb + j * 8 * 128);
      };
      load_frags(0);
#endif
#endif
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
#if W8_DEEP
        auto refill = [&](int) {};
        unsigned af[4][4], bf[8][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int v = 0; v < 4; ++v) af[i][v] = naf[i][v];
#pragma unroll
        for (int j = 0; j < 8; ++j) bf[j][0] = nbf[j][0], bf[j][1] = nbf[j][1];
        if (ks < 3) {
          load_frags(st, ks + 1);
          if (ks == 2) mbar_arrive_lane0(bar0 + 8 * (STAGES + st), lane);  // nothing more to read from this buffer
        } else {
          const int k2 = kc + 2;  // the stage two ahead goes into this buffer
          if (k2 < nk || has_next) {
            mbar_wait(bar0 + 8 * (STAGES + st), (g >> 1) & 1);
            issue(st, k2 < nk ? bx : nbx, k2 < nk ? by : nby, k2 < nk ? k2 : k2 - nk);
          }
          pre = W8_PRELOAD && (kc + 1 < nk || has_next);
          if (pre) {
            mbar_wait(bar0 + 8 * (st ^ 1), ((g + 1) >> 1) & 1);
            load_frags(st ^ 1, 0);
          }
        }
#else
        // Refill the other buffer with stage g + 1 (the next tile's stage 0 after this tile's last) once every
        // warp has left stage g - 1.
        auto refill = [&](int row) {
          if (ks != V5_COPY_AT || row != V5_COPY_ROW || g == 0) return;
          const bool last = kc + 1 == nk;
          if (!last || has_next) {
#ifdef XM_DIAG_WAITS
            { const unsigned nw = mbar_wait_count(bar0 + 8 * (STAGES + (st ^ 1)), ((g - 1) >> 1) & 1); dwe += nw; dwel += nw != 0; }
#else
            mbar_wait<W8_SLEEP_FREE>(bar0 + 8 * (STAGES + (st ^ 1)), ((g - 1) >> 1) & 1);
#endif
            issue(st ^ 1, last ? nbx : bx, last ? nby : by, last ? 0 : kc + 1);
          }
        };
        if (V5_COPY_AT == 0 || V5_COPY_AT == 3 || V5_COPY_ROW == 0) refill(V5_COPY_ROW);
#if W8_FRAG_AHEAD
        // Fragments one k-step ahead: this step's were loaded during the previous one (the first step's at the start
        // of the stage), so its first MMAs do not wait for ldmatrix (Nsight: ~10% of all stalled warp samples sat on
        // the first MMA after the loads, on both warps of a scheduler at once). Costs 32 registers.
        unsigned af[4][4], bf[8][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int v = 0; v < 4; ++v) af[i][v] = naf[i][v];
#pragma unroll
        for (int j = 0; j < 8; ++j) bf[j][0] = nbf[j][0], bf[j][1] = nbf[j][1];
#if W8_RELEASE_KS2
        // With fragments a step ahead, the k-step 2 loads are the last reads of this stage's buffer: release it here,
        // 1.5 k-steps before the old point (the start of the next stage), so the refill can be issued much earlier.
        if (ks == 2) { load_frags(3); mbar_arrive_lane0(bar0 + 8 * (STAGES + st), lane); } else
#endif
        if (ks < 3) {
          load_frags(ks + 1);
#if W8_PIN_LOADS
          // A never-taken branch on a run-time value ends the basic block: ptxas otherwise sinks these loads to just
          // before their first use in the next k-step, which is the wait they are meant to remove.
          if (K < 0) return;
#endif
        }
#else
        unsigned af[4][4], bf[8][2];
        const unsigned pa = sA + (a_off0 ^ (ks << 5));
#pragma unroll
        for (int i = 0; i < 4; ++i) ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], pa + i * 16 * 128);
        const unsigned pb = sB + (b_off0 ^ (ks << 5));
#pragma unroll
        for (int j = 0; j < 8; j += 2) ldmatrix_x4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], pb + j * 8 * 128);
#endif
#endif  // W8_DEEP
#if W8_EARLY_RELEASE
        // Release the buffer as soon as this warp has read its last fragments from it (not at the start of the next
        // stage, which a warp may reach much later than its scheduler partner): the refill can then start at once.
        if (ks == 3) mbar_arrive_lane0(bar0 + 8 * (STAGES + st), lane);
#endif
        if (ks == 0) {
          unsigned xs[4] = {0, 0, 0, 0};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
              xs[(i * 8 + j) & 3] ^=
                  (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(acc[i][j], snap[i][j], af[i], bf[j][0], bf[j][1]), hstep(ks, i, j);
          }
          const unsigned xp = xs[0] ^ xs[1] ^ xs[2] ^ xs[3];
          const int cp = (kc - 1) & 15;
          const unsigned r = R[cp], nr = ((r << 13) | (r >> 19)) ^ xp;
          R[cp] = kc > 0 ? nr : r;
        } else if (ks == 3) {
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8_oop(snap[i][j], acc[i][j], af[i], bf[j][0], bf[j][1]), hstep(ks, i, j);
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (V5_COPY_ROW > 0 && V5_COPY_AT != 0 && V5_COPY_AT != 3) refill(i);
            if (XM_PEARL_HASH_STAGGER && ks == V5_COPY_AT && i == V5_COPY_ROW && kc == hash_kc && has_prev) pending_hash();
#pragma unroll
            for (int j = 0; j < 8; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]), hstep(ks, i, j);
          }
        }
      }
      if (HASH && has_prev) {  // the previous tile's hash is complete: check it
        uint32_t h[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) h[q] = pv[q] ^ pv[q + 8];
        if (u256_le(h, bound)) {
          uint32_t slot = atomicAdd(&found[0], 1u);
          if (slot < max_found) {
            uint32_t* rec = found + 1 + slot * 18;
            rec[0] = p_rows;
            rec[1] = p_cols;
#pragma unroll
            for (int q = 0; q < 16; ++q) rec[2 + q] = P[q];
          }
        }
      }
    };
#if XM_PEARL_HASH_INLINE
    if (out) chunk(0, W8NoHash{}); else chunk(0, W8Hash{});
#else
    chunk(0, W8NoHash{});
#endif
    ++g;
    for (int kc = 1; kc < nk; ++kc, ++g) chunk(kc, W8NoHash{});
    {  // the last chunk's fold
      unsigned xl = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          xl ^= (unsigned)snap[i][j][0] ^ (unsigned)snap[i][j][1] ^ (unsigned)snap[i][j][2] ^ (unsigned)snap[i][j][3];
      const int cl = (nk - 1) & 15;
      const unsigned r = R[cl];
      R[cl] = ((r << 13) | (r >> 19)) ^ xl;
    }
    const uint32_t t_rows = (uint32_t)(by * 128 + wm * 64 + (lane >> 2));
    const uint32_t t_cols = (uint32_t)(bx * 256 + wn * 64 + 2 * (lane & 3));
    if (out) {
      const size_t idx = ((size_t)(t_rows / 64) * 8 + t_rows % 8) * (size_t)(N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2;
#pragma unroll
      for (int w = 0; w < 16; ++w) out[idx * 16 + w] = R[w];
      continue;
    }
#if XM_PEARL_HASH_INLINE || XM_PEARL_HASH_STAGGER
#pragma unroll
    for (int w = 0; w < 16; ++w) P[w] = R[w];
    p_rows = t_rows, p_cols = t_cols, has_prev = true;
    continue;
#endif
    uint32_t words[16];
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
#if XM_PEARL_HASH_INLINE || XM_PEARL_HASH_STAGGER
  if (has_prev) pending_hash();  // the block's last tile
#endif
#ifdef XM_DIAG_WAITS
  if (lane == 0 && scratch)
    atomicAdd(scratch + 400000, dwf), atomicAdd(scratch + 400001, dwfl), atomicAdd(scratch + 400002, g), atomicAdd(scratch + 400003, dwe),
        atomicAdd(scratch + 400004, dwel);
#endif
}


extern "C" __global__ void __launch_bounds__(256) xm_pearl_gemm_w8(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr,
    unsigned* __restrict__ scratch) {
  w8_pass(A, B, out, M, N, K, a_seed, bound, found, max_found, scratch);
}

// xm_pearl_mine_w8: a whole batch of passes in one launch, for matrices that stay in the L2 cache (a pass is then
// well under a millisecond and the kernel boundaries of separate noise and GEMM launches cost ~10% of it).
// Pass p: (0) every thread helps build A's noise inputs for seed p (uniform matrix, index pairs) and the staged 1 KB
// nonce chunk is copied into A; (1) A' = A + noise; (2) the GEMM pass with its fused jackpot check into found slot p.
// The grid meets at a global counter between the phases (sync[2p], sync[2p + 1], zeroed by the host), because every
// block reads what the others wrote. passes: 16 words per pass = a_noise_seed (8), nonce chunk index, 7 unused.
__device__ __forceinline__ void w8_grid_sync(unsigned* ctr, unsigned blocks) {
  __threadfence();  // this thread's stores must be visible to the other SMs before the grid moves on
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(ctr, 1u);
    while (*(volatile unsigned*)ctr < blocks) __nanosleep(64);
  }
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(256) xm_pearl_mine_w8(
    int8_t* __restrict__ sig, int8_t* __restrict__ Ap, const int8_t* __restrict__ Bp, int8_t* __restrict__ uni,
    uint32_t* __restrict__ pairs, int M, int N, int K, uint32_t rank, const uint32_t* __restrict__ passes,
    uint32_t npass, const uint4* __restrict__ stage, U256w bound, uint32_t* __restrict__ found, uint32_t max_found,
    uint32_t slot_words, unsigned* __restrict__ sync, unsigned* __restrict__ scratch) {
  const uint32_t G = gridDim.x, NT = G * 256, gid = blockIdx.x * 256 + threadIdx.x;
  long long tc = clock64();  // XM_DIAG_PHASE: block 0 / thread 0 accumulates SM clock cycles per phase
  auto phase = [&](int i) {
#ifdef XM_DIAG_PHASE
    if (gid == 0) { const long long now = clock64(); atomicAdd(scratch + 410000 + i, (unsigned)(now - tc)); tc = now; }
#endif
  };
  for (uint32_t p = 0; p < npass; ++p) {
    Key8 seed;
#pragma unroll
    for (int i = 0; i < 8; ++i) seed.w[i] = passes[p * 16 + i];
    // (0) nonce chunk, uniform matrix, index pairs
    if (gid < 64) reinterpret_cast<uint4*>(sig + (size_t)passes[p * 16 + 8] * 1024)[gid] = stage[p * 64 + gid];
    const uint32_t nhash = (uint32_t)M * rank / 32;
    for (uint32_t h = gid; h < nhash; h += NT) {
      uint32_t d[8];
      noise_hash(seed, h, 0, 0, d);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        *reinterpret_cast<uint32_t*>(uni + (size_t)h * 32 + i * 4) =
            ((d[i] & 63) - 32 & 255) | (((d[i] >> 8 & 63) - 32 & 255) << 8) | (((d[i] >> 16 & 63) - 32 & 255) << 16) |
            (((d[i] >> 24 & 63) - 32 & 255) << 24);
    }
    for (uint32_t h = gid; h * 8 < (uint32_t)K; h += NT) {
      uint32_t d[8];
      noise_hash(seed, h, 1, 0, d);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t l = h * 8 + j;
        if (l >= (uint32_t)K) break;
        const uint32_t u = d[j], first = u & (rank - 1), mulhi = (uint32_t)(((uint64_t)(rank - 1) * u) >> 32);
        pairs[2 * l] = first;
        pairs[2 * l + 1] = first ^ (1 + mulhi);
      }
    }
    phase(0);
    w8_grid_sync(sync + 2 * p, G);
    phase(1);
    // (1) A' = A + noise
    const uint32_t ntask = (uint32_t)M * ((uint32_t)K / 8);
#if XM_PEARL_APPLY_SMEM
    // Block b takes rows [b * per, (b + 1) * per) in chunks of RCH rows. A chunk's signal bytes and uniform lines, and
    // once per pass the index pairs, come into shared memory with cp.async (non-blocking, all lines in flight at once:
    // plain loads stall a warp for every L2 line, which made this phase 5% of a pass), then every thread noises 8
    // bytes at a time from shared memory and stores them in the GEMM layout.
    {
      extern __shared__ __align__(128) int8_t smem[];
      const unsigned sbase = (unsigned)__cvta_generic_to_shared(&smem[0]);
      const uint32_t Ku = (uint32_t)K, groups = Ku / 8, per = ((uint32_t)M + G - 1) / G;
      const uint32_t PB = Ku * 8;                                  // bytes of index pairs
      const uint32_t RCH = min(32u, (98304u - PB) / (Ku + rank));  // rows per chunk
      const uint32_t r_begin = blockIdx.x * per, r_end = min((uint32_t)M, r_begin + per);
      const unsigned s_pairs = sbase, s_uni = sbase + PB, s_sig = s_uni + RCH * rank;
      for (uint32_t i = threadIdx.x; i < PB / 16; i += 256) cp_async16(s_pairs + i * 16, reinterpret_cast<const int8_t*>(pairs) + (size_t)i * 16);
      for (uint32_t r0 = r_begin; r0 < r_end; r0 += RCH) {
        const uint32_t nr = min(RCH, r_end - r0);
        for (uint32_t i = threadIdx.x; i < nr * rank / 16; i += 256) cp_async16(s_uni + i * 16, uni + (size_t)r0 * rank + (size_t)i * 16);
        for (uint32_t i = threadIdx.x; i < nr * Ku / 16; i += 256) cp_async16(s_sig + i * 16, sig + (size_t)r0 * Ku + (size_t)i * 16);
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        const int8_t* l_sig = reinterpret_cast<const int8_t*>(__cvta_shared_to_generic(s_sig));
        const int8_t* l_uni = reinterpret_cast<const int8_t*>(__cvta_shared_to_generic(s_uni));
        const uint32_t* l_pairs = reinterpret_cast<const uint32_t*>(__cvta_shared_to_generic(s_pairs));
        // Consecutive lanes take consecutive 8-byte groups, so a store instruction covers 256 contiguous bytes (two
        // cache lines). Tried: lane = row (32 lines per store) is 4x slower: stores cost per line they touch.
        for (uint32_t j = threadIdx.x; j < nr * groups; j += 256) {
          const uint32_t rl = j / groups, k0 = (j - rl * groups) * 8;
          const uint2 v = apply_noise_8(l_sig + (size_t)rl * Ku + k0, l_pairs + 2 * k0, l_uni + (size_t)rl * rank);
          *reinterpret_cast<uint2*>(Ap + tiled_offset(r0 + rl, k0, Ku)) = v;
        }
        __syncthreads();
      }
    }
    (void)ntask;
#else
    for (uint32_t t = gid; t < ntask; t += NT) apply_noise_task(sig, uni, pairs, Ap, t, (uint32_t)K, rank);
#endif
    phase(2);
    w8_grid_sync(sync + 2 * p + 1, G);
    phase(3);
    // (2) GEMM + jackpot check
    w8_pass(Ap, Bp, nullptr, M, N, K, seed, bound, found + (size_t)p * slot_words, max_found, scratch);
    phase(4);
  }
}

#else  // XM_PEARL_USE_SIMT

// SIMT GEMM for GPUs without int8 tensor cores (Pascal sm_61 and newer, dp4a). Same tiled input layout, 128x256
// tiles of 8 warps in 2x4 and the same hash tiles as the tensor-core kernel: a lane owns rows (lane / 4) + 8 i
// (i < 8) and columns 2 (lane % 4) + {0, 1} + 8 j (j < 8) of its warp's 64x64 block, so the host's NVIDIA pattern and
// the debug index are unchanged. One block per tile (no persistence): grid (N / 256, M / 128), 256 threads, static
// shared memory. Stages of 64 k are staged through registers into one shared buffer; rows are padded to 80 bytes so
// the 32-bit reads of 8 rows (A) or of 4 rows 2 apart (B) hit distinct banks.
namespace {
constexpr int S_BM = 128, S_BN = 256, S_BK = 64, S_LD = S_BK + 16, S_ROWS = S_BM + S_BN;
constexpr int S_V16 = S_ROWS * S_BK / 16 / 256;  // 16-byte global loads per thread per stage
}  // namespace

// Turing int8 tensor op: an 8x8 block of C += A (8x16, row) * B (16x8, col). Lane l supplies 4 k-bytes 4 (l % 4).. of
// A row l / 4 and of B column l / 4, and holds C[l / 4][2 (l % 4) + {0, 1}]: the same ownership as the dp4a variant.
__device__ __forceinline__ void mma_m8n8k16(int& c0, int& c1, unsigned a, unsigned b) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}

template <bool TC>
__device__ __forceinline__ void pearl_legacy_body(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    const Key8& a_seed, const U256w& bound, uint32_t* __restrict__ found, uint32_t max_found) {
  __shared__ __align__(16) int8_t sm[S_ROWS * S_LD];  // rows 0..127: A, 128..383: B
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int bx = blockIdx.x, by = blockIdx.y;
  const size_t SLAB = 16384, RB = (size_t)(K / 128) * SLAB;  // bytes per 128-row block (tiled layout)
  const int8_t* gA = A + (size_t)by * RB;
  const int8_t* gB = B + (size_t)(2 * bx) * RB;

  uint4 st[S_V16];
  // Stage s = k [64 s, 64 s + 64): bytes (s & 1) * 64.. of slab s / 2 in each row. Load q = tid + 256 i covers row
  // q / 4, 16-byte part q % 4; i < 2 are A rows, the rest B rows (two 128-row blocks).
  auto gload = [&](int s) {
    const size_t so = (size_t)(s >> 1) * SLAB + (s & 1) * 64;
#pragma unroll
    for (int i = 0; i < S_V16; ++i) {
      const int q = tid + 256 * i, row = q >> 2, part = q & 3;
      const int8_t* src = row < S_BM ? gA + so + row * 128 + part * 16
                                     : gB + (size_t)((row - S_BM) >> 7) * RB + so + ((row - S_BM) & 127) * 128 + part * 16;
      st[i] = *reinterpret_cast<const uint4*>(src);
    }
  };
  auto sstore = [&]() {
#pragma unroll
    for (int i = 0; i < S_V16; ++i) {
      const int q = tid + 256 * i;
      *reinterpret_cast<uint4*>(sm + (q >> 2) * S_LD + (q & 3) * 16) = st[i];
    }
  };

  int acc[8][16];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[i][j] = 0;
  unsigned R[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) R[w] = 0;

  const int8_t* pa = sm + (wm * 64 + (lane >> 2)) * S_LD;

  const int ns = K / S_BK;
  gload(0);
  sstore();
  __syncthreads();
  for (int s = 0; s < ns; ++s) {
    if (s + 1 < ns) gload(s + 1);
    if constexpr (TC) {
      const int8_t* pat = pa + 4 * (lane & 3);                                           // A rows lane / 4 + 8 i
      const int8_t* pbt = sm + (S_BM + wn * 64 + (lane >> 2)) * S_LD + 4 * (lane & 3);  // B rows lane / 4 + 8 j
#pragma unroll
      for (int ks = 0; ks < S_BK; ks += 16) {
        unsigned a[8], b[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = *reinterpret_cast<const unsigned*>(pat + 8 * i * S_LD + ks);
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = *reinterpret_cast<const unsigned*>(pbt + 8 * j * S_LD + ks);
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_m8n8k16(acc[i][2 * j], acc[i][2 * j + 1], a[i], b[j]);
      }
    } else {
      const int8_t* pb = sm + (S_BM + wn * 64 + 2 * (lane & 3)) * S_LD;  // B rows 2 (lane % 4) + {0, 1} + 8 j
#pragma unroll 2
      for (int w = 0; w < S_BK; w += 4) {
        int a[8], b[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = *reinterpret_cast<const int*>(pa + 8 * i * S_LD + w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          b[2 * j] = *reinterpret_cast<const int*>(pb + 8 * j * S_LD + w);
          b[2 * j + 1] = *reinterpret_cast<const int*>(pb + (8 * j + 1) * S_LD + w);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int j = 0; j < 16; ++j) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
      }
    }
    __syncthreads();
    if (s + 1 < ns) {
      sstore();
      __syncthreads();
    }
    if (s & 1) {  // end of a 128-k Pearl chunk: fold
      unsigned x = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 16; ++j) x ^= (unsigned)acc[i][j];
      const int c = (s >> 1) & 15;
      const unsigned r = R[c];
      R[c] = ((r << 13) | (r >> 19)) ^ x;
    }
  }

  const uint32_t t_rows = (uint32_t)(by * S_BM + wm * 64 + (lane >> 2));
  const uint32_t t_cols = (uint32_t)(bx * S_BN + wn * 64 + 2 * (lane & 3));
  if (out) {
    const size_t idx = ((size_t)(t_rows / 64) * 8 + t_rows % 8) * (size_t)(N / 8) + (t_cols / 64) * 4 + (t_cols % 8) / 2;
#pragma unroll
    for (int w = 0; w < 16; ++w) out[idx * 16 + w] = R[w];
    return;
  }
  uint32_t words[16];
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

extern "C" __global__ void __launch_bounds__(256) xm_pearl_gemm(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr) {
  pearl_legacy_body<XM_PEARL_LEGACY_TC>(A, B, out, M, N, K, a_seed, bound, found, max_found);
}

#if XM_PEARL_LEGACY_TC
// The dp4a variant next to the m8n8k16 one: the host compares the two at start-up and mines with the faster one if
// they agree (GTX 16 Turing GPUs have no tensor cores), or with this one if they differ.
extern "C" __global__ void __launch_bounds__(256) xm_pearl_gemm_simt(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found, unsigned* __restrict__ wave_ctr) {
  pearl_legacy_body<false>(A, B, out, M, N, K, a_seed, bound, found, max_found);
}
#endif

#endif  // XM_PEARL_USE_SIMT
