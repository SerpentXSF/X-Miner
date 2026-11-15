// X-Miner Pearl kernels for AMD RDNA4 (gfx1201): shared Pearl device code + the WMMA int8 GEMM/transcript core.
// The GEMM is X-Miner's own pearl-rdna4 kernel (~108-110 TH/s on an RX 9070 XT, bit-exact transcript).
#include <hip/hip_runtime.h>

#include "pearl/pearl_common.h"

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));


// ---------------------------------------------------------------------------------------------------------
// Data layout ("packed", fragment-major)
//
// The miner is free to choose how A and B live in GPU memory. We store each operand so that one wave-wide
// 16-byte-per-lane load returns exactly the WMMA fragments for two consecutive k-steps:
//   for 16-row block rb and k-pair kp (32 k values): 512 contiguous bytes, lane l's 16 bytes are
//   row (rb*16 + l%16), k in [kp*32 + 8*(l/16), +8) followed by [kp*32 + 16 + 8*(l/16), +8).
// Every load is then a fully coalesced 512-byte read. (A and B use the same k-permutation, so the dot
// products are unchanged, and the permutation never crosses a 128-k fold boundary.)

// Grouped rasterization: consecutive workgroups sweep WGM row-blocks x all column-blocks, so the workgroups
// running at the same time share A/B blocks in L2 / Infinity Cache instead of re-reading them from VRAM.
// (Worth up to +35% at long K on this card.)
template <int WGM>
__device__ __forceinline__ void raster(int& bx, int& by) {
    const int gx = gridDim.x, gy = gridDim.y, pid = blockIdx.y * gx + blockIdx.x;
    const int per_group = WGM * gx, group = pid / per_group, first = group * WGM;
    const int rows = min(gy - first, WGM), in = pid % per_group;
    by = first + in % rows;
    bx = in / rows;
}

// Workgroup barrier that only waits for LDS traffic. __syncthreads() emits a workgroup release fence, which
// in WGP mode also waits for ALL outstanding global loads (s_wait_loadcnt 0) and so destroys prefetching.
__device__ __forceinline__ void lds_barrier() {
#if defined(__GFX12__)
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
#else
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
#endif
    asm volatile("" ::: "memory");
}

// GEMM paths, all with the same data layout, launch shape and tile ownership:
//  PATH_W12  RDNA4 WMMA (v_wmma_i32_16x16x16_iu8, 8 bytes per lane per operand).
//  PATH_W11  RDNA3 WMMA: 16 bytes per lane, lanes 16-31 repeat lanes 0-15, and lane l holds accumulator rows
//            2 r + l/16 (not 8 l/16 + r as on RDNA4). Reading A row 8 (p % 2) + p / 2 into WMMA row p makes element r
//            of lane l matrix row 8 l/16 + r again, so the hash tiles do not change. Not run on RDNA3 hardware by us:
//            the host compares it with xm_pearl_gemm_simt at start-up and falls back if they differ.
//  PATH_SIMT RDNA1-2 (no int8 WMMA): each lane computes its own 128 sums by 4-byte dot products.
// XM_PEARL_SIMT=1 forces the SIMT path on RDNA3/4 (developer check: --selftest then covers it).
#define PATH_W12 0
#define PATH_W11 1
#define PATH_SIMT 2
#if defined(__GFX12__) && !defined(XM_PEARL_SIMT)
#define XM_PEARL_PATH PATH_W12
#elif defined(__GFX11__) && !defined(XM_PEARL_SIMT)
#define XM_PEARL_PATH PATH_W11
#else
#define XM_PEARL_PATH PATH_SIMT
#endif
#ifndef XM_PEARL_BREAK
#define XM_PEARL_BREAK 0  // developer: 1 corrupts the WMMA paths' sums, to test the host's start-up check
#endif
// Signed 4 x int8 dot product + c, wrapping like the WMMA accumulators. XM_PEARL_DOT=0 forces the plain version.
#ifndef XM_PEARL_DOT
#define XM_PEARL_DOT 1
#endif
#if XM_PEARL_DOT && (defined(__GFX11__) || defined(__GFX12__))
#define XM_PEARL_HAS_DOT 2  // sudot4
#elif XM_PEARL_DOT && (defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1030__) || \
                       defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || \
                       defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx10_3_generic__))
#define XM_PEARL_HAS_DOT 1  // sdot4
#else
#define XM_PEARL_HAS_DOT 0  // gfx1010 (RX 5600 / 5700): no dot instructions
#endif
// XM_PEARL_PAD=1 (SIMT): one padding vector after every 8 in the LDS chunks, so the A reads of lanes 0-15 and 16-31
// (8 vectors apart) stop sharing banks. On by default for RDNA2 (sdot4), where it is ~24% faster although it spills
// ~20 VGPRs: RX 6800 8.48 -> 10.60, RX 6700 XT 5.51 -> 6.85, RX 6600 XT 4.32 -> 5.36 TH/s. Off for RDNA1 (plain
// multiplies; RX 5700 XT 1.3-1.7 TH/s -> 10-25% slower padded) and for RDNA3/4, which use the matrix path.
#ifndef XM_PEARL_PAD
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) ||     defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx10_3_generic__)
#define XM_PEARL_PAD 1
#else
#define XM_PEARL_PAD 0
#endif
#endif
__device__ __forceinline__ int dot4(int a, int b, int c) {
#if XM_PEARL_HAS_DOT == 2
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif XM_PEARL_HAS_DOT == 1
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    c += (int)(signed char)a * (int)(signed char)b;
    c += (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8);
    c += (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16);
    c += (a >> 24) * (b >> 24);
    return c;
#endif
}

// ---------------------------------------------------------------------------------------------------------
// Kernel. Structure follows AMD hipBLASLt's tuned gfx1201 int8 kernel (MT128x128x64, 4 waves), widened:
//  - 128x256 output block per workgroup, 8 waves in 2x4, each wave owns 64x64 (16 WMMA accumulators). The wider
//    block cuts global loads and LDS stores per MAC by 25%; the power saved goes into clock (RX 9070 XT at 70% PL:
//    103.2 -> 106.0 TH/s; 4x4 waves stall on barriers, 79.5).
//  - k advances 64 per stage. One LDS buffer holds the whole stage (A 8 KB + B 16 KB).
//  - Stage kt+1 global loads are issued at the top of stage kt and stored to LDS mid-stage.
//  - Per stage: WMMA(k-pair 0) | barrier | store next stage | WMMA(k-pair 1) | barrier | load next fragments.
//  - Unrolled by two stages (one Pearl fold per 128 k) with the last pair peeled: no branches in the hot loop.
// Hash tiles follow the WMMA accumulator layout (gfx12 wave32: lane l holds rows 8*(l/16) + 0..7 of column l%16
// in each 16x16 tile): rows {0..7, 16..23, 32..39, 48..55} x cols {0, 16, 32, 48} of the wave's 64x64 block
// (h = 32, w = 4; the host's AMD mining configuration), i.e. exactly the 128 accumulators one lane owns.
// Every 128 k each lane XORs its 128 running sums into x and folds t[c % 16] = rotl(t[c % 16], 13) ^ x.
// Requirements: M % 128 == 0, N % 128 == 0, K % 2048 == 0.
#ifndef XM_PEARL_WMW
#define XM_PEARL_WMW 2  // waves along M (64 rows each)
#endif
#ifndef XM_PEARL_WNW
#define XM_PEARL_WNW 4  // waves along N (64 columns each)
#endif
#ifndef XM_PEARL_WGM
#define XM_PEARL_WGM 32  // row blocks per rasterization group
#endif
template <int WGM, int PATH>
__device__ __forceinline__ void pearl_gemm_rdna4_body(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Bp, unsigned* __restrict__ out, int M, int N, int K,
    const Key8& a_seed, const U256w& bound, uint32_t* __restrict__ found, uint32_t max_found) {
    constexpr int WMW = XM_PEARL_WMW, WNW = XM_PEARL_WNW, NW = WMW * WNW;
    constexpr int FM = 4, FN = 4, KP = 2, RBA = 4 * WMW, RBB = 4 * WNW;
    // LDS: one 512-byte chunk (32 vectors) per row block and k-pair. SIMT pads a vector after every 8 (36 per chunk):
    // its lanes 0-15 and 16-31 read A vectors 8 apart, which without the pad sit in the same banks (2-way conflicts).
    constexpr bool PAD = PATH == PATH_SIMT && XM_PEARL_PAD;  // padded LDS chunks (see XM_PEARL_PAD)
    constexpr int CHV = PAD ? 36 : 32;
    auto vix = [](int v) { return PAD ? v + (v >> 3) : v; };  // vector v of a chunk -> LDS slot
    constexpr int STAGE_V4 = RBA * KP * CHV;     // LDS vectors of the A stage (the B stage follows)
    constexpr int CA = RBA * KP / NW, CB = RBB * KP / NW;  // 512-byte chunks per wave per stage
    static_assert(CA >= 1 && CB >= 1 && CA * NW == RBA * KP && CB * NW == RBB * KP, "wave grid");
    __shared__ v4i lds[(RBA + RBB) * KP * CHV];  // [A stage | B stage]

    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int wm = wave / WNW, wn = wave % WNW;
    int bx, by;
    raster<WGM>(bx, by);
    const size_t rb_bytes = (size_t)K * 16;      // bytes per 16-row block in the packed layout
    const unsigned loff = lane * 16u;
    const int8_t* gA = Ap + (size_t)(by * RBA) * rb_bytes;
    const int8_t* gB = Bp + (size_t)(bx * RBB) * rb_bytes;

    // Global -> VGPR staging. Chunk c = wave + NW*i (wave-uniform): row-block c/2, k-pair c%2.
    // RDNA4 has no direct global->LDS copy, so data passes through VGPRs.
    v4i ga[CA], gb[CB];
    auto gload = [&](int kt) {
#pragma unroll
        for (int i = 0; i < CA; ++i) {
            const int c = wave + NW * i;
            ga[i] = *(const v4i*)(gA + (size_t)(c >> 1) * rb_bytes + (size_t)(kt * KP + (c & 1)) * 512 + loff);
        }
#pragma unroll
        for (int i = 0; i < CB; ++i) {
            const int c = wave + NW * i;
            gb[i] = *(const v4i*)(gB + (size_t)(c >> 1) * rb_bytes + (size_t)(kt * KP + (c & 1)) * 512 + loff);
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < CA; ++i) lds[(wave + NW * i) * CHV + vix(lane)] = ga[i];
#pragma unroll
        for (int i = 0; i < CB; ++i) lds[STAGE_V4 + (wave + NW * i) * CHV + vix(lane)] = gb[i];
    };
    v4i fa[KP][FM], fb[KP][FN];                  // RDNA4 WMMA fragments of the current stage
    auto lload = [&]() {
        if constexpr (PATH == PATH_W12) {
#pragma unroll
            for (int kp = 0; kp < KP; ++kp) {
#pragma unroll
                for (int i = 0; i < FM; ++i) fa[kp][i] = lds[((wm * FM + i) * KP + kp) * 32 + lane];
#pragma unroll
                for (int j = 0; j < FN; ++j) fb[kp][j] = lds[STAGE_V4 + ((wn * FN + j) * KP + kp) * 32 + lane];
            }
        }  // other paths read their operands from LDS inside mma()
    };

    v8i acc[FM][FN];
    constexpr int acc0 = (XM_PEARL_BREAK && PATH != PATH_SIMT) ? 1 : 0;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = (v8i){acc0, 0, 0, 0, 0, 0, 0, 0};
    // This lane's tile transcript, kept rotated so the word updated by the current 128-k chunk is always R[0]:
    // after N chunks word w is R[(w - N) mod 16], and K % 2048 == 0 brings it back to R[w] at the end.
    unsigned R[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) R[w] = 0;

    // In the packed layout row x of a 16-row block keeps its 32 k-bytes of a k-pair in the 16-byte vectors x and
    // x + 16; A and B share the k order, so any split of those 32 bytes gives the same dot products.
    const int cc = lane & 15, rr0 = 8 * (lane >> 4);
    const int* lw = reinterpret_cast<const int*>(lds);
#ifndef XM_PEARL_SIMT_QU
#define XM_PEARL_SIMT_QU 1  // words per unrolled step: more unrolling makes the compiler hoist loads and spill
#endif
    auto mma = [&](int kp) {
        if constexpr (PATH == PATH_W12) {
#if defined(__GFX12__)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        v2i a = h ? (v2i){fa[kp][i][2], fa[kp][i][3]} : (v2i){fa[kp][i][0], fa[kp][i][1]};
                        v2i b = h ? (v2i){fb[kp][j][2], fb[kp][j][3]} : (v2i){fb[kp][j][0], fb[kp][j][1]};
                        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[i][j], false);
                    }
#endif
        } else if constexpr (PATH == PATH_W11) {
#if defined(__GFX11__)
            // Two WMMAs per k-pair: vector x then vector x + 16 of each row. B column = lane % 16; A WMMA row
            // p = lane % 16 reads matrix row 8 (p % 2) + p / 2 (see PATH_W11 above).
            const int arow = 8 * (cc & 1) + (cc >> 1);
            v4i b[FN][2];
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                const int base = STAGE_V4 + ((wn * FN + j) * KP + kp) * 32 + cc;
                b[j][0] = lds[base];
                b[j][1] = lds[base + 16];
            }
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const int base = ((wm * FM + i) * KP + kp) * 32 + arow;
                const v4i a0 = lds[base], a1 = lds[base + 16];
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b[j][0], acc[i][j], false);
                    acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b[j][1], acc[i][j], false);
                }
            }
#endif
        } else {
            // SIMT: lane l owns rows 8 (l / 16) + r (accumulator element r) of column l % 16, as RDNA4 WMMA does.
            // One 4-byte word of the k-pair at a time (word q: vector q / 4, component q % 4): 4 B words + 32 A words
            // live next to the 128 sums, instead of whole 32-byte rows (which spilled).
#pragma unroll XM_PEARL_SIMT_QU
            for (int q = 0; q < 8; ++q) {
                const int off = 4 * vix(16) * (q >> 2) + (q & 3);  // ints: vector + 16 (q / 4), component q % 4
                int b[FN];
#pragma unroll
                for (int j = 0; j < FN; ++j) b[j] = lw[(STAGE_V4 + ((wn * FN + j) * KP + kp) * CHV + vix(cc)) * 4 + off];
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int a = lw[(((wm * FM + i) * KP + kp) * CHV + vix(rr0) + r) * 4 + off];
#pragma unroll
                        for (int j = 0; j < FN; ++j) acc[i][j][r] = dot4(a, b[j], acc[i][j][r]);
                    }
            }
        }
    };
    // Pearl fold: XOR of this lane's 128 running sums, folded into its own transcript. No cross-lane traffic.
    auto fold = [&](int) {
        unsigned x = 0;
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                v8i c = acc[i][j];
                x ^= (unsigned)(c[0] ^ c[1] ^ c[2] ^ c[3] ^ c[4] ^ c[5] ^ c[6] ^ c[7]);
            }
        unsigned r0 = ((R[0] << 13) | (R[0] >> 19)) ^ x;
#pragma unroll
        for (int w = 0; w < 15; ++w) R[w] = R[w + 1];
        R[15] = r0;
    };
    auto stage = [&](int kt) {
        if constexpr (PATH == PATH_W12) {
            gload(kt + 1);
            mma(0);
            lds_barrier();   // all waves finished reading stage kt from LDS
            lstore();        // stage kt+1 -> LDS (waits only on this thread's own global loads)
            mma(1);          // from the fragments loaded before the store
            lds_barrier();   // stage kt+1 visible to all waves
            lload();
        } else {             // these paths read LDS in both k-pairs, so the store waits until both are done
            gload(kt + 1);
            mma(0);

            mma(1);
            lds_barrier();
            lstore();
            lds_barrier();
        }
    };

    const int nk = K / 64;
    gload(0);
    lstore();
    lds_barrier();
    lload();
    for (int kt = 0; kt < nk - 2; kt += 2) {
        stage(kt);
        stage(kt + 1);
        fold((kt + 2) / 2 - 1);
    }
    stage(nk - 2);       // prefetches the final stage
    mma(0);
    mma(1);
    fold(nk / 2 - 1);


    // Epilogue: every lane hashes its own tile's transcript.
    const uint32_t t_rows = (uint32_t)(by * 64 * WMW + wm * 64 + 8 * (lane >> 4));
    const uint32_t t_cols = (uint32_t)(bx * 64 * WNW + wn * 64 + (lane & 15));
    if (out) {  // debug path: one 16-word transcript per lane tile, in launch order
        const size_t idx = ((size_t)(blockIdx.y * gridDim.x + blockIdx.x) * NW + wave) * 32 + lane;
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

// Entry point looked up by name from the host. Grid (N/(64 WNW), M/(64 WMW)), block 32 WMW WNW.
// Requirements: M % (64 WMW) == 0, N % (64 WNW) == 0, K % 2048 == 0.
// out != nullptr: write per-lane-tile transcripts (debug). Otherwise hash each tile in the epilogue and append tiles
// whose jackpot hash is <= bound to found[] (count, then 18-word records: t_rows, t_cols, 16 transcript words).
extern "C" __global__ __launch_bounds__(32 * XM_PEARL_WMW * XM_PEARL_WNW) void xm_pearl_gemm(const int8_t* __restrict__ Ap,
                                                               const int8_t* __restrict__ Bp,
                                                               unsigned* __restrict__ out, int M, int N, int K,
                                                               Key8 a_seed, U256w bound, uint32_t* __restrict__ found,
                                                               uint32_t max_found) {
  pearl_gemm_rdna4_body<XM_PEARL_WGM, XM_PEARL_PATH>(Ap, Bp, out, M, N, K, a_seed, bound, found, max_found);
}

#if XM_PEARL_PATH != PATH_SIMT
// The SIMT path next to a WMMA one, same arguments and launch: the host's start-up check compares the two on RDNA3
// (whose WMMA kernel we could not run ourselves) and mines with this one if they differ.
extern "C" __global__ __launch_bounds__(32 * XM_PEARL_WMW * XM_PEARL_WNW) void xm_pearl_gemm_simt(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Bp, unsigned* __restrict__ out, int M, int N, int K,
    Key8 a_seed, U256w bound, uint32_t* __restrict__ found, uint32_t max_found) {
  pearl_gemm_rdna4_body<XM_PEARL_WGM, PATH_SIMT>(Ap, Bp, out, M, N, K, a_seed, bound, found, max_found);
}
#endif
