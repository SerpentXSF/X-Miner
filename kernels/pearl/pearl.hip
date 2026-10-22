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
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    asm volatile("" ::: "memory");
}

// ---------------------------------------------------------------------------------------------------------
// Kernel. Structure follows AMD hipBLASLt's tuned gfx1201 int8 kernel (MT128x128x64, 4 waves):
//  - 128x128 output block per workgroup, 4 waves in 2x2, each wave owns 64x64 (16 WMMA accumulators).
//  - k advances 64 per stage. One LDS buffer holds the whole stage (A 8 KB + B 8 KB).
//  - Stage kt+1 global loads are issued at the top of stage kt and stored to LDS mid-stage.
//  - Per stage: WMMA(k-pair 0) | barrier | store next stage | WMMA(k-pair 1) | barrier | load next fragments.
//  - Unrolled by two stages (one Pearl fold per 128 k) with the last pair peeled: no branches in the hot loop.
// Hash tiles follow the WMMA accumulator layout (gfx12 wave32: lane l holds rows 8*(l/16) + 0..7 of column l%16
// in each 16x16 tile): rows {0..7, 16..23, 32..39, 48..55} x cols {0, 16, 32, 48} of the wave's 64x64 block
// (h = 32, w = 4; the host's AMD mining configuration), i.e. exactly the 128 accumulators one lane owns.
// Every 128 k each lane XORs its 128 running sums into x and folds t[c % 16] = rotl(t[c % 16], 13) ^ x.
// Requirements: M % 128 == 0, N % 128 == 0, K % 2048 == 0.
template <int WGM>
__device__ __forceinline__ void pearl_gemm_rdna4_body(
    const int8_t* __restrict__ Ap, const int8_t* __restrict__ Bp, unsigned* __restrict__ out, int M, int N, int K,
    const Key8& a_seed, const U256w& bound, uint32_t* __restrict__ found, uint32_t max_found) {
    constexpr int FM = 4, FN = 4, KP = 2, RB = 8;
    constexpr int STAGE_V4 = RB * KP * 32;       // 16-byte vectors per operand per stage (8 KB)
    __shared__ v4i lds[2 * STAGE_V4];            // [A stage | B stage]

    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int wm = wave >> 1, wn = wave & 1;
    int bx, by;
    raster<WGM>(bx, by);
    const size_t rb_bytes = (size_t)K * 16;      // bytes per 16-row block in the packed layout
    const unsigned loff = lane * 16u;
    const int8_t* gA = Ap + (size_t)(by * RB) * rb_bytes;
    const int8_t* gB = Bp + (size_t)(bx * RB) * rb_bytes;

    // Global -> VGPR staging. Thread's vector q = tid + 128*i belongs to chunk c = wave + 4*i (wave-uniform):
    // row-block c/2, k-pair c%2. RDNA4 has no direct global->LDS copy, so data passes through VGPRs.
    v4i ga[4], gb[4];
    auto gload = [&](int kt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = wave + 4 * i;
            const size_t off = (size_t)(c >> 1) * rb_bytes + (size_t)(kt * KP + (c & 1)) * 512 + loff;
            ga[i] = *(const v4i*)(gA + off);
            gb[i] = *(const v4i*)(gB + off);
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = (wave + 4 * i) * 32 + lane;
            lds[q] = ga[i];
            lds[STAGE_V4 + q] = gb[i];
        }
    };
    v4i fa[KP][FM], fb[KP][FN];                  // WMMA fragments of the current stage
    auto lload = [&]() {
#pragma unroll
        for (int kp = 0; kp < KP; ++kp) {
#pragma unroll
            for (int i = 0; i < FM; ++i) fa[kp][i] = lds[((wm * FM + i) * KP + kp) * 32 + lane];
#pragma unroll
            for (int j = 0; j < FN; ++j) fb[kp][j] = lds[STAGE_V4 + ((wn * FN + j) * KP + kp) * 32 + lane];
        }
    };

    v8i acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
    // This lane's tile transcript, kept rotated so the word updated by the current 128-k chunk is always R[0]:
    // after N chunks word w is R[(w - N) mod 16], and K % 2048 == 0 brings it back to R[w] at the end.
    unsigned R[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) R[w] = 0;

    auto mma = [&](int kp) {
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
        gload(kt + 1);
        mma(0);
        lds_barrier();   // all waves finished reading stage kt from LDS
        lstore();        // stage kt+1 -> LDS (waits only on this thread's own global loads)
        mma(1);
        lds_barrier();   // stage kt+1 visible to all waves
        lload();
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
    const uint32_t t_rows = (uint32_t)(by * 128 + wm * 64 + 8 * (lane >> 4));
    const uint32_t t_cols = (uint32_t)(bx * 128 + wn * 64 + (lane & 15));
    if (out) {  // debug path: one 16-word transcript per lane tile, in launch order
        const size_t idx = ((size_t)(blockIdx.y * gridDim.x + blockIdx.x) * 4 + wave) * 32 + lane;
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

// Entry point looked up by name from the host. Grid (N/128, M/128), block 128.
// Requirements: M % 128 == 0, N % 128 == 0, K % 2048 == 0.
// out != nullptr: write per-lane-tile transcripts (debug). Otherwise hash each tile in the epilogue and append tiles
// whose jackpot hash is <= bound to found[] (count, then 18-word records: t_rows, t_cols, 16 transcript words).
extern "C" __global__ __launch_bounds__(128) void xm_pearl_gemm(const int8_t* __restrict__ Ap,
                                                               const int8_t* __restrict__ Bp,
                                                               unsigned* __restrict__ out, int M, int N, int K,
                                                               Key8 a_seed, U256w bound, uint32_t* __restrict__ found,
                                                               uint32_t max_found) {
  pearl_gemm_rdna4_body<16>(Ap, Bp, out, M, N, K, a_seed, bound, found, max_found);
}
