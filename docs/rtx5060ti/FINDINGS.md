# X-Miner on RTX 5060 Ti (sm_120): findings log (2026-11-20)

All at core lock 2752 (reads 2745) MHz, VF offset +300, PL 150, memory 810 MHz unless noted. Tensor peak = 101.3 TMAC/s.

## Steady-state numbers (4-minute runs, power from the last 90 s; short cool runs read ~10 W low)
| build | mode | TH/s | W | GH/W |
|---|---|---|---|---|
| X-Miner 0.5.0 stock | memory unlocked | 88.57 | 148.7 | 596 |
| w8 kernel (xm_pearl_gemm_w8) | memory unlocked, stock shape | 90.97 | 150.0 (PL cap, 2685 MHz) | 606 |
| w8 fused (xm_pearl_mine_w8) | mem 810, 2304x8192x2048, batch 128 | 83.05 | 131.8 | 630 |
| a closed-source miner (reference) | mem 810 | 100.95 | 109.0 | 926 |

## Where the power goes (pw.py, steady state)
- pure MMA loop, fixed operands, 100% of peak: 89 W; MMA + shared-memory operand loads: constant data 57 W, Pearl-like data
  (63/64 + triangular noise) 101-103 W at ~99% of peak. Signal offsets other than 63/64 (0, -64, 32, mixed A/B): 103-107 W.
  => the tensor math itself costs ~1.03 W per TH/s with real data; the closed-source reference miner as a whole is ~6 W above that floor.
- copy path alone (cp.async L2 -> shared): 25 W per TB/s with Pearl-like data (15 W/TB/s with constant data).
- w8 harness (2304x8192x2048): 140 W at 89.4% with copies, 107 W at 92.5% without => copies cost ~33 W.
- direct global loads (v2): 113 W at 78% (data path cheaper, 0.25 W/TH, but IMMAs wait on LDG even one step ahead).
- kernel "active" floor (clocks running, polls): ~25 W; idle 8 W.
- m16n8k32 is the only sensible int8 MMA shape (k16: half the rate, m8n8k16: a fifth).

## Instruction costs next to MMAs (8 extra per MMA, % of peak)
free: xor, or, and, shl, shr. Costly: add/sub 84%, byte_perm (PRMT) 66%, mul-add 66%, min 52%, funnel rotate 71%.

## What a closed-source miner submits (its own share proofs, read from my plain-TCP stratum traffic through a local relay)
m=17922, n=62976, k=65536, noise_rank=128. Every signal byte in the revealed rows is 64 (A and B): the matrices are virtual
(1.2 GB / 4.1 GB declared, never stored). a_rows: 16 rows, stride 8 (one lane's rows of a 128-row block); b_cols: 12 =
{c, c+1} + 64 j, j < 6 (block of 384 columns, 8 warps x 48 columns interleaved by 64). Lane tile 16 x 12 = 192
accumulators; block tile 128 x 384; proofs are 1.87 MB raw, ~5.4 KB zstd. m is not a multiple of 128 (17922 = 140*128 + 2):
probably the nonce is in the tail rows / the row count.
=> it moves no matrix data: operands are rebuilt in the SM from the uniform noise table (128 bytes per row) and the pairs.

## In-SM operand generation experiments (mb/mb6.cu, mb/mb7.cu)
- Scalar byte gathers by helper warps (mb6): 240K cycles per 48 KB stage (needs ~4K), tensor down to 81-43%. Dead.
- Vectorised by the math warps (mb7): column-major U tables, word = 0x40404040 + U[col p0][r..r+3] - U[col p1][r..r+3],
  4x4 byte transpose with shifts (free), row-major stage buffer of one k-step (512 rows x 32 k), tile 128 x 384.
  Ceiling without generation: 99.8% of peak with 192 accumulators per lane (241 registers).
  With generation: 43% (no sync) - Nsight: mio_throttle + short_scoreboard, i.e. shared-memory bank conflicts
  (column pitch a multiple of 128 bytes puts all loads of a warp in 4 banks; stores of different row groups alias too).
  A per-k-step "everyone has read the buffer" barrier costs 26 points on its own (warps are skewed by >1 k-step).
  Next: conflict-free layout (U column pitch 132, stage row pitch 36, lanes = 8 row groups x 4 k groups), 3 stage buffers
  instead of a blocking read barrier, premultiplied offsets + OR addressing to drop the address adds.

## Other dead ends (all measured)
hash interleaved with MMAs, hash staggered between warps, hash helper warp, prefetch warps, direct fragment loads, 4-buffer
BK64 pipeline, refill two stages ahead, early buffer release, sleeping polls, pinned loads via block boundary, larger K in
memory-unlocked mode, K=1024 (+1%), cache-blocked big-A (71 TH/s at mem 810), apply-noise via cp.async, lane=row stores.
Kept: w8 kernel (+2.7% steady state in memory-unlocked mode), fragments one k-step ahead (W8_FRAG_AHEAD, +1.5% in the
harness, +0.9% memory-unlocked, -1.8% in the fused path), fused batch launch, merged noise prep, threadfence grid sync.

## Generation microbenchmarks, continued (mb/mb8.cu, mb/mb9.cu; tile 128 x 256, 8 warps)
- mb8: conflict-reduced layout + 3 stage buffers (built/read mbarriers): the read barrier is now free (57.2% with it,
  52.9% without), generation itself still halves the tensor rate.
- mb9: tables at fixed shared addresses (UA 0x4000, UB 0x8000) so gather addresses are ORs, one SUB per word
  ((x | 0x40404040) - y), delta-swap transpose, lane = row group. Ceiling without generation 99.1%; with generation 57.3%.
  Nsight (mb/mixprof.sh): per IMMA still 1.0 SHF, 0.66 IMAD, 0.56 IADD (address arithmetic and the shifts compile to
  costly ops), 1.75 LDS + 0.38 STS, 32% excessive shared wavefronts, memory pipes 64% busy, math_pipe_throttle dominant.
  => per-k-step gathers through shared memory are too heavy as written. Open ideas: hoist every address (no IMAD/IADD
  in the loop), check which SASS the shifts become (SHF is costly, plain SHL/SHR were free in mb.cu), generate per
  128-k stage instead of per k-step, or find how to get fragment-ordered data with fewer shared accesses.
