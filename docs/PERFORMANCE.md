# Performance notes

Measured numbers and what we learned while tuning. Hashrates are at the power limits shown.

CPU use: the GPU workers wait on blocking-sync events (and HIP runs with hipDeviceScheduleBlockingSync), so a
worker sleeps while its kernel runs. RX 9070 XT Pearl: under 5% of one core (it was a full core spinning in the
runtime before 2026-10-22). One Pearl pass (16384x32768x4096) on the 9070 XT is 21.3 ms GEMM + 0.3 ms noise with
~0.1 ms of host time between passes.

## Pearl (pearlhash)

| Card | Power limit | X-Miner | Reference | Notes |
|---|---|---|---|---|
| RX 9070 XT (gfx1201) | -30% (~237 W) | 101.1 TH/s | SRBMiner 3.7.6: ~100.3 TH/s @ 244 W (2% fee) | per-lane hash tiles; Kryptex live: shares accepted |
| RX 9070 XT (gfx1201) | stock (~340 W) | 115.0 TH/s | WildRig 0.51.3: 107.8 TH/s @ 311 W cap | 0.34 TH/s/W stock, 0.43 at -30% |
| RTX 3060 (sm_86) | stock (170 W) | 44.4 TH/s | SRBMiner 3.7.1: 49.8 TH/s @ 170 W / 1785 MHz | persistent blocks + copy warp; Kryptex live: shares accepted |
| RTX 3060 (sm_86) | 127 W | 38.1 TH/s before the copy warp (42.3 as measured under WSL2, whose clock runs 10% slow - see below) | SRBMiner 3.7.1: 48.7 TH/s @ 127 W / 1755 MHz (re-measured 2026-10-22) | 128x256 blocks |

AMD per-lane hash tiles (2026-10-22): with the gfx12 WMMA layout a lane owns rows 8*(lane/16) + 0..7 of column
lane%16 in each 16x16 accumulator, so the AMD configuration uses rows {0..7, 16..23, 32..39, 48..55} x cols
{0, 16, 32, 48} (h*w = 128). The fold no longer needs a 16-lane DPP reduction and a select per tile every 128 k,
and the epilogue hashes one tile per lane with no LDS gather: 94.1 -> 100.8 TH/s at the same 237 W (+7%).
Verified by --selftest, the official verifier and live Kryptex shares.

AMD pass breakdown before that (RX 9070 XT, 16384x16384x4096): noise for A 0.29 ms, GEMM + fused jackpot check 11.6 ms.
The GEMM core alone is at parity with WildRig at the same power (pearl-rdna4 kernel).

NVIDIA findings (RTX 3060):
- With the transcript fold removed (timing only), the same GEMM runs at ~54-57 TH/s and higher clocks, so the
  fold, i.e. reading all int32 accumulators every 128 k, is where the time goes.
- Things that did NOT help: non-volatile mma, 3-stage pipeline, row-interleaved XOR, 8 warps with 32x64 warp
  tiles (same speed as 4 warps of 64x64), out-of-place accumulation at chunk boundaries (removes the WAR hazard
  between the fold and the next chunk's mma).
- Small wins: unconditional fold after a 2-stage unroll (+3%), reduce-scatter instead of per-tile butterflies,
  precomputed swizzled ldmatrix/cp.async addresses.
- Per-lane hash tiles (2026-10-22): Pearl allows periodic, non-contiguous tiles, so the NVIDIA configuration uses
  exactly the accumulators one lane owns (64x64 warp tile: rows {0,8,..,56} x cols {0,1,8,9,..,56,57}, h*w = 128).
  The fold becomes an in-register XOR with no shuffles: 35.9 -> 37.0 TH/s (32x64 warp tiles), 37.4 (64x64, 4 warps).
  The official reference miner does the same (rows [0,8], 64 columns). The "no fold" variant above likely let the
  compiler delete the MMAs (55 registers), so the fold was never the 40% it seemed.
- At the 127.5 W cap the card runs 1500 MHz (throttle reason: software power cap) and the kernel reaches ~86% of the
  int8 tensor peak per clock; the gap to SRBMiner is energy per MAC, not issue rate.
- 128x256 blocks (8 warps of 64x64, 72 KB dynamic shared memory): 37.4 -> 39.5 TH/s. 4 stages (96 KB) 38.4;
  rasterization groups of 4 / 16 rows 39.1 / 38.4 (8 stays).
- 128-byte k stages with a 2-stage pipeline (96 KB): 39.5 -> 41.8 TH/s (half the barriers per MAC). Fragment
  double-buffering, L2::128B prefetch hints, other rasterization groups and 256x128 blocks: no gain or worse.
- Noise apply rewritten (32-bit indexing, 8-byte vector access): 1.8 -> 0.65 ms per pass, 42.3 TH/s.
- Nsight Compute (2026-10-22, 16384x32768 pass, Windows): tensor pipe 81% busy, 2 warps per scheduler (223 registers
  and 96 KB shared memory allow one block per SM), stalls per issue: math-pipe throttle 8.7, fixed-latency wait 2.9,
  MIO 0.9, LG 0.5, barrier 0.2. L2 hit 54%, DRAM ~11 GB per pass. The idle 19% is latency that 2 warps cannot hide.
- 128-byte row padding (L2 aliasing) +1.6%; tiled A'/B' layout (16 KB slabs, replaces the padding) +2%:
  46.3 TH/s at 170 W under WSL Linux, 41.3 natively on Windows. Persistent blocks: worse (L2 hit 38%).
- **Correction (2026-10-23): there is no Windows gap.** On the test machine, WSL2's CLOCK_MONOTONIC runs at 0.900x
  real time (a 30 s sleep takes 33.4 s; miscalibrated TSC clocksource), so every hashrate measured under WSL2 here
  is 11% too high: 46.3 -> 41.7 TH/s, the same as native Windows (41.3). Proof: with the core locked at 1500 MHz,
  X-Miner's GEMM timed with CUDA events in a bare driver-API process runs 62.8 ms per 16384x32768x4096 pass
  (35.0 TMAC/s, 81% of peak) natively and under WSL2 alike, and register-only IMMA (100% of peak), streaming reads
  (341 GB/s) and random gathers match too, with 4 KB/64 KB or 2 MB (VMM) pages. The WSL2 figures elsewhere in
  this file remain valid for comparisons with each other, not as absolute rates. Real gap to SRBMiner: ~17%.
- Where the time goes (bare harness, locked 1500 MHz, kernel shape kept, 2026-10-23): register-only IMMA with 8 warps
  of 4x8 accumulators 99.6% of peak; + ldmatrix from shared memory 99.6%; + a barrier per 128 k 97.3%; + the XOR fold
  92.3%; real kernel 81.4%, of which the global -> shared copies cost ~7.7% even when the data sits in L2 (so not
  DRAM) and the BLAKE3 epilogue ~1%. Spreading each stage's copies over its first 3 k32 steps: 81.4 -> 83.3%
  (native Windows, same hour, 170 W: 40.44 -> 41.28 TH/s). L2 prefetch, BK 64 stages, and a deferred fold lose.
- Power is what limits the RTX 3060 (170 W cap at stock), so energy per MAC decides the hashrate. Board power at a
  locked 1500 MHz (harness, constant data; see the data note below for Pearl-like values): full kernel 77 W; without the global -> shared copies 51 W; with the
  data resident in L2 the copies still cost ~17 W, DRAM ~9 W more. Block grouping changes power (G 1: 87 W, G 4/8:
  77 W) but not speed. Bytes per MAC fall only with bigger block tiles, which 128-k double buffering caps at
  BM + BN <= 396 rows (99 KB of shared memory) and the register file at ~32K accumulators per SM. A change that
  is faster at a locked clock can be slower at the cap (an ordered-XOR fold: +1.6% at 1500 MHz, -1.2% at 170 W),
  so every kernel change is checked at stock settings too.
- **Data, DRAM drift and a copy warp (2026-10-23 night).** The bare harness now fills A'/B' with Pearl-like values
  (signal [-8, 7] plus the difference of two [-32, 31] noise terms); the constant pattern used before drew ~30% less
  power. At a locked 1500 MHz the kernel draws 110 W on Pearl-like data and 65 W on all-zero matrices: operand
  switching in the tensor cores and data paths is the largest variable cost, and it depends on the values (unsigned
  [0, 15] data 86 W, [0, 127] 104 W), not something a kernel can avoid. Tensor cores + ldmatrix alone on Pearl-like
  data reach 55.8 TMAC/s at 1942 MHz inside 170 W (+ the fold: 51.6), so SRBMiner's 49.8 is plausible; X-Miner sat
  at 43.6 because feeding the tensor cores cost ~34 W at 1500 MHz, ~18 W of it DRAM (all data L2-resident: 110 -> 92
  W). Nsight showed why DRAM was high: 8-10 GB per pass against a 4.4-5.0 GB minimum (L2 hit 64-70% instead of 81%),
  varying from run to run, because the blocks that share A/B slabs drift apart in k as the grid runs. Changes:
  - XOR fold in 4 independent chains instead of one 64-LOP3 dependency chain: +1.6% at the cap.
  - Persistent blocks (one per SM, tiles in the same grouped order) with a 9th warp that issues every global ->
    shared copy up to two stages ahead, across tile boundaries; mbarriers replace __syncthreads, so the math warps
    no longer issue copies or wait for each other (+5% at a locked 1500 MHz). The copy warps start each wave of tiles
    together (one global counter), which holds DRAM at 4.96 GB per pass (L2 hit 81.35%). Without that sync the copy
    warp's bursts push DRAM to 12 GB and the gain at the cap disappears (persistent blocks without a copy warp were
    already measured worse in 2026-10-22, L2 hit 38%).
  - Result, native Windows, stock 170 W, same hour: 41.35 -> 44.4 TH/s (+7.4%, incl. a branch-free lane-0 mbarrier
    arrive, +0.7%); --selftest, the official verifier
    and live Kryptex shares pass. Group rows 4/16, copy-warp back-off sleeps and an earlier copy schedule: slower.
  - Nine warps cap registers at 168 per thread (three warps share one scheduler's register file); the math warps fit
    once the ldmatrix offsets are derived from one base (swz(row, 2ks + c) = swz(row, c) ^ (ks << 5)).
  - Left: noise for A is 0.69 ms of a 49.9 ms pass (1.4%); the rest of the gap to SRBMiner (~12%) is per-clock
    efficiency (tensor pipe ~89% busy at 1500 MHz) and the energy of the remaining L2 -> SM traffic.
- Pearl allows no shortcut through the noise: E_A = E_AL x E_ARᵀ has rank r = 128, the same as the 128-k chunk
  that the transcript needs, so a low-rank product costs as many MACs as the direct one. r = 128 is also the
  optimum: larger ranks scale the share bound by 128/r, smaller ones fold more often.
- (Superseded by the correction above.) Windows gap: at the same clock (~1.78 GHz) the same kernel does 22.8K MAC/clk on Windows vs 25.6K under WSL; not
  the wait mode, the PTX JIT (XM_FORCE_PTX), pass size, carveout, TLB or L2 aliasing. SRBMiner does 49.9 TH/s
  (~97% per clock) on Windows at 170 W, so Windows is not the limit. Open.
  Further checks (2026-10-22): the gap is the same for short launches (4096x8192 passes, ~3 ms: 39.0 vs 42.8 TH/s),
  so it is not compute preemption by the desktop compositor (the test card drives a 4K 60 Hz desktop); it is the
  same for the compute-bound Parano1d kernel (8.7 vs 9.6 MH/s), so it is not memory; raising the process's WDDM
  GPU scheduling priority class (normal/high/realtime) changes nothing. Next: a Nsight Systems WDDM trace.
- SRBMiner cannot be profiled (it exits under Nsight).
- Per clock (2026-10-23, Nsight Compute, 16384x32768x4096): X-Miner 25.2 TH/s per GHz at 1815 MHz (81% tensor pipe,
  SM active = elapsed, 2.4 instructions per IMMA, issue 13%, stall mostly math-pipe throttle); SRBMiner 28.4 (~99%).
  No change from group rows 4/8/16, IMMA order, or the transcript in local memory (kept: 223 -> 211 registers).
  12 warps per block (192x256, BK 64 x 3, 168-register cap) spill ~45 registers past L1: 18 TH/s, DRAM traffic x2.
- SRBMiner sustains ~97% of the int8 tensor peak per clock (48.7 TH/s at 1755 MHz = 28.7K MAC/clk x 0.97) and still
  holds 1755 MHz inside 127 W, so its energy per MAC is ~25% lower than ours. Our loop is already lean (128 IMMA vs
  ~200 other instructions per 128 k, most of them the fold's XORs); finding the rest needs a profiler (Nsight).
- Windows vs Linux (same RTX 3060, same driver, same cubin, ~1750 MHz both): native Windows x-miner.exe runs the
  GEMM ~10% slower (60 vs 55 ms per pass): Pearl 35.5 TH/s, Parano1d 8.7 MH/s on Windows against 39.1 / 9.6 under
  WSL2 Linux. Not the wait mode (spin vs blocking identical), pass size or shared-memory carveout. SRBMiner reaches
  48.7 natively on Windows. Open question.
- CPU: blocking-sync waits work on native Windows (0.6 s CPU per 30 s); under WSL2 the GPU paravirtualisation
  driver polls with ioctls (~one core of system time) whatever the flags.
- Earlier: every variant sat at 35-36 TH/s and ~1600 MHz at the power cap, while SRBMiner holds 1760 MHz at the same
  power. Next to try: larger block tiles (128x256 / 256x128) with dynamic shared memory (needs
  cuFuncSetAttribute support in the CUDA backend) to cut L2 and shared-memory traffic per MAC.

## Parano1d (noid)

| Card | X-Miner | Reference |
|---|---|---|
| RX 9070 XT | 17.7 MH/s @ ~237 W (512 threads/block) | SRBMiner: 0 H/s; closed AMD miner: 20.9 MH/s on RX 9070 |
| RTX 3060 | 9.6 MH/s @ 127 W | closed miners: ~29.5 MH/s @ 90 W |

History: generic Karatsuba constant multiply 11.1 / 5.32 MH/s; shared-memory/LDS nibble tables (NoidMiner) 16.6 /
8.72 MH/s; AMD block size 512 17.7 MH/s. Two nonces per thread was slower on both vendors (15.3 / 8.30).
NVIDIA block size is flat from 256 to 768.

NVIDIA general multiply: 32x32 -> 64 carry-less products from IMAD.WIDE on operands split in 4 bit classes
(16 multiplies per 32x32, Karatsuba 32 -> 64 -> 128 = 9 of them) instead of NoidMiner's 16-bit XMAD path, which
costs a full IMAD plus half-word extraction per product on Ampere: 8.65 -> 9.6 MH/s. On AMD the same code is slower
(17.7 -> 14.9 MH/s: 32-bit multiplies are quarter rate; the 16-bit path uses the full-rate 24-bit multiplier).

Where the time goes on the RTX 3060 (timing-only builds): without the S-boxes 18.0 MH/s; without the table constant
multiplies 13.2 MH/s. The table multiply (64 conflict-free LDS.64 per product) is close to the LDS limit, so the
next step is fewer bytes per constant multiply: every constant (g, h, d_i + 1) lies in the GF(2^16) subfield, so in a
composite basis GF((2^32)^4) a constant multiply is 4 independent GF(2^32) products (32 LDS.32 from 512-byte nibble
tables, 4x fewer LDS wavefronts) while the general multiply keeps the 9 IMAD.WIDE products. Not done yet.

RX 9070 XT, same timing-only builds: 17.5 MH/s full, 24.9 without the table multiplies, 31.5 without the S-boxes.
There the LDS bandwidth ceiling is far away (~50 MH/s); the table multiply costs issue slots (64 LDS + ~85 ALU per
product), so the composite basis would save only ~5% on AMD (cheaper constant multiply, slightly dearer general
multiply) against an estimated ~15% on NVIDIA. With NOID at ~$0.54 (~1.1 NOID/day for a 9070 XT at 17.5 MH/s on a
656 GH/s network, 2026-10-22) this is parked behind Pearl work.

## Quantus (quantus)

| Card | X-Miner | Reference |
|---|---|---|
| RX 9070 XT | 182.9 MH/s @ ~297 W (stock power) | SRBMiner 3.7.9: 221.5 MH/s @ ~316 W |
| RTX 3060 | 148.9 MH/s @ 170 W | SRBMiner 3.7.9: 230 MH/s @ 170 W |

The hash is a Poseidon2 sponge (Goldilocks, width 12, rate 8, 8 full + 22 partial rounds, x^7) with five
permutations. The host absorbs everything that does not depend on the low 64 nonce bits once per job (the official
miner's prestate split), so the kernel finishes permutation 3, runs permutation 4 and compares the top 64 bits of
the first squeeze with the target: just under two permutations per nonce. Goldilocks products are reduced lazily: the
folds skip corrections that occur with probability ~2^-32 (a slip can only lose a share), and every candidate is
re-checked on the CPU with the exact arithmetic. AMD uses VCC carry chains (`v_add_co_u32` / `v_mad_co_u64_u32`),
NVIDIA PTX `.cc` chains; on NVIDIA part of the reduction runs on the multiply pipe so the integer pipe is not the
only one busy.

RX 9070 XT: the per-hash cost model (ALU ops plus ~2.4 add slots per 64-bit multiply-add) matches the measured
rate to 2%, so the kernel runs at the issue rate. RTX 3060: compute-bound per clock (0.076 MH/s per MHz); locking
the core clock at 1500 MHz gives 113 MH/s at ~80 W, the better rate per watt. Tried and slower: several nonces per
thread (register pressure), independent-product multiplies, linear layers on the multiply pipe, unrolling, other
block sizes.

## BLAKE2b (Bitcoin Knots RDTS)

| Card | X-Miner | Reference |
|---|---|---|
| RX 9070 XT | 4.63 GH/s @ ~237 W (-30% power) | original Python/OpenCL X-Miner: ~4.2 GH/s |
| RTX 3060 | 1.97 GH/s @ ~157 W | |

One BLAKE2b compression per nonce (80-byte header); the host precomputes the three round-1 column steps that do not
touch the nonce word, and the kernel forms only the first output word (the rest of the last round is dead code).
Every candidate is re-checked on the CPU.

RX 9070 XT: rotates written as shifts compiled to ~2.9 VALU instructions each; `v_alignbit_b32` per half brings them
to 2 and the kernel from ~2500 to ~2215 VALU instructions per hash (3.83 -> 4.63 GH/s). At 4.63 GH/s that is
~10.3 T lane-ops/s, the single-issue peak at the clocks this power limit allows; the remaining instructions (64-bit
adds with carry, XORs, alignbits) have no VOPD dual-issue form. RTX 3060: ~2030 SASS instructions per hash (3-input
IADD3 adds, SHF funnel rotates), bound by the integer pipe.
