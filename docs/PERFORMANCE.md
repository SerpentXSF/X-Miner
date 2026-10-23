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
| RTX 3060 (sm_86) | 127 W | 38.1 TH/s (42.3 as measured under WSL2, whose clock runs 10% slow - see below) | SRBMiner 3.7.1: 48.7 TH/s @ 127 W / 1755 MHz (re-measured 2026-10-22) | 128x256 blocks |

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
