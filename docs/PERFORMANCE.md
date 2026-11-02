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
  Re-confirmed 2026-10-31 for integer work (Quantus-like multiply-add probe, same cubin, driver API): native Windows
  and WSL2 both 22.88 M SM clocks per launch at 1940 MHz, CUDA-event time 11.96 vs 11.94 ms; only WSL2's host clock
  disagreed (10.84 ms per launch; `sleep 30` = 33.5 s, while the NTP-synced wall clock stays right). So the later
  "WSL2 ~10% faster on every algorithm" readings were this clock too: the native Windows rates are the true ones.
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
- (Superseded by the correction above.) Windows vs Linux (same RTX 3060, same driver, same cubin, ~1750 MHz both): native Windows x-miner.exe runs the
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
| RX 9070 XT | 20.1 MH/s @ ~236 W (-30% power) | SRBMiner: 0 H/s; closed AMD miner: 20.9 MH/s on RX 9070 |
| RTX 3060 | 9.8 MH/s @ 148 W | closed miners: ~29.5 MH/s @ 90 W |

History: generic Karatsuba constant multiply 11.1 / 5.32 MH/s; shared-memory/LDS nibble tables (NoidMiner) 16.6 /
8.72 MH/s; AMD block size 512 17.7 MH/s; IMAD.WIDE products on NVIDIA; composite basis (below) 20.1 / 9.8 MH/s
(same session, same settings: 17.7 -> 20.1 and 8.61 -> 9.81 MH/s, 3060 power 169 -> 148 W).

NVIDIA general multiply: 32x32 -> 64 carry-less products from IMAD.WIDE on operands split in 4 bit classes
(16 multiplies per 32x32, 9 of them per 128-bit product) instead of NoidMiner's 16-bit XMAD path, which costs a full
IMAD plus half-word extraction per product on Ampere. On AMD the 16-bit path stays (32-bit multiplies are quarter
rate there; the 16-bit products use the full-rate 24-bit multiplier).

Composite basis (kernels/noid/noid_comp.h): the kernel represents GF(2^128) as K[z]/(z^4 + z^3 + z^2 + y z + y) over
K = GF(2^32) = GF(2)[y]/(y^32 + y^7 + y^3 + y^2 + 1); the host converts the midstate, fields, nonce bits and round
constants, and maps the digest back. Every MDS constant (g, h, d_i + 1) lies in GF(2^16), inside K, so a constant
multiply is four K-products by a constant: 16 lookups into 24 KB of byte tables instead of 64 LDS.64 into 48 KB of
nibble tables. The general multiply stays at 9 carry-less 32 x 32 products; the z^4..z^6 terms fold back with
one-bit shifts and each coefficient is reduced once. RTX 3060 profile (Nsight Compute), flat -> composite: instructions
per nonce -16%, shared loads 4x fewer, registers 72 -> 64, occupancy 33% -> 66% (the tables no longer cap it); the ALU
pipe stays the limit (~75% of peak, math-pipe throttle), now almost all in the S-box multiplies. Block size is flat on
both vendors (3060 128-1024, 9070 XT 128-1024 within 2%).

Dead ends on the RTX 3060 after the composite basis (2026-10-25): 6 bit classes so the multiplier sums all class
products (36 IMAD.WIDE per 32 x 32, no XOR tree) 9.81 -> 6.96 MH/s (logic ops barely drop, the multiplies double);
right shifts as multiply-high on the idle multiply pipe 9.82 -> 9.72-9.75; threads per block 128-1024 flat. Moving
work between pipes does not help: the profile's dispatch stalls point at issue limits, not at one busy pipe.

Where the time went on the RTX 3060 with the flat basis (timing-only builds): without the S-boxes 18.0 MH/s; without
the table constant multiplies 13.2 MH/s. RX 9070 XT, same builds: 17.5 MH/s full, 24.9 without the table multiplies,
31.5 without the S-boxes. The closed NVIDIA miners are ~3x faster still, so they must have a much cheaper 128-bit
multiply than 9 class-split integer products; that is the open item. With NOID at ~$0.54 (~1.1 NOID/day for a
9070 XT at 17.5 MH/s on a 656 GH/s network, 2026-10-22) this stays behind Pearl work.

## Quantus (quantus)

| Card | X-Miner | Reference |
|---|---|---|
| RX 9070 XT | 195.8 MH/s @ ~330 W (stock power), 179.6 MH/s at -30% | SRBMiner 3.7.9: 221.5 MH/s @ ~316 W |
| RTX 3060 | 173.9 MH/s @ 169 W (generated SASS; ptxas build 151.6), 127.5 MH/s locked at 1400 MHz | SRBMiner 3.7.9: 228 MH/s @ 170 W |

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

NVIDIA carry corrections without SEL (2026-10-27): the 2^64 carry of a fold or add is applied as an `addc` into the
high word and `lo + (t - hi)` (one IADD3) instead of materializing the carry with SEL plus two adds. SASS per
permutation tail 14612 -> 13918 instructions (-4.7%; partial round 282 -> 267), but only +1% on the RTX 3060: 148.0
-> 149.4 MH/s at the 170 W cap, 117.0 -> 118.2 MH/s locked at 1500 MHz (82 -> 80 W). The saved instructions are
integer-pipe adds; the kernel is held by the multiply pipe, and the multiply count (about 736 64-bit products per
permutation, random diagonal and round constants) is fixed by the algorithm. Reduction mix re-checked after the
change: XM_QTC_MIX 2 still best (0: 144.8, 3: 146.0, 6: 146.9 MH/s).

Generated SASS on RTX 30 (2026-10-31, tools/qgen): the sm_86 kernel's permutations are written as scheduled SASS
and assembled with CuAssembler. Nonce setup and the target check stay ptxas. The steps below were measured as SM
clocks per warp-permutation (checked on the GPU against the CPU permutation, 0 of 262144 states wrong):

| Step | Clocks | vs ptxas |
|---|---|---|
| ptxas | 24.14k | 1.00x |
| straight-line generator | 26.6k | 0.91x |
| multiplies split into head (4 `IMAD.WIDE`) and tail (reduction), one tail interleaved with the next head on alternating registers and carry predicates | 23.6k | 1.02x |
| top words of the 96-bit sums on the multiply pipe | 22.54k | 1.07x |
| rounds as loops with `LDC` round constants (the straight-line code stalled on instruction fetch: Nsight no_instruction 3.3 -> 0.03) | 21.40k | 1.13x |
| three-input adds in the linear layers | 20.99k | 1.15x |

Moving any further work between the ALU and the multiply pipe, in either direction, measured slower.

The latency model is GA106 measured with a hand-SASS harness:
- 4 cycles within a pipe, 5 across the ALU and multiply pipes;
- 6 cycles from WIDE to WIDE, and for a WIDE carry read on the multiply pipe;
- 13 cycles for `IMAD.HI`, and from `ISETP` to a branch.

In the miner (RTX 3060, native Windows, 170 W cap):
- `--benchmark`: 151.6 -> 169.6 MH/s (+11.9%) at the same 169 W, every share CPU-verified;
- self-test passes;
- Kryptex: 11.5 minutes, 7 accepted / 0 rejected at 167.6 MH/s.

Further steps (2026-11-01), measured in the miner at a locked 1400 MHz core clock (`--gpu-cclock 1400`). That is
reproducible to 0.01 MH/s; at the 170 W cap the clock drifts and runs vary by about 1%.

| Step | MH/s at 1400 MHz |
|---|---|
| first generated kernel | 123.35 |
| permutation 4's last round computes only s0 (the shared terminal loop runs 3 rounds there, then the S-boxes and s0's row of the external layer) | 124.34 (+0.8%) |
| low and high registers of 6 state elements swapped, so three-input adds rarely read three registers of one bank (868 -> 128 such instructions per nonce) | 125.20 (+0.7%) |
| diag2's two one-carry adds on the multiply pipe | 125.79 (+0.5%) |
| full-round constants folded into the external layer: w = x + z + M4^-1 rc as one three-input add per element, instead of a separate add after the fold (22,860 -> 22,140 instructions per nonce) | 127.54 (+1.4%) |

Not kept:
- operand reuse flags: +0.05%;
- squares with 3 `IMAD.WIDE` (x^2 and x^4 in the S-box): +0.07%;
- further carry adds onto the multiply pipe: -3.4%;
- carry adds and 96-bit top words from the multiply pipe to the ALU: -0.9% to -7.9%.

Nsight Compute on one launch:
- FMA-heavy pipe 84% active, ALU 60%, issue 58%, 7.9 warps per scheduler;
- top stall is math-pipe throttle (5.7 per issue), then not-selected 3.4 and dispatch 2.3.

Taking work off the FMA pipe still measures slower, so the counters do not name the limit on their own.

Instruction costs on the RTX 3060 (GA106): instruction mixes timed in a loop inside the Quantus kernel, at 1400 MHz.
Costs are SMSP cycles per warp instruction.

| Instruction | Cycles |
|---|---|
| `IMAD.WIDE.U32` | 4 |
| `IMAD.HI.U32` | 4 |
| `IMAD`, `IMAD.X` | 2 |
| `IADD3` / `IADD3.X` (also with three sources and two carries) | 2, ALU |

- Multiply-pipe and ALU instructions overlap fully in independent mixes.
- **Register-bank penalty (+1 cycle):** the penalty also delays ALU issue in mixed code. It applies to:
  - an `IMAD.WIDE` whose two register sources are different registers of one bank (`a * a` is free);
  - an `IMAD.WIDE` with a 64-bit addend whose first source is an even register;
  - an `IADD3` with three different registers of one bank.

  The kernel has about 1,800 such `IMAD.WIDE` per nonce. Removing them from an S-box round takes it from 987 to 980
  cycles, so they are a small part of the gap.
- **The kernel's instruction mix accounts for about 91% of its time.** Per nonce it runs:
  - 5,890 `IMAD.WIDE`;
  - 4,828 `IMAD.X`;
  - 11,861 ALU instructions.
- **Phases measured as separate loops:**
  - an S-box round and a partial round are bound by the multiply pipe (987 and 356 cycles);
  - the external layer is bound by the ALU (290 cycles).

  Moving mul3's carry adds to the ALU makes the S-box round slower (1050 cycles), which the cycle counts above do not
  predict.
- **Warp lockstep ruled out:** starting half the warps up to ~11 us late changes nothing. So the gap to SRBMiner needs
  fewer instructions, not other scheduling.

At the 170 W cap the kernel goes from 169.0 to 171.8 MH/s (+1.6%, 3 paired runs); Kryptex 11.5 minutes, 10 accepted /
0 rejected at 170.9 MH/s. Locked at 1400 MHz the card
does 125.8 MH/s at 69 W: 1.82 MH/s per W, against 1.01 at the cap.

Folding the round constants: 172.1 -> 173.9 MH/s at the cap (+1.1%, 3 paired 30 s runs); Kryptex 11.5 minutes,
5 accepted / 0 rejected at 172.8 MH/s.

The rest of the gap to SRBMiner (228 MH/s) is kernel time, not the platform: see the WSL2 clock note above.

## KawPow (Ravencoin)

| Card | X-Miner | Reference (same card, same pool, 2026-10-28) |
|---|---|---|
| RX 9070 XT (-30% power) | 34.0 MH/s @ 237 W | WildRig 0.51.3: 34.1 MH/s @ 238 W |
| RTX 3060 (170 W) | 20.6 MH/s @ 169 W | T-Rex 0.26.8: 20.85 MH/s @ 169 W |

Epoch 613 (height ~4.6 M), a 6.2 GB DAG. Each hash reads 64 x 256 bytes of DAG, so at these rates the 3060 moves
~330 GB/s of its 360 GB/s and the 9070 XT ~560 of ~640 GB/s: KawPow is a memory-bandwidth algorithm and the kernels
of every miner end up within a few percent of each other. With a wrongly sized 1.9 GB DAG (an int overflow in the
size formula, fixed before release) the 9070 XT ran 39.6 MH/s: the larger DAG costs it ~14% (address translation).

Settings (40 s benchmarks): RX 9070 XT `--gpu-mclock 2700` 33.95 -> 36.35 MH/s (+7.1%), 2800 unstable (18.2);
`--gpu-voffset -50` and `--gpu-coffset -300` no change. RTX 3060 `--gpu-moffset 500` 21.28, `1000` 21.99 MH/s (+6.9%),
and with `--gpu-cclock 1500` 21.91 MH/s at 135 W instead of 169 W.

Design: the program of each 3-block period is generated as CUDA / HIP source with every register index and operation
fixed, compiled at run time (NVRTC / hipRTC; the next period compiles in the background), one hash per 16 lanes with
the 16 KB cache in shared memory, and the DAG generated on the GPU from the light cache (512 parents per item).

Neurai (`neurai`) is KawPow unchanged (NeuraiProject/Neurai src/crypto/ethash: the same padding words, epoch length
and sizes), at a younger chain: epoch 246, a 3.1 GB DAG. RX 9070 XT 39.4 MH/s @ 237 W (the smaller DAG costs fewer
address translations, as above), RTX 3060 20.6 MH/s @ 170 W (bandwidth-bound either way).

## FiroPow, ProgPoW 0.9.2 (Epic Cash) and ProgPowZ (Zano)

| Card | FiroPow (epoch 650, 6.7 GB) | Epic Cash (epoch 125, 2 GB) | Zano (epoch 129, 2 GB) |
|---|---|---|---|
| RX 9070 XT (-30% power) | 33.1 MH/s | 39.9 MH/s | 40.0 MH/s |
| RTX 3060 (170 W) | 20.5 MH/s | 20.6 MH/s | 20.6 MH/s |

The same engine as KawPow. 0.9.2 / ProgPowZ have 50-block periods, 12 cache reads and 20 math operations per round
and a 256-parent DAG, so the smaller DAG fits the 9070 XT's caches better than KawPow's 6.2 GB. Published: WildRig
41.1 MH/s for Zano on a 9070 XT at 314 W; SRBMiner / WildRig 23.3 MH/s on an RTX 3060 at 132 W with a +2000 memory
offset. 180 s benchmarks found 106 and 97 shares where 107 were expected (all CPU-verified, none rejected).

Epic Cash on epic.pool-pay.com: the node's share difficulty is 200,000 (one share per 200,000 hashes, ~150 per second
on the 9070 XT) and each share is credited at that difficulty, but the pool answers about 60 per second per
connection and drops the rest (measured through a logging proxy: 10,944 replies to 25,970 submits in 3 minutes, up
to 9 s late). X-Miner sends at most 50 per second for the current height and re-checks only that many on the CPU, so
the pool credits ~9-10 MH/s per connection (live: 9,790 accepted / 161 rejected in 4 minutes).

## Etchash (Ethereum Classic)

| Card | X-Miner |
|---|---|
| RX 9070 XT | 35.5 MH/s @ ~205 W |
| RTX 3060 | 41.0 MH/s @ 155 W |

Height 25.6 M, epoch 427, a 4.4 GB DAG. Each hash is a chain of 64 dependent 128-byte DAG reads. RTX 3060: 8 lanes
per hash (a lane reads 16 of the 128 bytes) 41.0, 4 lanes 38.6, 2 lanes 35.9, 1 lane 16.6 MH/s. RX 9070 XT: 35.5 MH/s
for 1-8 lanes, any block size and with a reciprocal-multiply modulo; 41.4 MH/s with a 1 GB DAG, power below the cap.
Like KawPow (2.2 G reads/s of 256 B) it sits at ~2.3 G random reads per second, so on this card the limit is the
memory's random-access rate, which a memory overclock raises.

EthereumPoW (`ethash`) is plain Ethash on the same engine (no ECIP-1099: 30000-block epochs, the cache from the epoch's
own seed): height 25.36 M, epoch 845, an 8.1 GB DAG. RX 9070 XT 32.5 MH/s @ ~200 W (the bigger DAG costs it more
address translation than ETC's), RTX 3060 41.1 MH/s @ 157 W (live).

## Autolykos v2 (Ergo)

| Card | X-Miner | Table build (per block) |
|---|---|---|
| RX 9070 XT | 64.5 MH/s @ ~199 W | 4.8 s |
| RTX 3060 | 96.3 MH/s @ 137 W | 8.9 s |

Height 1,904,505, N = 238.6 M, a 7.6 GB table; Ergo blocks are ~2 minutes, so the table build costs ~4% (9070 XT) to
~7% (3060). The search is 3 BLAKE2b compressions and 33 random 32-byte reads per nonce. Element reads as two 16-byte
loads plus the read batch the compiler may form: RTX 3060 unroll 1 96.3, 2 95.6, 4 85.6, 8 64.6 MH/s (registers vs
occupancy); RX 9070 XT unroll 8 64.5, 1-4 62.8 MH/s. The first version (four 8-byte loads, unroll 4) ran 68.0 / 62.8.

## Octopus (Conflux)

| Card | X-Miner 0.2.0 (matrix) | X-Miner 0.1.0b (scalar) | Published figures for other miners |
|---|---|---|---|
| RX 9070 XT (-30% power) | 58.5 MH/s @ 236 W | 42.7 MH/s @ 237 W | 79 MH/s @ 230 W |
| RTX 3060 (170 W cap) | 31.4 MH/s @ 169 W | 24.2 MH/s @ 169 W | 42 MH/s @ 140 W |

Height 160.2 M, stage 305, an 8.8 GB DAG. Per nonce, 32 values of a degree-1023 polynomial mod p = 1032193 (shared
coefficients per group of 32 nonces), then 32 dependent 256-byte DAG reads (8 KB per hash: 3060 ~44 MH/s, 9070 XT ~78
MH/s at full memory bandwidth).

The values a group of 32 nonces needs are its polynomial at all 1024 header points, i.e. one row of a matrix product:
coefficients (groups x 1024) times the job's Vandermonde matrix V[i][k] = x_k^i (1024 x 1024). Both factors are below
2^20 and go in as three bytes each, so it is nine u8 x u8 -> s32 products (3060: mma.sync m16n8k32; 9070 XT: WMMA
16x16x16 iu8) summed by byte shift and reduced once: about 295 k int8 multiply-adds per nonce instead of 32 k
scalar multiply-add-reduce steps. Per 2^21-nonce batch (serial times): 3060 matrix 19 ms + DAG mix 55 ms; 9070 XT
matrix 4.75 ms + mix 33 ms. The mix reads at ~87% (3060) / ~81% (9070 XT) of DRAM bandwidth, so it is the limit
now: mix alone would be ~44 / ~63 MH/s.

The next batch's coefficients and matrix products run on a second, high-priority stream while this batch's mix runs.
On the 3060 that only pays when the matrix kernel is persistent and small (two 4-warp blocks per SM, 31.4 MH/s; one
block per tile: 26.3; a cp.async double-buffered version with 46 KB of shared memory per block: 26.8, it takes L1
from the mix); the card is then at its 170 W cap. On the 9070 XT the overlap does not help and one block per tile is
best for separate launches (52.6; two per WGP: 44.4), but the AMD runtime on Windows runs a second stream's work
after the first's (a CU-masked second stream changed nothing), so AMD uses one fused launch instead: the first 128
blocks compute the next batch's matrix products (persistent), the rest run this batch's mix (58.5 MH/s; 64 blocks:
53.5, 256: 54.6). A fused kernel reserves the same registers and LDS for every block, so the matrix part must not
need more than the mix: the operands are stored in WMMA fragment order (a wave reads each 16 x 16 fragment as one
256-byte load from L2 / Infinity Cache, two 16-k steps in flight) and it uses no LDS. That layout also made the matrix
products faster on their own (4.75 vs 6.3 ms; ~67% of int8 peak). An LDS-staged fused version reached 55.5 (214
VGPRs and 23 KB LDS per block cut the mix's waves). The scalar path remains for other GPUs and as `XM_OCTOPUS_TC=0`.

Memory clock on the RX 9070 XT (-30% power; 60-120 s benchmarks, every share CPU-verified): stock 2518 MHz 58.6-58.8,
`--gpu-mclock` 2600 60.5, 2650 61.7, 2700 63.0 (+7.3%), 2725 63.4, 2750 64.0, then a cliff: 2775 52.9, 2800 39.9 MH/s
(the memory starts retrying; no bad results reach the pool, it just gets slower). Speed follows the memory clock almost
1:1, power stays at ~234 W. Adrenalin's fast memory timing (Overdrive8 setting 14) changed nothing (62.90 vs 62.89 at
2700). 2700 leaves a margin below this card's cliff; live on Kryptex 16 min at 62.3 MH/s, 3 accepted / 0 rejected.

## FishHash (Iron Fish) and KarlsenHashV2 (Karlsen)

| Card | X-Miner | Published (Kryptex) |
|---|---|---|
| RX 9070 XT | 23.5 MH/s @ ~190 W; 25.2 with `--gpu-mclock 2700` | 30 MH/s @ 185 W (settings unknown) |
| RTX 3060 | 27.5 MH/s @ 140 W | 31 MH/s |

A hash is 32 rounds of three random 128-byte reads from the 4.5 GB dataset (96 reads, 12 KB). The RX 9070 XT gives
23.53 MH/s for 4 or 8 lanes per hash and blocks of 64-256 threads: 2.26 G random reads per second, the same ceiling as
KawPow and Etchash on this card, raised only by the memory clock. Karlsen runs at the same rate.

## QHash (Qubitcoin)

| Card | X-Miner (float) | X-Miner (double, exact) | Published |
|---|---|---|---|
| RX 9070 XT (-30% power) | 2.51 GH/s @ 237 W | 1.11 GH/s | - |
| RTX 3060 (170 W cap) | 0.96 GH/s @ 147 W | 0.46 GH/s | 0.70 GH/s @ 60 W |

QHash is SHA-256(header) -> 16-qubit circuit -> SHA-256. The reference simulates the circuit on a 2^16-amplitude
state vector (cuStateVec); X-Miner evaluates it exactly in ~200 floating-point operations: after the first rotation
layer and CNOT chain the state is a chain with a 2-dimensional bond, the second layer's RZ drops out of Z measurements,
and the second CNOT chain turns each <Z_j> into a parity of the first j + 1 qubits, so every expectation is a
contraction of 4-component transfer steps (bond density in the Pauli basis). Per hash the GPU does three SHA-256
compressions (the header's second block from a host midstate with its first three rounds precomputed, the result
block, and the constant padding block with a precomputed schedule) plus the contraction, so it is integer (SHA) bound.
Float rounds the Q15 value differently from the double reference on ~0.7% of hashes (cancellation in the last steps);
those can only be missed shares, since the CPU re-checks every candidate. Double runs at 1/16 - 1/64 rate on these
cards, which costs more than it saves.

## XelisHashV3 (Xelis)

| Card | X-Miner | Published |
|---|---|---|
| RX 9070 XT (-30% power) | 5.0 kH/s @ ~170 W (4,096 hashes in flight) | - |
| RTX 3060 (170 W cap) | 4.49 kH/s @ 112 W (17,728 in flight) | 5.5 kH/s @ 80 W |

Each hash fills its own 543,744-byte scratchpad, then makes 67,968 strictly sequential steps, each a chain of dependent
random 8-byte reads (and two read-modify-writes) in that scratchpad, with 16 kinds of 64/128-bit arithmetic chosen by
the data. It cannot be parallelised within a hash, so the GPU runs thousands of hashes at once and is bound by memory
latency and cache / TLB misses, at low power. RX 9070 XT by hashes in flight: 2048 3.91, 3072 4.71, 4096 5.00, 6144
5.04, 8192 4.08, 16384 3.59 kH/s (falls once the scratchpads thrash the caches). RTX 3060: 1024 1.77, 2048 2.50, 4096
2.68, 8192 3.50, 12288 4.04, 16384 4.36, 17728 4.49 kH/s (keeps rising to the memory limit). Splitting the walk into
slices (no launch over ~0.5 s, for the Windows display watchdog), L2-only loads and block size changed nothing.

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

## Clock, voltage and memory settings (2026-10-27)

60 s `--benchmark` runs, alternated back to back (RX 9070 XT at Adrenalin's -30% power, ~237 W; RTX 3060 at its
170 W cap, Windows):

- RX 9070 XT `--gpu-voffset -50`: Quantus 181.2 / 184.1 -> 191.1 / 191.0 MH/s (+4.5%), Parano1d 20.17 -> 20.93 MH/s
  (+3.8%), BLAKE2b 4.63 -> 4.79 GH/s (+3.5%), all at the same power. Pearl: +4-6% in the benchmark (101.3 / 103.0 ->
  107.2 / 107.3 TH/s, -100 mV no better) but no gain while mining real jobs (5 min A/B on Kryptex: 102.9 / 101.1 at
  0 mV, 100.7 / 101.1 at -50 mV). Pearl's clock at the power cap moves with the job data (1993-2030 MHz).
- RTX 3060 memory lock (fixed clocks 7501 / 7301 / 5001 / 810 / 405 MHz): Quantus at 810 MHz 149.8 vs 147.9 MH/s
  (+1.3%, core 1927 vs 1897 MHz); Parano1d and BLAKE2b same speed with 15 W less (148 -> 133 W, 161 -> 146 W). Pearl
  needs the memory: locked 7301 MHz 41.1, 5001 MHz 32.2, unlocked 44.5 TH/s.
- Pearl intensity 64 (16384 x 65536 columns per pass) vs 32: RX 9070 XT 103.0 -> 103.9 TH/s, RTX 3060 44.47 -> 44.80
  TH/s; now the default. Quantus intensity 26 vs 24: no change (148.1 vs 147.9 MH/s).

## NexaPow (Nexa)

Per nonce: SHA256d of the commitment and nonce (the mining hash, used as the private key), SHA256 of that (h1),
the RFC6979 nonce k (an HMAC-SHA256 DRBG: 16 SHA-256 compressions), P = x G and R = k G, the quadratic-residue test
on R.y, e = SHA256(R.x | P | h1), s = k + e x, then SHA256 of the signature.

The kernel:
- **Scalar multiplications:** a signed fixed-base comb with no doublings. A 20-bit window gives 13 point additions per
  scalar (Jacobian + affine, 7M + 4S). The 436 MB table is built at start on all CPU cores.
- **Field arithmetic:** 9 limbs of 29 bits, so each product is one 64-bit multiply-add and the carries are resolved
  once per multiply.
- **Inversion:** one per block, by Montgomery's trick over the block's z values. The inverse itself uses
  Bernstein-Yang divsteps on thread 0, overlapped with the other warps' quadratic-residue test.
- **Quadratic-residue test:** divsteps with Jacobi tracking (the method of libsecp256k1's jacobi32_maybe_var).

Steps on the RTX 3060 (170 W) and RX 9070 XT:

| Step | RTX 3060 | RX 9070 XT |
|---|---|---|
| first working kernel (8 x 32-bit limbs, 8-bit comb, Fermat inverse and Legendre) | 5.5 MH/s | wrong (compiler) |
| 16-bit comb, scalar shifting instead of indexed arrays | 7.2 | |
| 9 x 29-bit limbs | 10.6 | |
| 2 blocks per SM (register budget) | 15.0 | |
| binary Jacobi instead of the Legendre exponentiation | 17.1 | 17.6 |
| block inversion overlapped with the other warps | 17.5 | |
| 20-bit comb (22-bit: +2% for a 1.6 GB table) | 19.9 | |
| divsteps Legendre | 21.5 | |
| divsteps inverse (removes the block stall) | 25.1 | 26.5 |
| AMD: asm fence instead of an out-of-line reduction | | 49.1 |

AMD gfx1201: when the field reduction is inlined at the squaring sites, the compiler gives wrong results at -O1 and
above. Each lane's result then depends on the other lanes' data. An empty `asm volatile` on the 64-bit product
columns before the reduction fixes it at no cost; making the reduction a separate function also works but halves the
speed.

Cost split on the RTX 3060 at 25 MH/s, measured by removing parts (timing only):
- the two scalar multiplications: 55%;
- the HMAC chain: 14%;
- the other SHA-256: ~6%;
- the rest: ~25%.

Nsight Compute: ~144k instructions per nonce, ALU 68% and FMA-heavy 59% busy. The kernel is near its instruction
throughput.

Tried without gain:
- an affine + affine first addition;
- keeping P in shared memory during the second multiplication;
- other block sizes and register budgets;
- PTX carry-chain multiplies (ptxas emits 64 IMAD + 64 IMAD.HI + 115 IADD3.X).

lolMiner runs at ~50 / ~95 MH/s on these cards. The next lever on NVIDIA would be multiplies written as SASS
IMAD.WIDE carry chains (carry predicates in and out), as for Quantus.

Live on Kryptex: RX 9070 XT, 9 accepted / 0 rejected in 25 minutes.

## Cuckaroo29 (Tari, cuckaroo29)

One graph per nonce: 2^29 edges from siphash-2-4 (blocks of 64), trimmed until ~50K edges are left, then a CPU cycle
search for 42-cycles (one per ~42 graphs). The work is memory traffic: each edge is 8 bytes, and the first rounds move
gigabytes.

The trimmer:
- **Buckets:** 4096 per side by the node's top 12 bits, so one block can count a bucket's degrees in shared memory (two
  16 KB bitmaps: seen once, seen twice).
- **Two-step scatter:** a round writes its survivors into 64 coarse buckets of the other side, and a split kernel
  spreads each coarse bucket over its 64 fine ones. Scattering straight into 4096 buckets does not coalesce.
- **Fixed segments:** while edges are many (generation and rounds 0-3), every producing block owns a fixed segment of
  each coarse bucket. That removes generation's counting pass (siphash computed once) and the rounds' reservation
  pass. The rare edge past a segment's end is dropped, which costs a graph nothing measurable.
- **Flat list:** below 256K edges the remaining rounds use one list and global bitmaps.
- **Host:** counts come back every 4th round. The CPU cycle search of one graph runs during the next graph's trimming.

Steps on the RX 9070 XT (200 rounds):

| Step | g/s |
|---|---|
| 64 buckets, global bitmaps (first version) | 4.4 |
| 4096 buckets, direct scatter | 3.1 |
| 64 coarse + split | 4.2 |
| flat list from 256K edges (was 4M) | 5.1 |
| split slices of ~16K edges | 5.4 |
| fixed segments while edges are many | 5.9 |
| counts every 4th round | 6.2 |
| CPU search overlapped | 6.67 |

RTX 3060: 1.38 -> 2.47 g/s. lolMiner runs at ~14 / ~4-5 g/s on these cards.

Per graph on the RX 9070 XT before the overlap (~150 ms of GPU time):

| Part | Time |
|---|---|
| generation + split | 34 ms |
| round 0 | 30 ms |
| rounds 1-3 | 27 ms |
| rounds 4-91 (bucketed, down to 256K edges) | ~43 ms |
| flat rounds 92-199 | ~15 ms |

The first rounds are bandwidth-bound, so the next lever is fewer bytes per edge.

Live on Kryptex: RX 9070 XT, 5 accepted / 0 rejected in 15 minutes at difficulty 20 (6.62 g/s shown).
