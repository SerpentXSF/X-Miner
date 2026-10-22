# Performance notes

Measured numbers and what we learned while tuning. Hashrates are at the power limits shown.

## Pearl (pearlhash)

| Card | Power limit | X-Miner | Reference | Notes |
|---|---|---|---|---|
| RX 9070 XT (gfx1201) | -30% (~238 W) | 93.6 TH/s | SRBMiner 3.7.6: ~100 TH/s @ 244 W | 16384x32768x4096 per pass; Kryptex live: 19/0 shares in 40 min |
| RTX 3060 (sm_86) | 127 W | 35.9 TH/s @ ~1600 MHz | SRBMiner 3.7.1: 48.8 TH/s @ 127 W / 1760 MHz | |

AMD pass breakdown (RX 9070 XT, 16384x16384x4096): noise for A 0.29 ms, GEMM + fused jackpot check 11.6 ms.
The GEMM core alone is at parity with WildRig at the same power (pearl-rdna4 kernel).

NVIDIA findings (RTX 3060):
- With the transcript fold removed (timing only), the same GEMM runs at ~54-57 TH/s and higher clocks, so the
  fold, i.e. reading all int32 accumulators every 128 k, is where the time goes.
- Things that did NOT help: non-volatile mma, 3-stage pipeline, row-interleaved XOR, 8 warps with 32x64 warp
  tiles (same speed as 4 warps of 64x64), out-of-place accumulation at chunk boundaries (removes the WAR hazard
  between the fold and the next chunk's mma).
- Small wins: unconditional fold after a 2-stage unroll (+3%), reduce-scatter instead of per-tile butterflies,
  precomputed swizzled ldmatrix/cp.async addresses.
- Every variant sits at 35-36 TH/s and ~1600 MHz at the power cap, while SRBMiner holds 1760 MHz at the same
  power. Next to try: larger block tiles (128x256 / 256x128) with dynamic shared memory (needs
  cuFuncSetAttribute support in the CUDA backend) to cut L2 and shared-memory traffic per MAC.

## Parano1d (noid)

| Card | X-Miner | Reference |
|---|---|---|
| RX 9070 XT | 11.1 MH/s @ 236 W (first version) | SRBMiner: 0 H/s; closed AMD miner: 20.9 MH/s on RX 9070 |
| RTX 3060 | see below | closed miners: ~29.5 MH/s @ 90 W |

First version uses the generic Karatsuba multiply for the MDS constants. NoidMiner's shared-memory nibble tables and
tower-basis partial rounds are the next steps.
