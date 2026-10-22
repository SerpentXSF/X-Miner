# Performance notes

Measured numbers and what we learned while tuning. Hashrates are at the power limits shown.

## Pearl (pearlhash)

| Card | Power limit | X-Miner | Reference | Notes |
|---|---|---|---|---|
| RX 9070 XT (gfx1201) | -30% (~238 W) | 93.6 TH/s | SRBMiner 3.7.6: ~100 TH/s @ 244 W | 16384x32768x4096 per pass; Kryptex live: 19/0 shares in 40 min |
| RTX 3060 (sm_86) | 127 W | 37.4 TH/s @ ~1500 MHz | SRBMiner 3.7.1: 48.8 TH/s @ 127 W / 1760 MHz | per-lane hash tiles |

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
- Per-lane hash tiles (2026-10-22): Pearl allows periodic, non-contiguous tiles, so the NVIDIA configuration uses
  exactly the accumulators one lane owns (64x64 warp tile: rows {0,8,..,56} x cols {0,1,8,9,..,56,57}, h*w = 128).
  The fold becomes an in-register XOR with no shuffles: 35.9 -> 37.0 TH/s (32x64 warp tiles), 37.4 (64x64, 4 warps).
  The official reference miner does the same (rows [0,8], 64 columns). The "no fold" variant above likely let the
  compiler delete the MMAs (55 registers), so the fold was never the 40% it seemed.
- At the 127.5 W cap the card runs 1500 MHz (throttle reason: software power cap) and the kernel reaches ~86% of the
  int8 tensor peak per clock; the gap to SRBMiner is energy per MAC, not issue rate.
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
