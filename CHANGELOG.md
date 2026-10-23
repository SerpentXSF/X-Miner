# Changelog

## 0.1.0 (first release)

Algorithms
- Pearl (pearlhash): int8 tensor-core / WMMA GEMM with the Pearl transcript fused in and the share check on the GPU.
  Every share is re-checked on the CPU before it is sent. Gzip-compressed share proofs for pools that negotiate them
  (Kryptex).
- Parano1d (noid): GPU search with CPU re-verification.

GPUs
- AMD RDNA4 (RX 9070 / 9070 XT) on Windows (Adrenalin driver) and Linux (ROCm 6.x).
- NVIDIA RTX 30xx / 40xx (sm_86 / sm_89); newer GPUs through PTX JIT (untested).

Measured (see docs/PERFORMANCE.md)
- RX 9070 XT: Pearl 115.0 TH/s at stock power (~340 W), 101.1 TH/s at -30% (~237 W); Parano1d 17.5 MH/s.
- RTX 3060: Pearl 41.5 TH/s at 170 W (36.7 at 127 W); Parano1d 8.7 MH/s.

Features
- Pools: Pearl object stratum (Kryptex, HeroMiners), parano1d-stratum-v1 (rplant, InnovLab); TLS; failover lists.
- GPU tuning: power limit, core / memory clock locks, clock offsets, intensity, per GPU; restored on exit.
- Watchdog: exits for the launcher to restart after repeated GPU errors, a stalled GPU, or no usable GPU.
- HTTP API and web dashboard, pool-credited hashrate, HiveOS custom-miner package.
- No runtime dependencies beyond the GPU driver: static Windows exe, Linux binary for glibc 2.29+ (Ubuntu 20.04+).

Dev fee: 1%, time-sliced and shown in the console and API. It is off in builds without developer wallets.
