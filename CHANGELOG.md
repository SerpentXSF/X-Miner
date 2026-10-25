# Changelog

## 0.1.0 (first release)

Algorithms
- Pearl (pearlhash): int8 tensor-core / WMMA GEMM with the Pearl transcript fused in and the share check on the GPU.
  Every share is re-checked on the CPU before it is sent. Gzip-compressed share proofs for pools that negotiate them
  (Kryptex).
- Parano1d (noid): GPU search with CPU re-verification.
- Quantus (quantus, QTC): Poseidon2 / Goldilocks QPoW search with CPU re-verification, Kryptex stratum.
  No dev fee for now. Checked against the official Quantus test vectors.
- Bitcoin Knots BLAKE2b (blake2b, RDTS chain): GPU search with CPU re-verification, DATUM / AlphaPool stratum.
  No dev fee. Checked against the BLAKE2b activation block mined on Knots regtest.

GPUs
- AMD RDNA4 (RX 9070 / 9070 XT) on Windows (Adrenalin driver) and Linux (ROCm 6.x).
- NVIDIA RTX 30xx / 40xx (sm_86 / sm_89); newer GPUs through PTX JIT (untested).

Measured (see docs/PERFORMANCE.md)
- RX 9070 XT: Pearl 115.0 TH/s at stock power (~340 W), 101.1 TH/s at -30% (~237 W); Parano1d 17.5 MH/s;
  BLAKE2b 4.63 GH/s at -30%; Quantus 182.9 MH/s at stock power (~297 W).
- RTX 3060: Pearl 44.4 TH/s at stock 170 W; Parano1d 8.7 MH/s; Quantus 148.9 MH/s; BLAKE2b 1.97 GH/s.

Features
- Pools: Pearl object stratum (Kryptex, HeroMiners), parano1d-stratum-v1 (rplant, InnovLab), Quantus
  (Kryptex), DATUM BLAKE2b (DATUM gateway, AlphaPool); TLS; failover lists.
- GPU tuning: power limit, core / memory clock locks, clock offsets, intensity, per GPU; restored on exit.
- Watchdog: exits for the launcher to restart after repeated GPU errors, a stalled GPU, or no usable GPU.
- HTTP API and web dashboard, pool-credited hashrate, HiveOS custom-miner package.
- No runtime dependencies beyond the GPU driver: static Windows exe, Linux binary for glibc 2.29+ (Ubuntu 20.04+).

Dev fee: 1% on Pearl and Parano1d, time-sliced (60 s per 100 minutes of mining) and shown in the console and API;
none on Quantus and BLAKE2b.

GPU control: --gpu-fan (fixed fan speed), --gpu-off-temperature / --gpu-resume-temperature (pause a hot GPU).
