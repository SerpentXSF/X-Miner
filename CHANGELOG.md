# Changelog

## 0.1.0b (2026-10-28)

Ten more coins, all verified against their published test vectors where they exist, by the GPU-against-CPU
`--selftest`, and with shares accepted on live pools from an RX 9070 XT and an RTX 3060.

New algorithms
- KawPow (Ravencoin, `kawpow`): ProgPoW programs generated per period and compiled at run time (NVRTC / hipRTC,
  shipped in the packages), DAG built on the GPU. RX 9070 XT 34.0 MH/s @ 237 W, RTX 3060 20.6 MH/s (WildRig 34.1 /
  T-Rex 20.85 on the same cards). Kryptex.
- FiroPow (Firo, `firopow`), ProgPoW 0.9.2 (Epic Cash, `progpow-epic`) and ProgPowZ (Zano, `progpowz`) on the same
  engine: 9070 XT 33.1 / 39.9 / 40.0 MH/s, RTX 3060 20.5 / 20.6 / 20.6 MH/s. WoolyPooly (Firo), epic.pool-pay.com (Epic,
  Epic node stratum), HeroMiners (Zano, ethproxy stratum).
- Etchash (Ethereum Classic, `etchash`): ECIP-1099 epochs, 8 lanes per hash. RTX 3060 41.0, RX 9070 XT 35.5 MH/s (the
  card's random-access limit). Kryptex.
- Autolykos v2 (Ergo, `autolykos2`): the 7.6 GB table rebuilt on the GPU per block; the selftest compares 4096
  consecutive nonces with the CPU. RTX 3060 96.3 MH/s @ 137 W, RX 9070 XT 64.5 MH/s. Kryptex.
- Octopus (Conflux, `octopus`): per-nonce polynomial evaluation (Shoup multiply; full-rate 24-bit multiplies in inline
  asm on AMD) and the DAG mix. RX 9070 XT 42.7 MH/s, RTX 3060 24.2 MH/s; other miners reach ~79 / ~42 with a
  tensor-core method that X-Miner does not have yet. Kryptex.
- FishHash (Iron Fish, `fishhash`) and KarlsenHashV2 (Karlsen, `karlsenhashv2`): one fixed 4.5 GB dataset, BLAKE3 and
  32 rounds of three 128-byte reads. RX 9070 XT 23.5 MH/s (25.2 with `--gpu-mclock 2700`), RTX 3060 27.5 MH/s @ 140 W.
  Kryptex (Iron Fish, native Iron Fish stratum), WoolyPooly (Karlsen, Kaspa-style stratum).

- XelisHashV3 (Xelis, `xelishashv3`): one GPU thread per hash with its own 544 KB scratchpad (ChaCha8 fill, the
  sequential random walk with exact 128-bit division and the reference's double-precision square root, BLAKE3 tree),
  run as fill / walk-slices / final kernels so no launch runs long. Matches the published xelis-hash vectors; the
  selftest compares 256 nonces with the CPU. RX 9070 XT 5.0 kH/s, RTX 3060 4.49 kH/s. Kryptex (Xelis stratum).

Pools
- Epic Cash: shares are sent at most 50 per second (the pool answers about 60 per second per connection and drops the
  rest) and only for the current block height; the worker is joined with `+`.
- All pool clients keep recent jobs in arrival order (a sorted set could drop the current job when job ids are
  random, turning good shares into "job expired"), and a malformed pool message is logged and ignored instead of
  ending the miner.

Other
- Native RTX 50 (sm_120, CUDA 12.8) kernels for Pearl, Parano1d, Quantus and BLAKE2b, RX 9060 (gfx1200) and native RTX
  40 (sm_89) builds (untested on hardware).
- Quantus NVIDIA: carry corrections without SEL (+1% on RTX 3060).
- A DAG or table that fails to rebuild is no longer taken as current on the next job; implausible block heights are
  rejected instead of building a huge cache.
- Dev fee (1%) on every new coin; `XM_RTC_DUMP=<dir>` saves the run-time compiled kernels and `XM_RTC_OPTS` adds
  compiler options (developers).
- Linux packages preload the shipped NVRTC builtins library, so the run-time compiled algorithms work on NVIDIA
  from the package.

## 0.1.0 (first release)

Algorithms
- Pearl (pearlhash): int8 tensor-core / WMMA GEMM with the Pearl transcript fused in and the share check on the GPU.
  Every share is re-checked on the CPU before it is sent. Gzip-compressed share proofs for pools that negotiate them
  (Kryptex).
- Parano1d (noid): GPU search with CPU re-verification.
- Quantus (quantus, QTC): Poseidon2 / Goldilocks QPoW search with CPU re-verification, Kryptex stratum.
  Checked against the official Quantus test vectors.
- Bitcoin Knots BLAKE2b (blake2b, RDTS chain): GPU search with CPU re-verification, DATUM / AlphaPool stratum.
  No dev fee. Checked against the BLAKE2b activation block mined on Knots regtest.

GPUs
- AMD RDNA4 (RX 9070 / 9070 XT) on Windows (Adrenalin driver) and Linux (ROCm 6.x).
- NVIDIA RTX 30xx / 40xx (sm_86 / sm_89); newer GPUs through PTX JIT (untested).

Measured (see docs/PERFORMANCE.md)
- RX 9070 XT: Pearl 115.0 TH/s at stock power (~340 W), 101.1 TH/s at -30% (~237 W); Parano1d 20.1 MH/s;
  BLAKE2b 4.63 GH/s at -30%; Quantus 195.8 MH/s at stock power (~330 W).
- RTX 3060: Pearl 44.4 TH/s at stock 170 W; Parano1d 9.8 MH/s; Quantus 148.9 MH/s; BLAKE2b 1.97 GH/s.

Features
- Pools: Pearl object stratum (Kryptex, HeroMiners), parano1d-stratum-v1 (rplant, InnovLab), Quantus
  (Kryptex), DATUM BLAKE2b (DATUM gateway, AlphaPool); TLS; failover lists.
- GPU tuning: power limit, core / memory clock locks, clock offsets, core voltage offset (AMD RDNA3/4), fan speed,
  intensity, per GPU; restored on exit (Ctrl+C, closing the window, end of a benchmark, watchdog restart, SIGTERM /
  SIGHUP on Linux). Every launcher has an optional TUNING line.
- Watchdog: exits for the launcher to restart after repeated GPU errors, a stalled GPU, or no usable GPU.
- HTTP API and web dashboard, pool-credited hashrate, HiveOS custom-miner package (not yet tested on a HiveOS rig).
- No runtime dependencies beyond the GPU driver: static Windows exe, Linux binary for glibc 2.29+ (Ubuntu 20.04+).

Dev fee: 1% on Pearl, Parano1d and Quantus, time-sliced (60 s per 100 minutes of mining) and shown in the console
and API; none on BLAKE2b.

GPU control: --gpu-fan (fixed fan speed), --gpu-off-temperature / --gpu-resume-temperature (pause a hot GPU).
