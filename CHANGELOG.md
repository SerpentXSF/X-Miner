# Changelog

## 0.2.0b (unreleased)

- Parano1d (NOID) on NVIDIA: in each 32 x 32 carry-less product, two of a bit class's four integer products are summed
  in the multiply's addend instead of XORed (carry-free except when all four class masks are full, probability 2^-32
  per pair; such a nonce is lost at most, candidates are re-checked on the CPU). RTX 3060 9.9 -> 10.8 MH/s.

- Tari Cuckaroo29: edge generation stages edges per bucket in shared memory and writes them out in contiguous runs
  (it wrote one scattered 8-byte store per edge, and spilled its hashes with 1024-thread blocks); the coarse-to-fine
  split stages the same way and runs in one pass (its staged runs reserve their slots when written). RTX 3060
  2.47 -> 3.00 g/s (live on Kryptex 2 accepted / 0 rejected). AMD keeps direct stores (faster there).

- Nexa: a 24-bit comb window on NVIDIA cards with 10 GB or more (11 point additions per scalar instead of 13; the
  generated RTX 30 kernel is rebuilt for it) and 22 bits elsewhere. The 5.9 / 1.6 GB table is now built on the GPU by
  a prebuilt kernel (kernels/nexatab) in a few seconds, instead of on the CPU (where 24 bits took ~25 s on 24 cores
  and as much host memory per GPU). RTX 3060 30.5 -> 33.3 MH/s, RX 9070 XT 53.5 -> 55.6 MH/s.
- Run-time compilers run on a thread with a 256 MB stack: compiling a large kernel overflowed a mining thread's
  default 1 MB stack on Windows (the process ended with 0xC00000FD).

## 0.2.0 (2026-11-02)

- Octopus (Conflux): the polynomial values as int8 matrix products on the GPU's matrix units (NVIDIA sm_80+ tensor
  cores, AMD RDNA4 WMMA): the job's Vandermonde matrix and each nonce group's coefficients as bytes, nine products
  summed by byte shift and reduced once. The next batch's products run on a second GPU stream beside the DAG mix. RX
  9070 XT 42.7 -> 58.5 MH/s (one fused launch: the next batch's products in persistent blocks beside this batch's
  mix, operands in WMMA fragment order), RTX 3060 24.2 -> 31.4 MH/s, same power. Other GPUs keep the scalar path
  (`XM_OCTOPUS_TC=0` selects it everywhere). Octopus is now memory-bound: `--gpu-mclock 2700` on the RX 9070 XT gives
  63.0 MH/s (+7.3%; launchers and docs/PERFORMANCE.md list the measured sweep).
- Quantus (QTC) on RTX 30: the sm_86 kernel's permutations are now generated, scheduled SASS (tools/qgen, assembled
  with CuAssembler; nonce setup and target check stay compiled CUDA). RTX 3060 151.6 -> 173.9 MH/s (+15%) at the
  same 169 W; self-test passes, live on Kryptex with 0 rejects. Other NVIDIA generations and AMD are unchanged.
- GPU backends: an optional second stream (`set_stream`), created at the highest priority.
- Tari Cuckaroo29 (XTM c29, `cuckaroo29`), new and in progress:
  - CPU reference from the Tari node source; Tari's own test vectors pass, including a whole 42-cycle proof.
  - GPU solver: edge generation and trimming rounds on the GPU, cycle search on the CPU (overlapped with the next
    graph's trimming), edge index recovery on the GPU, every proof verified on the CPU. Trimming keeps the edges in
    4096 buckets whose degree counts fit in shared memory, reached through 64 coarse buckets (writing straight into
    4096 buckets does not coalesce: 80 ms of generation went to scattered writes), with fixed per-block segments while
    edges are many; below 256K edges one flat list. The selftest finds the known 42-cycle of Tari's test graph.
  - Kryptex (`xtm-c29-us.kryptex.network:7040`, XMRig-style stratum; the 8-byte target is little-endian, so
    `cccccccccccccc0c` means difficulty 20): 4 accepted / 0 rejected in 10 minutes from the RX 9070 XT (first
    version), 5 accepted / 0 rejected in 15 minutes with the current trimmer.
  - RX 9070 XT 6.67 g/s, RTX 3060 2.47 g/s (first version 4.4 / 1.4; lolMiner ~14 / ~4-5). Needs ~9 GB of GPU
    memory.
- Nexa (NEXA, `nexapow`), new: SHA256d of the header commitment and nonce, then a BCH Schnorr signature with that hash
  as the private key (RFC6979 nonce, secp256k1), then SHA256 of the signature. The CPU reference reproduces the Echelon
  spec vectors and real main-chain blocks. The GPU kernel (compiled at run time) uses:
  - a signed 20-bit fixed-base comb (13 point additions per scalar, a 436 MB table built on all CPU cores);
  - 29-bit field limbs;
  - one batched inversion per block, by Bernstein-Yang divsteps, overlapped with the other warps' work;
  - divsteps with Jacobi tracking for the quadratic-residue test.

  The selftest compares 512 signatures with the CPU. Kryptex (Miningcore Nexa stratum): 9 accepted / 0 rejected in
  25 minutes from the RX 9070 XT. RX 9070 XT 49.1 MH/s, RTX 3060 25.1 MH/s (lolMiner ~95 / ~50). The AMD compiler
  (gfx1201) miscompiles the inlined field reduction unless an empty asm fence separates it from the products.

  RTX 30 (sm_86): the two comb multiplications are generated, scheduled SASS (tools/ngen, assembled with
  CuAssembler; a prebuilt cubin):
  - 8 x 32-bit field words; a multiply sums each column in a 64-bit accumulator with IMAD.WIDE carry-outs (1.68 times
    the compiled multiply's throughput);
  - the kernel runs as three launches there (hashes / combs / inversion and check), so that the generated one keeps
    nothing else live;
  - the block is checked against Python elliptic-curve arithmetic before it is assembled.

  RTX 3060 25.3 -> 27.7 MH/s, 169 -> 162 W; selftest 512/512; Kryptex 2 accepted / 0 rejected in 11 minutes.
  The first build got ~1% of nonces wrong under full load: a wait on a load / store scoreboard issued one cycle
  after the operation passed before it was counted. The scheduler now keeps such waits 4 cycles back. A
  262144-signature check (`XM_NEXA_SELFTEST_N`) leaves only the accepted zero-digit slips (10, ~2^-19 per window).
  `XM_NEXA_GEN=0` keeps the compiled kernel.

  The inversion / signature kernel (-36% on the RTX 3060):
  - the block's batched inversion is a product tree (~3 field multiplies per nonce instead of ~18);
  - P.X, P.Y, R.X and the scalars are loaded only after the inversion and the quadratic-residue test, so the kernel
    fits 64 registers and runs 4 blocks per SM;
  - the quadratic-residue test runs a fixed number of branch-free divstep iterations per batch in every lane, so
    lanes no longer wait for the slowest one in the warp, and updates only the limbs still in use.

  RTX 3060 27.7 -> 30.5 MH/s (168 W; Kryptex 11 accepted / 0 rejected in 15 minutes). The RX 9070 XT now runs the
  three-kernel form too: 49.1 -> 53.5 MH/s
  (`XM_NEXA_SPLIT=0` keeps the single kernel).

## 0.1.0c (2026-10-30)

- Neurai (XNA, `neurai`): KawPow unchanged on its own chain (checked line by line against NeuraiProject/Neurai), with
  its own algorithm name so the dev fee mines Neurai. CPU reference reproduces real main-chain blocks (mix and final
  hash); live on Kryptex (`xna-us.kryptex.network:7024`) from both cards with 0 rejects. RX 9070 XT 39.4 MH/s, RTX 3060
  20.6 MH/s. `start-neurai-kryptex` launchers.
- EthereumPoW (ETHW, `ethash`): plain Ethash on the Etchash engine (30000-block epochs, no ECIP-1099). CPU reference
  reproduces the mix digest of real EthereumPoW blocks; live on Kryptex (`ethw-us.kryptex.network:7034`) from both
  cards, dev-fee shares included. RX 9070 XT 32.5 MH/s, RTX 3060 41.1 MH/s (8.1 GB DAG). `start-ethw-kryptex` launchers.
- Quai (QUAI, `quai`): KawPow over Ravencoin-format donor headers; pools send Ravencoin-style jobs with Ravencoin
  heights. Live on Kryptex (`quai-kawpow-us.kryptex.network:7043`, which needs a Quai `0x...` address): 20 accepted
  in 20 minutes on the RX 9070 XT (34.0 MH/s), shares from the RTX 3060 (20.6 MH/s), dev-fee shares accepted.
  `start-quai-kryptex` launchers.
- Pool jobs without a height on EthereumStratum pools now map the seed's epoch to its first block (right for ETC and
  ETHW alike).

## 0.1.0b (2026-10-28)

Eleven more coins, all verified against their published test vectors where they exist, by the GPU-against-CPU
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

- QHash (Qubitcoin, `qhash`): SHA-256 of the header, a 16-qubit circuit (two layers of RY / RZ rotations and CNOT
  chains) evaluated exactly as a bond-dimension-2 chain instead of a 65,536-amplitude state vector (about 200 floating
  point operations per hash), its Q15 expectations and SHA-256 again, with the chain's zero-byte rule. Checked against
  real Qubitcoin main-chain block hashes from both angle eras and against a full state-vector simulation; the
  selftest compares 262,144 nonces with the CPU in float and in double. Float by default (the CPU re-checks every
  share; ~0.7% of hashes round differently and are skipped, never sent), `XM_QHASH_FP64=1` for double. RX 9070 XT
  2.51 GH/s, RTX 3060 0.96 GH/s (published 0.70). LuckyPool (Bitcoin stratum).

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
