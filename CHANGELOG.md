# Changelog

## 0.4.0b (unreleased)

- Beam (BeamHash III) confirmed on a live pool: 2Miners accepted 26 of 26 shares in a 10-minute test from an RTX 3060
  (20.9 Sol/s at 99 W, pool-credited 21.0). The 1% dev fee now applies to Beam, like the other algorithms.

## 0.4.0 (2026-11-16)

- AMD RDNA4 (RX 9000): DAG reads bypass the caches (non-temporal loads), so the random DAG traffic no longer pushes the rest of the
  working set out of the L2 / Infinity Cache. RX 9070 XT at -30% power: Octopus 58.5 -> 67.9 MH/s (+16%), Autolykos2
  64.5 -> 75.5 (+17%), Ethash 32.5 -> 38.8, Etchash 35.5 -> 38.8, KawPow 34.0 -> 38.5, FiroPow 33.1 -> 38.4, MeowPow
  and Evrmore ~+10-20%, FishHash / KarlsenHashV2 23.5 -> 25.9. ProgPoW coins with DAGs under 3.5 GB (Neurai, Epic
  Cash, Zano) keep cached reads, which are ~3% faster for them. Kryptex accepted 5 of 5 Octopus shares in a 10-minute
  test. On NVIDIA only Autolykos2 gains (evict-first reads, RTX 3060 96.3 -> 99.2 MH/s); the others measured
  unchanged or slower there (Octopus 28.6 -> 25.0 at 100 W) and keep cached reads. Older AMD GPUs keep cached reads until
  measured there. `XM_DAG_NT=0` / `1` forces either.
- Octopus on NVIDIA: the matrix tiles run in bands that share their Vandermonde slices in L2, and the mix uses
  256-thread blocks. RTX 3060 31.4 -> 32.6 MH/s at 170 W, 28.2 -> 29.3 at 100 W.
- New algorithm: BeamHash III (Beam, BEAM), `--algorithm beamhash3` (aliases `beam`, `beamv3`). A GPU solver for
  Beam's five-round collision search (2^25 elements, ~5.7 GB of GPU memory), with every solution checked by a CPU
  verifier written from Beam's source; Beam's own stratum, with TLS (2Miners). RX 9070 XT 38.1 Sol/s at 274 W,
  RTX 3060 20.7 Sol/s at 99 W (on NVIDIA each of 65,536 small buckets is processed in shared memory and whole
  elements are written per warp store; both vendors store the last two rounds' elements smaller). New launchers `start-beam-2miners`. The dev fee is off on Beam for now.
  Not yet confirmed on a live pool: please report accepted or rejected shares.
- KawPow family: an implausible block height from a pool is rejected before any DAG work starts.
- New launchers `start-tari-kryptex` for Tari (Cuckaroo29), which the README now lists with the other coins.
- Cuckaroo29 (Tari): RX 9070 XT 8.8 -> 10.75 g/s (+22%), RTX 3060 at 100 W 4.16 -> 4.41 g/s (+6%). Bucketed edges
  take 6 bytes instead of 8 (a round's degree count reads 4), and on AMD every round after the first writes straight
  into the next round's buckets. Kryptex accepted 10 of 10 shares in a 10-minute test.

## 0.3.0c (2026-11-15)

- Pearl on NVIDIA RTX 30 / 40 with a locked memory clock (a common Pearl overclock, e.g. HiveOS "LMEM"): +16%. A
  second GEMM reads 1.5 chunks ahead instead of one, which hides the longer memory latency; X-Miner times both at
  start-up with the clocks as set and mines with the faster one. RTX 3060 at 100 W, memory locked at 5001 MHz:
  31.1 -> 36.1 TH/s (at stock memory the existing kernel stays, 37.9 TH/s). `XM_PEARL_NV_BK=64` / `128` forces one.
- README: on NVIDIA, `--gpu-mclock 810` for Quantus, Parano1d and BLAKE2b (+4-5% at a power limit, or up to 10% less
  power; RTX 3060 numbers); the Watchdog section describes 0.3.0b's behaviour (a stalled GPU is dropped, the others
  keep mining).

## 0.3.0b (2026-11-15)

- Pearl +6% on the RX 9070 XT (121.4 -> 128.7 TH/s at stock, 105.9 -> 112.0 at a 70% power limit) and +5% on the
  RTX 3060 (36.3 -> 38.0 at 100 W, 44.4 -> 46.6 at 170 W): the matrices are now filled with 63 or 64 instead of
  [-8, 7], so every value the GPU multiplies (matrix + noise) is positive. Matrix-unit power depends on how many bits flip between operands, and without sign changes the GPU runs
  a higher clock at the same power. The work and the shares are the same (Kryptex accepted 11 of 11 in a 15-minute
  test), and share proofs compress better. Credit to Iedoc, the developer of BzMiner (https://bzminer.com/), whose
  miner showed this choice of matrix values.
- A GPU that stops responding (a hung kernel, or a card that fell off the bus) no longer stops the whole rig. Before,
  the watchdog exited so the launcher could restart X-Miner, but the process could not finish exiting while a thread
  was stuck in the GPU driver, so every GPU sat idle until a reboot (seen with a GTX 1080 Ti, Xid 79, on HiveOS).
  Now that GPU is dropped after 120 s without progress and the other GPUs keep mining; the miner exits only when no
  GPU is left.

## 0.3.0 (2026-11-15)

- More GPUs (tested on the cards listed below; the other chips are experimental, run `--selftest` first): AMD RDNA1 (RX 5000, BC-160),
  RDNA2 (RX 6000) and RDNA3 (RX 7000); NVIDIA Pascal, Volta and Turing (GTX 10 / 16, RTX 20, Titan V, P10x, CMP
  30HX-50HX). Pearl, Parano1d, Quantus and BLAKE2b now ship code objects for gfx1010-1012, gfx1030-1032, gfx1034 and
  gfx1100-1102 and a compute_61 PTX; the run-time compiled algorithms already covered these GPUs and were compiled
  for them as a check.
- Pearl on GPUs without int8 matrix units: a SIMT kernel with the same tiles and transcript (4-byte dot products:
  `sdot4` on RDNA1/2 except the RX 5600/5700, `sudot4` on RDNA3, plain multiplies on gfx1010; `dp4a` on NVIDIA). Forced
  on our cards it passes the self-test and mines without CPU re-verify failures: RX 9070 XT 24.0 TH/s (15.5 with plain
  multiplies), RTX 3060 at 99 W 9.3 TH/s (the matrix kernels: 121.5 and 36.5).
- Pearl on AMD RDNA3 (RX 7000): a WMMA kernel (RDNA3's 16-byte fragments; A rows are read in an order that keeps the
  RDNA4 hash tiles). We have no RDNA3 GPU, so at start-up the miner compares it with the SIMT kernel on random data and
  mines with the SIMT kernel if they differ (`XM_PEARL_CHECK=0` skips the check). The check itself was tested on an
  RX 9070 XT with a deliberately broken WMMA kernel.
- Pearl on NVIDIA Turing (RTX 20, CMP 40HX / 50HX): an int8 tensor-core kernel (mma m8n8k16) with the same tiles. Forced on
  an RTX 3060 it passes the self-test and mines at 21.5 TH/s (dp4a kernel 9.3). It gets the same start-up check as RDNA3,
  which also times both kernels and keeps the faster one when they agree (GTX 16 GPUs have no tensor cores).
- AMD chips without their own code object (BC-250, Steam Deck, Radeon 680M / 780M / 890M and other APUs): the prebuilt
  kernels also ship as LLVM family-generic code objects (gfx10-1, gfx10-3, gfx11 and gfx12-generic), loaded when no
  exact one exists. Forced on an RX 9070 XT, the gfx12-generic ones pass the self-test and Pearl mines at the same
  121.9 TH/s.
- Parano1d (NOID) and TAPE on NVIDIA with the CLMAD kernel: RTX 3060 at 99 W 18.8 -> 20.9 MH/s (+11%). The
  partial rounds' MDS layer is unrolled with compile-time constants (table bases as load offsets, no per-constant
  branch or lane rotation, each product starting from the round's sum), the full layers likewise, the general
  multiply adds P0 inside the carry-less multiply instead of on the ALU, and the S-box squarings are unrolled.
  Self-test and 19/0 shares live on rplant.
- Quantus on AMD before RDNA4: the 64-bit multiply-add instruction under its RDNA1-3 name.
- NVIDIA: when several PTX versions fit a GPU, the newest one is loaded.
- Tested on HiveOS rigs: RTX 3080 / 3070 / 3060 Ti and GTX 1080 Ti (driver 550, CUDA 12.4), RX 5700 XT, RX 6600 XT,
  RX 6700 XT and RX 6800. Every algorithm passes `--selftest` on every one of them (cuckaroo29 needs more than 8 GB);
  these chips (sm_61, gfx1010, gfx1030-1032) are no longer labelled experimental.
- AMD on HiveOS and other Linux without ROCm's HIP (HiveOS ships only ROCm's OpenCL, so X-Miner found no AMD GPU there):
  the Linux and HiveOS packages include AMD's ROCm 6.4.4 HIP runtime in `hip/` (unmodified, license texts in
  `hip/licenses/`), used only when the system has no HIP, together with the libnuma, libelf and libdrm libraries it
  needs (used when the system lacks them). Checked on bare Ubuntu 22.04 / 24.04, Debian 12 and Fedora 40 installs and
  on HiveOS; needs glibc 2.35+ (on Ubuntu 20.04 the miner says to install ROCm for AMD).
- NVIDIA drivers older than CUDA 13.3 (e.g. 550 / CUDA 12.4): Parano1d no longer crashes the miner inside the driver
  (the CLMAD kernels are only offered to CUDA 13.3+ drivers; the others use the previous kernel, RTX 3080 20.9 MH/s).
- NVIDIA drivers older than CUDA 12.5: the prebuilt PTX (GTX 10 / 16, RTX 20, and newer GPUs without a cubin) was
  rejected (CUDA error 222); it is now marked PTX ISA 7.1, which any driver since CUDA 11.1 compiles.
- Cuckaroo29 on Pascal (GTX 10): no 64 KB shared-memory degree counters (Pascal allows 48 KB; CUDA error 1). GTX 1080 Ti
  3.76 g/s.
- Quantus on RDNA1 (RX 5600 / 5700): every share was wrong with the inline-assembly carry chains; RDNA1 builds plain
  C++ ones instead.
- Pearl on RDNA2 (RX 6000): the padded SIMT kernel is the default, ~24% faster (RX 6800 8.5 -> 10.6 TH/s).
- `--selftest`: each GPU is checked in its own process (a GPU that faults no longer fails the ones after it), `-d`
  selects GPUs (Ctrl+C stops all of them), an algorithm that needs more memory than the GPU has is SKIP rather than
  FAIL, a GPU whose test process crashes is named, and wrong results get a
  hint about unstable overclocks (a +150 MHz core offset made a GTX 1080 Ti fail the DAG algorithms at random).
- `--benchmark`: a GPU whose DAG fits but whose working buffers do not (Conflux's 8.9 GB DAG on a 10 GB RTX 3080) gets
  a clear error instead of a 0 H/s result.
- AMD on Linux: `--gpu-plimit` outside the card's allowed range (RX 6800: 197-247 W) says so instead of "needs root".
- Iron Fish: Kryptex delisted it on 2026-11-10 (its servers close every connection), so the Kryptex launcher is gone
  and FishHash has no dev fee (there is no fee pool). FishHash itself still works with other Iron Fish pools and for
  Karlsen.

## 0.2.0c (2026-11-11)

- Nexa: the affine conversion's inverses come from a new kernel (Montgomery's trick over 32 nonces per thread, no
  block-wide barriers) and the points are stored coalesced. RX 9070 XT 54.7 -> 59.2 MH/s; RTX 3060 at 99 W 28.2 ->
  28.8 MH/s.
- Nexa on AMD: the hashing kernel's SHA-256 rounds in groups of 32 (the compiler copied the state at every group of 16).
  RX 9070 XT 59.0 -> 62.0 MH/s.
- Nexa on AMD: 128 threads per block (was 256). RX 9070 XT 62.0 -> 63.7 MH/s.
- Nexa on AMD: squaring doubles one factor instead of the 64-bit product columns. RX 9070 XT 63.8 -> 64.2 MH/s.
- Pearl on AMD: 128x256 output blocks of 8 waves (was 128x128 of 4), so a quarter fewer global loads and LDS stores
  per MAC; the saved power goes into clock. RX 9070 XT at a 70% power limit 103.2 -> 105.9 TH/s, at stock
  117.2 -> 121.4 TH/s.

## 0.2.0b (2026-11-09)

- New coin: Evrmore (EVR), `--algorithm evrprogpow` (aliases `evr`, `evrmore`): EvrProgPow = KawPoW with 12000-block
  epochs, a 3 GiB epoch-0 DAG and its own Keccak padding (EvrmoreOrg/Evrmore @0a7ba5a). GPU checked against the CPU
  reference; RTX 3060 at 99 W 16.5 MH/s; 24 / 0 shares accepted on zpool. 1% dev fee, mined on zpool with LTC payout. Launchers `start-evrmore-zpool.bat` / `.sh`.

- New coin: Meowcoin (MEWC), `--algorithm meowpow` (aliases `mewc`, `meowcoin`): MeowPow = ProgPoW with 16 mix
  registers, period 6, 6 cache reads and 9 math ops per round, its own Keccak padding, and from epoch 110 a light
  cache and DAG sized for 4x the epoch (Meowcoin-Foundation/Meowcoin @3186f63). Today's DAG is ~10.6 GB, so it needs
  a 12 GB card. GPU checked against the CPU reference; RTX 3060 at 99 W 20.5 MH/s; 21 shares accepted on zpool (1 stale).
  1% dev fee, mined on zpool with LTC payout. Launchers `start-meowcoin-zpool.bat` / `.sh`.

- Neoxa (NEOX) and Satoxcoin (SATOX) run plain KawPoW: `--algorithm neoxa` / `satox` are aliases of `kawpow`.

- Selftest: Nexa no longer fails on the other-signature candidates the search kernel returns on purpose.

- Parano1d (NOID) and TAPE on NVIDIA: a kernel with the GPU's carry-less multiply instruction (CLMAD, new in CUDA
  13.3), for RTX 30 / 40 / 50, working in a 64-bit tower basis of GF(2^128) so a multiply takes 6 full-width CLMADs.
  RTX 3060 at 99 W 9.8 -> 18.8 MH/s. It needs a driver that supports CUDA 13.3; with an older driver the miner uses
  the previous kernel automatically.

- Nexa: the GPU no longer tests whether R.y is a quadratic residue for every nonce (the test only picks k or n - k);
  it checks the powHash of both candidate signatures and the CPU keeps the real one. RTX 3060 26.9 -> 28.5 MH/s
  (+6%) at ~97 W.

- A GPU that is out of memory at start (often a previous miner still exiting, after a quick restart) is retried
  five times, 5 s apart, instead of dropped.

- Cuckaroo29 rates show as graphs per second (`g/s`) in the console and dashboard (were labelled H/s); the API
  summary carries the unit (`x.unit`).

- RETRO BOOMER (TAPE), `--algorithm tape` (alias `retroboomer`): a Parano1d fork with the same proof of work and
  stratum, mined by the Parano1d engine under its own name (its own dev-fee target). rplant
  (`stratum-na.rplant.xyz:7066`), with launchers `start-tape-rplant.bat` / `.sh`. Live on rplant: RX 9070 XT 34
  accepted / 0 rejected in 10 minutes at 21.7 MH/s, RTX 3060 5 / 0 at 10.8 MH/s. 1% dev fee, mined on TAPE.

- Parano1d (NOID) on NVIDIA: in each 32 x 32 carry-less product, two of a bit class's four integer products are summed
  in the multiply's addend instead of XORed (carry-free except when all four class masks are full, probability 2^-32
  per pair; such a nonce is lost at most, candidates are re-checked on the CPU). RTX 3060 9.9 -> 10.8 MH/s.

- Tari Cuckaroo29: edge generation stages edges per bucket in shared memory and writes them out in contiguous runs
  (it wrote one scattered 8-byte store per edge, and spilled its hashes with 1024-thread blocks); the coarse-to-fine
  split stages the same way and runs in one pass (its staged runs reserve their slots when written). RTX 3060
  2.47 -> 3.00 g/s (live on Kryptex 2 accepted / 0 rejected). AMD keeps direct stores (faster there). NVIDIA also
  counts node degrees with 4-bit counters in the big rounds and stages round 0's survivors (3.06 g/s), and the
  staged writers (generation, split, round 0) take several edges per thread between flushes, for longer runs:
  3.36 g/s.
- Tari Cuckaroo29: fewer trimming rounds, adapted while mining (40-200, from 60; the CPU's cycle search, which runs
  alongside the next graph, is kept at about a third to a half of the GPU time), and a cycle search with flat arrays
  instead of a hash map and a vector per node (5-7x faster). RX 9070 XT 6.65 -> 8.36 g/s, RTX 3060 3.36 -> 4.07 g/s.
  The CPU now does real work: about one core per GPU at ~35-50%. Live on Kryptex: 7 accepted / 0 rejected (RTX 3060).
- Tari Cuckaroo29: the bucket rounds below 2^24 edges run 256-thread blocks (several resident per SM; the late rounds
  were waiting on ~150 waves of 1024-thread blocks). RTX 3060 4.08 -> 4.20 g/s; RX 9070 XT unchanged.
- Tari Cuckaroo29: a round's second pass over its bucket reads it back to front, so it starts on edges the cache still
  holds. RX 9070 XT 8.44 -> 8.53 g/s (round 0 18.0 -> 16.5 ms); RTX 3060 unchanged. AMD generation with 512-thread
  blocks like NVIDIA (was 1024): 8.61 g/s. AMD siphash rotations as funnel shifts (v_alignbit; the compiler made a
  64-bit shift, a shift and an OR of each): hashing 9.9 -> 8.0 ms per graph, 8.70 g/s.
- Tari Cuckaroo29 NVIDIA: edge generation (hashing each sipblock twice, ~43 of its ~46 ms) keeps the first 32 hashes
  of the first pass in registers with the hash state after them, so the second pass hashes only the other 32.
  Generation 45.9 -> 39.2 ms, RTX 3060 4.20 -> 4.30 g/s; with 256-thread generating blocks (were 512) 36.1 ms,
  4.37 g/s.
- Tari Cuckaroo29 AMD: the big rounds run one workgroup per WGP, each looping over buckets, so the buckets in flight
  fit the Infinity Cache and the second pass over each hits it. RX 9070 XT 8.62 -> 8.88 g/s (round 0 16.5 ->
  13.7 ms); NVIDIA keeps a block per bucket (no gain on the RTX 3060).

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
