# X-Miner design

X-Miner is an open-source GPU miner from SerpentX Tech. It continues the X-Miner name of the Blake2b miner
(`GitHub\Bitcoin Knots`), rebuilt as a native multi-algorithm miner.

## Why it exists

Community miners want three things that closed miners don't give them together:

1. **Trust.** The source is open, the dev fee is in plain code, and anyone can audit or build it.
2. **First support for new coins**, especially on AMD, where new coins often have no working miner.
   On 2026-10-15, SRBMiner sat at 0 H/s for Parano1d on an RX 9070 XT.
3. **Best-in-class speed on the coins it does support.** We don't chase a long algorithm list.

Anti-goals: matching SRBMiner's algorithm list for its own sake; any code, kernel or "ABI" taken from
decompiled or decrypted closed miners (see "Clean room"); fee tricks the user can't see.

## Launch scope

| Algorithm | AMD (HIP) | NVIDIA (CUDA) | Notes |
|---|---|---|---|
| `pearlhash` (Pearl, PRL) | RDNA4 first (gfx1201) | Ampere+ (sm_86 first) | Starts from our pearl-rdna4 kernel (~110 TH/s, bit-exact) |
| `noid` (Parano1d, NOID) | RDNA4 first | Ampere+ | No working AMD miner exists today |
| `blake2b` (Bitcoin Knots RDTS) | RDNA4 | Ampere+ | Port of the original X-Miner; 0% fee (lottery) |

Intel GPUs and CPU mining are out of scope for now.

## Shape of the product

- **One download that just runs.** A single native executable per OS (`x-miner.exe`, `x-miner`). No Python,
  no SDK. GPU kernels ship precompiled inside the executable:
  AMD code objects are loaded through the HIP runtime that comes with the Adrenalin / amdgpu driver
  (`amdhip64_6.dll`/`amdhip64_7.dll`, `libamdhip64.so`). NVIDIA cubins and PTX are loaded through the driver API
  (`nvcuda.dll`, `libcuda.so`). Both runtimes are loaded dynamically, so the same binary runs on AMD-only,
  NVIDIA-only and mixed rigs.
- **Ready-made launchers**: `.bat` files for Windows and `.sh` scripts for Linux, one per coin and pool,
  written to be edited by end users, as SRBMiner ships them.
- **Familiar command line.** We follow the flag conventions miners already know (`--algorithm`, `--pool`,
  `--wallet`, `--password`, `--worker`, `--gpu-id`, `--api-port`, failover pools) and an SRBMiner-compatible
  JSON API shape, so HiveOS and monitoring tools work with minimal glue. These are public interfaces learned by
  using the tools and reading their public docs, not their code.
- **Look**: a colored console (24-bit ANSI) and a local web dashboard, both in the SerpentX / Gundam Wing
  cockpit palette (see `docs/mockups/x-miner-ui-mockup.html`). `--no-color` gives plain logs.
- **HiveOS** custom-miner package, as the Blake2b X-Miner has.

## Dev fee

- **1.00%** by default, set in one place in the source (`src/core/devfee.hpp`). Blake2b: 0%.
- Time-sliced: the miner mines to the developer's pool for a short window in each cycle. The console and API
  always show the fee state, the time to the next window and fee-window shares. The fee window is logged.
- No hidden endpoints, no obfuscation, no anti-tamper. Forks can change or remove it; that is the cost of being
  open, and it is acceptable.

## Architecture

```
src/
  main.cpp            entry, signal handling, shutdown
  core/               cli + config, logging/console, stats, dev fee scheduler, job/share types
  net/                TCP (+ TLS later), line-delimited JSON-RPC stratum transport
  pool/               per-coin pool dialects (pearl, noid) on top of the transport
  api/                HTTP API (SRBMiner-like JSON) + embedded web dashboard
  gpu/                dynamic HIP and CUDA loaders, device discovery, sensors (ADL / NVML / sysfs)
  algo/<name>/        CPU reference implementation, host-side job prep, kernel launch glue
kernels/<name>/       HIP (.hip) and CUDA (.cu) sources, compiled to embedded blobs at build time
tests/                known-answer tests against the coin's reference code; pool-dialect fixtures
launchers/            windows/*.bat, linux/*.sh
```

Rules:

- Every algorithm has a **CPU reference path** first. A GPU kernel counts as working only when it matches the
  reference bit-exactly on known-answer vectors, and a coin counts as supported only after the pool accepts its
  shares on real hardware (RX 9070 XT and RTX 3060 are the test cards).
- Every algorithm/vendor/OS cell is "unverified" until it has been run. An unsupported combination gives a clear
  error, never a silent 0 H/s.

## Clean room

Sources we use: the coins' own open-source repositories (Pearl: pearl-research-labs/pearl, ISC), public pool
documentation, open-source miners under compatible licences (credited), our own kernels and measurements, and
the observable behaviour of other miners (command line, API, speed). We do not use, read further or derive from
the decrypted SRBMiner archive (kernels, SASS, host code, fee or anti-tamper analysis).

## Licence

MIT for X-Miner's own code. Third-party code keeps its licence and is listed in `THIRD_PARTY.md`.
