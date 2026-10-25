# X-Miner

Open-source GPU miner by SerpentX Tech for **Pearl (pearlhash)**, **Parano1d (noid)**, **Quantus (quantus)** and
**Bitcoin Knots BLAKE2b (blake2b)** on AMD and NVIDIA GPUs. MIT licensed. 1% dev fee on Pearl, Parano1d and
Quantus (time-sliced: about 60 s per 100 minutes of mining, shown in the console and API); no dev fee on blake2b.

## Quick start

1. Unzip the release.
2. Open the launcher for your coin and pool in a text editor and set `WALLET` (and optionally `WORKER`):
   - `start-pearl-kryptex.bat` / `.sh`: Pearl on Kryptex (use your Kryptex `krx...` username)
   - `start-pearl-herominers.bat` / `.sh`: Pearl on HeroMiners (your PRL address)
   - `start-noid-rplant.bat` / `.sh`: Parano1d on rplant.xyz (your `o1...` NOID address)
   - `start-quantus-kryptex.bat` / `.sh`: Quantus on Kryptex (your Kryptex `krx...` username)
   - `start-blake2b-alphapool.bat` / `.sh`: Bitcoin Knots BLAKE2b (RDTS) on AlphaPool (your RDTS address)
3. Run it. The launcher restarts the miner if it ever exits.

The web dashboard is at <http://127.0.0.1:21550/> while the miner runs (`--api-host 0.0.0.0` for LAN access).

Command line (all options: `x-miner --help`):

```
x-miner --algorithm pearlhash --pool prl.kryptex.network:7048 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm noid --pool na.rplant.xyz:17185 --wallet o1... --worker rig1
x-miner --algorithm quantus --pool qtc-us.kryptex.network:7049 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm blake2b --pool us1.alphapool.tech:5555 --wallet ADDRESS --worker rig1
```

## GPU tuning

Clocks, offsets, power limits and intensity can be set per GPU on the command line (one value for every GPU or a
comma list in `--gpu-id` order, `-` to skip a GPU). Everything is restored when X-Miner exits.

| Option | NVIDIA (Windows / Linux) | AMD Windows | AMD Linux |
|---|---|---|---|
| `--gpu-plimit 150` / `--gpu-plimit 70%` | watts or % of default | % only (Adrenalin's power slider) | watts or % |
| `--gpu-cclock 1500` | locks the core clock | - | Overdrive max core |
| `--gpu-mclock 7000` | locks the memory clock | max memory clock | Overdrive max memory |
| `--gpu-coffset 100` | core clock offset | max clock offset | - |
| `--gpu-moffset 500` | memory clock offset | - | - |
| `--gpu-intensity 32` | work per launch (pearlhash: N x 1024 columns, noid / quantus / blake2b: 2^N nonces; 0 = auto) | same | same |

`--gpu-tune-info` lists what each GPU allows and its current values. NVIDIA tuning and AMD Linux tuning need
administrator / root rights (HiveOS runs miners as root).

## Watchdog

If a GPU returns 10 errors in a row, makes no progress for 2 minutes, or no GPU can be started, X-Miner exits with a
non-zero code; the launchers restart it.

## Supported hardware

| Vendor | Pearl | Parano1d | Quantus | BLAKE2b | Requirement |
|---|---|---|---|---|---|
| AMD RDNA4 (RX 9070 / 9070 XT, gfx1201) | yes | yes | yes | yes | Windows: Adrenalin driver (provides the HIP runtime); Linux: ROCm 6.x |
| NVIDIA RTX 30xx / 40xx (sm_86 / sm_89) | yes | yes | yes | yes | driver 555+ |
| NVIDIA A100, RTX 50xx and newer | via PTX JIT (untested) | via PTX JIT (untested) | via PTX JIT (untested) | via PTX JIT (untested) | driver 555+ |
| Other AMD, NVIDIA Turing and older | no | no | no | no | |

`x-miner --list-devices` shows each GPU and which algorithms it can run.

## Measured hashrates

| Card | Pearl | Parano1d | Quantus | BLAKE2b |
|---|---|---|---|---|
| RX 9070 XT (-30% power, ~237 W) | 101.1 TH/s | 20.1 MH/s | | 4.63 GH/s |
| RX 9070 XT (stock power, ~300-340 W) | 115.0 TH/s | | 182.9 MH/s | |
| RTX 3060 (stock, 170 W) | 44.4 TH/s | 9.8 MH/s | 148.9 MH/s | 1.97 GH/s |

The console and dashboard also show a **pool-credited** hashrate: the work of your accepted shares (share
difficulty x 2^32 for Pearl) per hour and per run. That is what the pool should credit, so it is the number to compare
with the pool's dashboard once a few dozen shares are in.

Every share is re-checked on the CPU before it is sent. Pearl shares were also checked against the official Pearl
verifier. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Pools

TLS pools work with `stratum+ssl://host:port` (Windows: SChannel; Linux: OpenSSL 1.1 or 3, loaded at run time).
Pool certificates are not verified, as with other miners (pools use self-signed certificates).

- Pearl: Kryptex (`prl.kryptex.network:7048`, `prl-us.kryptex.network:7048`), HeroMiners, and other pools that use
  the object-style Pearl stratum. Kryptex gets gzip-compressed share proofs (negotiated automatically).
- Parano1d: rplant (`na.rplant.xyz:17185`, plain or TLS on the same port), InnovLab
  (`stratum+ssl://us.innovlab.cc:19601`) and other `parano1d-stratum-v1` pools.
- Quantus: Kryptex (`qtc-us.kryptex.network:7049`, `qtc.kryptex.network:7049`; XMRig-style stratum).
- BLAKE2b (Bitcoin Knots RDTS chain): the DATUM gateway's Sia-style stratum, solo through your own DATUM gateway
  or on AlphaPool (`us1.alphapool.tech:5555`). Pool difficulties are sized for Sia-class ASICs, so GPU mining here
  is a lottery.

## Antivirus warnings

Windows Defender and other antivirus tools flag most cryptocurrency miners, X-Miner included, as "potentially
unwanted" because miners are often installed without the owner's consent. X-Miner is open source: build it yourself
if you prefer, check the release's `.sha256` file, and add an exclusion for the X-Miner folder if your antivirus
quarantines `x-miner.exe`.

## Building from source

Windows (VS 2022): `.\build.ps1`. Linux (gcc 11+, CMake, Ninja): `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
GPU kernels are prebuilt in `kernels/prebuilt/`; rebuild them with `tools/build_kernels.ps1` (ROCm HIP SDK and CUDA 12).
Release packages: `tools/package.ps1` (Windows) and `tools/package.sh` (Linux).

See [docs/DESIGN.md](docs/DESIGN.md), [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
