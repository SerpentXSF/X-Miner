# X-Miner

Open-source GPU miner by SerpentX Tech for **Pearl (pearlhash)**, **Parano1d (noid)**, **Quantus (quantus)**,
**Ravencoin (kawpow)**, **Ethereum Classic (etchash)**, **Ergo (autolykos2)**, **Firo (firopow)** and **Bitcoin Knots
BLAKE2b (blake2b)** on AMD and NVIDIA GPUs, with **Epic Cash (progpow-epic)** and **Zano (progpowz)** kernels verified
and their pool protocols in development. MIT licensed. 1% dev fee (time-sliced: about 60 s per 100 minutes of
mining, shown in the console and API) where a developer wallet is set; none on blake2b and, for now, Firo.

## Quick start

1. Unzip the release.
2. Open the launcher for your coin and pool in a text editor and set `WALLET` (and optionally `WORKER`):
   - `start-pearl-kryptex.bat` / `.sh`: Pearl on Kryptex (use your Kryptex `krx...` username)
   - `start-pearl-herominers.bat` / `.sh`: Pearl on HeroMiners (your PRL address)
   - `start-noid-rplant.bat` / `.sh`: Parano1d on rplant.xyz (your `o1...` NOID address)
   - `start-quantus-kryptex.bat` / `.sh`: Quantus on Kryptex (your Kryptex `krx...` username)
   - `start-kawpow-kryptex.bat` / `.sh`: Ravencoin (KawPow) on Kryptex (your Kryptex `krx...` username)
   - `start-etchash-kryptex.bat` / `.sh`: Ethereum Classic on Kryptex (your Kryptex `krx...` username)
   - `start-ergo-kryptex.bat` / `.sh`: Ergo on Kryptex (your Kryptex `krx...` username)
   - `start-firo-rplant.bat` / `.sh`: Firo on rplant.xyz (your FIRO address; not yet live-tested)
   - `start-blake2b-alphapool.bat` / `.sh`: Bitcoin Knots BLAKE2b (RDTS) on AlphaPool (your RDTS address)
3. Run it. The launcher restarts the miner if it ever exits.

The web dashboard is at <http://127.0.0.1:21550/> while the miner runs (`--api-host 0.0.0.0` for LAN access).

**HiveOS** (custom miner, **not yet tested on a HiveOS rig**; reports welcome): put the release's
`x-miner-<version>-hiveos.tar.gz` link in the flight sheet's Installation URL; `README.txt` inside the package has
the flight sheet fields.

Command line (all options: `x-miner --help`):

```
x-miner --algorithm pearlhash --pool prl.kryptex.network:7048 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm noid --pool na.rplant.xyz:17185 --wallet o1... --worker rig1
x-miner --algorithm quantus --pool qtc-us.kryptex.network:7049 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm kawpow --pool rvn-us.kryptex.network:7031 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm etchash --pool etc-us.kryptex.network:7033 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm autolykos2 --pool erg-us.kryptex.network:7021 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm firopow --pool na.rplant.xyz:7045 --wallet FIRO_ADDRESS --worker rig1
x-miner --algorithm blake2b --pool us1.alphapool.tech:5555 --wallet ADDRESS --worker rig1
```

KawPow and the other ProgPoW coins compile each new random program on the GPU's own compiler at run time, and
Etchash and Autolykos2 compile their kernels the same way once per start: the packages ship NVIDIA's NVRTC (and, on
Windows, AMD's hipRTC) next to `x-miner`; on Linux, AMD GPUs use the hipRTC that ROCm installs. Ergo needs about 8 GB
of GPU memory (its 7.6 GB table is rebuilt on the GPU at every block, 5-9 s). The first start of an epoch builds its light cache on the CPU and the DAG (6.2 GB for Ravencoin in late
2026) on the GPU, which takes 10-30 s.

## GPU tuning

Clocks, offsets, power limits and intensity can be set per GPU on the command line (one value for every GPU or a
comma list in `--gpu-id` order, `-` to skip a GPU). Everything is put back to what it was when X-Miner exits: on Ctrl+C,
when its window is closed, when a benchmark ends, when the watchdog restarts it, and on Linux on SIGTERM / SIGHUP
(HiveOS `miner stop`). A forced kill (Task Manager "End task", `kill -9`), a system crash or a power cut gives the miner
no chance to do that: reset the card then (AMD: Adrenalin > Performance > Tuning > Reset; NVIDIA: reboot, or
`nvidia-smi -rgc`, `-rmc` and `-pl <watts>` as administrator). Tested on an RX 9070 XT (Windows) and an RTX 3060
(Windows); the AMD Linux path (amdgpu sysfs) has not been tested on hardware yet.

| Option | NVIDIA (Windows / Linux) | AMD Windows | AMD Linux |
|---|---|---|---|
| `--gpu-plimit 150` / `--gpu-plimit 70%` | watts or % of default | % only (Adrenalin's power slider) | watts or % |
| `--gpu-cclock 1500` | locks the core clock | - | Overdrive max core |
| `--gpu-mclock 7000` | locks the memory clock (GeForce: the nearest of the card's fixed memory clocks, listed by `--gpu-tune-info`; use `--gpu-moffset` to overclock memory) | max memory clock | Overdrive max memory |
| `--gpu-coffset 100` | core clock offset | max clock offset | - |
| `--gpu-moffset 500` | memory clock offset | - | - |
| `--gpu-voffset -50` | - (lock the core clock instead, see below) | core voltage offset in mV (Adrenalin "Voltage Offset"; tested on RDNA4) | core voltage offset in mV (RDNA3/4, untested) |
| `--gpu-fan 70` | fixed fan speed in % | flat fan curve at N% | flat fan curve at N% |
| `--gpu-intensity 64` | work per launch (pearlhash: N x 1024 columns, auto 64; noid / quantus / blake2b: 2^N nonces; 0 = auto) | same | same |

NVIDIA has no voltage control in its management library (NVML). To undervolt an NVIDIA card, lock the core clock
(`--gpu-cclock 1500`): the GPU then runs at the voltage of that point of its voltage/frequency curve, which is how
HiveOS undervolts NVIDIA cards too.

Example (Windows `.bat`, two GPUs: an RTX 3060 and an RX 9070 XT; `-` leaves a GPU untouched):

```
x-miner.exe --algorithm quantus --pool qtc-us.kryptex.network:7049 --wallet krxXXXXXXX --worker rig1 ^
  --gpu-plimit 170,70% --gpu-mclock 810,- --gpu-voffset -,-50 --gpu-fan 70,-
```

Measured settings (60 s benchmarks, 2026-10-27; every card and sample differs, so check stability on yours):

| Card | Pearl | Quantus | Parano1d | BLAKE2b |
|---|---|---|---|---|
| RX 9070 XT at `--gpu-plimit 70%` | `--gpu-voffset -50`: no gain while mining | `--gpu-voffset -50`: +4.5% | `--gpu-voffset -50`: +3.8% | `--gpu-voffset -50`: +3.5% |
| RTX 3060 (170 W) | do not lock memory: `--gpu-mclock` 7301 costs 8%, 5001 costs 28%; `--gpu-cclock 1500` for the best TH/W | `--gpu-mclock 810`: +1% | `--gpu-mclock 810`: same speed, 15 W less | `--gpu-mclock 810`: same speed, 15 W less |

`--gpu-tune-info` lists what each GPU allows and its current values. NVIDIA tuning and AMD Linux tuning need
administrator / root rights (HiveOS runs miners as root).

## Watchdog

If a GPU returns 10 errors in a row, makes no progress for 2 minutes, or no GPU can be started, X-Miner exits with a
non-zero code; the launchers restart it.

## Supported hardware

| Vendor | Pearl | Parano1d | Quantus | KawPow, Etchash, Autolykos2, FiroPow | BLAKE2b | Requirement |
|---|---|---|---|---|---|---|
| AMD RDNA4 (RX 9070 / 9070 XT, gfx1201) | yes | yes | yes | yes | yes | Windows: Adrenalin driver (provides the HIP runtime); Linux: ROCm 6.x |
| AMD RDNA4 (RX 9060 / 9060 XT, gfx1200) | built, untested | built, untested | built, untested | untested | built, untested | same |
| NVIDIA RTX 30xx (sm_86) | yes | yes | yes | yes | yes | driver 555+ |
| NVIDIA RTX 40xx (sm_89) | native build, untested | native build, untested | native build, untested | untested | native build, untested | driver 555+ |
| NVIDIA RTX 50xx (sm_120) | native build, untested | native build, untested | native build, untested | untested | native build, untested | driver 570+ |
| NVIDIA A100, H100 and other newer | via PTX JIT (untested) | via PTX JIT (untested) | via PTX JIT (untested) | untested | via PTX JIT (untested) | driver 555+ |
| Other AMD, NVIDIA Turing and older | no | no | no | no | no | |

`x-miner --list-devices` shows each GPU and which algorithms it can run.

## Measured hashrates

| Card | Pearl | Parano1d | Quantus | KawPow (6.2 GB DAG) | Etchash (4.4 GB) | Autolykos2 (7.6 GB) | BLAKE2b |
|---|---|---|---|---|---|---|---|
| RX 9070 XT (-30% power, ~237 W) | 101.1 TH/s | 20.1 MH/s | 179.6 MH/s | 34.0 MH/s | 35.5 MH/s | 64.5 MH/s | 4.63 GH/s |
| RX 9070 XT (stock power, ~300-340 W) | 115.0 TH/s | | 195.8 MH/s | | | | |
| RTX 3060 (stock, 170 W) | 44.4 TH/s | 9.8 MH/s | 149.7 MH/s | 20.6 MH/s | 41.0 MH/s | 96.3 MH/s | 1.97 GH/s |

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
- KawPow (Ravencoin): Kryptex (`rvn-us.kryptex.network:7031`, `rvn.kryptex.network:7031`) and other pools that speak
  the kawpowminer stratum (`mining.subscribe` / `mining.notify` with header hash, seed hash, target and height).
- FiroPow (Firo): pools speaking the same kawpowminer stratum, e.g. rplant (`na.rplant.xyz:7045`); not yet live-tested.
- Etchash (Ethereum Classic): EthereumStratum/1.0.0 pools, e.g. Kryptex (`etc-us.kryptex.network:7033`).
- Autolykos2 (Ergo): Ergo stratum pools, e.g. Kryptex (`erg-us.kryptex.network:7021`).
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
