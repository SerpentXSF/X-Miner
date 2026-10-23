# X-Miner

Open-source GPU miner by SerpentX Tech for **Pearl (pearlhash)** and **Parano1d (noid)** on AMD and NVIDIA GPUs.
MIT licensed. 1% dev fee (time-sliced, shown in the console and API; off in builds without developer wallets).

## Quick start

1. Unzip the release.
2. Open the launcher for your coin and pool in a text editor and set `WALLET` (and optionally `WORKER`):
   - `start-pearl-kryptex.bat` / `.sh`: Pearl on Kryptex (use your Kryptex `krx...` username)
   - `start-pearl-herominers.bat` / `.sh`: Pearl on HeroMiners (your PRL address)
   - `start-noid-rplant.bat` / `.sh`: Parano1d on rplant.xyz (your `o1...` NOID address)
3. Run it. The launcher restarts the miner if it ever exits.

The web dashboard is at <http://127.0.0.1:21550/> while the miner runs (`--api-host 0.0.0.0` for LAN access).

Command line (all options: `x-miner --help`):

```
x-miner --algorithm pearlhash --pool prl.kryptex.network:7048 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm noid --pool na.rplant.xyz:17185 --wallet o1... --worker rig1
```

## Supported hardware

| Vendor | Pearl | Parano1d | Requirement |
|---|---|---|---|
| AMD RDNA4 (RX 9070 / 9070 XT, gfx1201) | yes | yes | Windows: Adrenalin driver (provides the HIP runtime); Linux: ROCm 6.x |
| NVIDIA RTX 30xx / 40xx (sm_86 / sm_89) | yes | yes | driver 555+ |
| NVIDIA A100, RTX 50xx and newer | via PTX JIT (untested) | via PTX JIT (untested) | driver 555+ |
| Other AMD, NVIDIA Turing and older | no | no | |

`x-miner --list-devices` shows each GPU and which algorithms it can run.

## Measured hashrates

| Card | Pearl | Parano1d |
|---|---|---|
| RX 9070 XT (-30% power, ~237 W) | 100.8 TH/s | 17.5 MH/s |
| RTX 3060 (127 W / 170 W), Linux | 42.3 / 46.3 TH/s | 9.6 MH/s |
| RTX 3060 (170 W), Windows | 41.3 TH/s | 8.7 MH/s |

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

## Building from source

Windows (VS 2022): `.\build.ps1`. Linux (gcc 11+, CMake, Ninja): `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build`.
GPU kernels are prebuilt in `kernels/prebuilt/`; rebuild them with `tools/build_kernels.ps1` (ROCm HIP SDK and CUDA 12).
Release packages: `tools/package.ps1` (Windows) and `tools/package.sh` (Linux).

See [docs/DESIGN.md](docs/DESIGN.md), [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
