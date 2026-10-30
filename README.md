# X-Miner

Open-source GPU miner by SerpentX Tech for AMD and NVIDIA GPUs. MIT licensed. 1% dev fee (time-sliced: about 60 s
per 100 minutes of mining, shown in the console and API); none on BLAKE2b.

| Coin | `--algorithm` | Coin | `--algorithm` |
|---|---|---|---|
| Pearl (PRL) | `pearlhash` | Conflux (CFX) | `octopus` |
| Parano1d (NOID) | `noid` | Iron Fish (IRON) | `fishhash` |
| Quantus (QTC) | `quantus` | Karlsen (KLS) | `karlsenhashv2` |
| Ravencoin (RVN) | `kawpow` | Firo (FIRO) | `firopow` |
| Ethereum Classic (ETC) | `etchash` | Epic Cash (EPIC) | `progpow-epic` |
| Ergo (ERG) | `autolykos2` | Zano (ZANO) | `progpowz` |
| Bitcoin Knots RDTS | `blake2b` | Xelis (XEL) | `xelishashv3` |
| Qubitcoin (QTC) | `qhash` | Neurai (XNA) | `neurai` |
| EthereumPoW (ETHW) | `ethash` | Quai (QUAI) | `quai` |

Every algorithm is checked against its coin's published test vectors where they exist (ProgPowZ and Octopus have
none: GPU against the CPU reference), by a GPU-against-CPU `--selftest`, and with shares accepted on a live pool from
both an RX 9070 XT and an RTX 3060.

## Quick start

1. Unzip the release.
2. Open the launcher for your coin and pool in a text editor and set `WALLET` (and optionally `WORKER`):
   - `start-pearl-kryptex.bat` / `.sh`: Pearl on Kryptex (use your Kryptex `krx...` username)
   - `start-pearl-herominers.bat` / `.sh`: Pearl on HeroMiners (your PRL address)
   - `start-noid-rplant.bat` / `.sh`: Parano1d on rplant.xyz (your `o1...` NOID address)
   - `start-quantus-kryptex.bat` / `.sh`: Quantus on Kryptex (your Kryptex `krx...` username)
   - `start-kawpow-kryptex.bat` / `.sh`: Ravencoin on Kryptex (your Kryptex `krx...` username)
   - `start-neurai-kryptex.bat` / `.sh`: Neurai on Kryptex (your Kryptex `krx...` username)
   - `start-etchash-kryptex.bat` / `.sh`: Ethereum Classic on Kryptex (your Kryptex `krx...` username)
   - `start-ethw-kryptex.bat` / `.sh`: EthereumPoW on Kryptex (your Kryptex `krx...` username)
   - `start-quai-kryptex.bat` / `.sh`: Quai on Kryptex (your Quai `0x...` address; Kryptex has no username login for Quai)
   - `start-ergo-kryptex.bat` / `.sh`: Ergo on Kryptex (your Kryptex `krx...` username)
   - `start-octopus-kryptex.bat` / `.sh`: Conflux on Kryptex (your Kryptex `krx...` username)
   - `start-fishhash-kryptex.bat` / `.sh`: Iron Fish on Kryptex (your Kryptex `krx...` username)
   - `start-karlsen-woolypooly.bat` / `.sh`: Karlsen on WoolyPooly (your `karlsen:...` address)
   - `start-firo-woolypooly.bat` / `.sh`: Firo on WoolyPooly (your FIRO address)
   - `start-epic-poolpay.bat` / `.sh`: Epic Cash on epic.pool-pay.com (your epicbox address)
   - `start-zano-herominers.bat` / `.sh`: Zano on HeroMiners (your Zano address)
   - `start-xelis-kryptex.bat` / `.sh`: Xelis on Kryptex (your Kryptex `krx...` username)
   - `start-qubitcoin-luckypool.bat` / `.sh`: Qubitcoin on LuckyPool (your Qubitcoin address)
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
x-miner --algorithm neurai --pool xna-us.kryptex.network:7024 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm etchash --pool etc-us.kryptex.network:7033 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm ethash --pool ethw-us.kryptex.network:7034 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm quai --pool quai-kawpow-us.kryptex.network:7043 --wallet 0xYOUR_QUAI_ADDRESS --worker rig1
x-miner --algorithm autolykos2 --pool erg-us.kryptex.network:7021 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm octopus --pool cfx-us.kryptex.network:7027 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm fishhash --pool iron-us.kryptex.network:7017 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm karlsenhashv2 --pool pool.woolypooly.com:3132 --wallet karlsen:qq... --worker rig1
x-miner --algorithm firopow --pool pool.woolypooly.com:3104 --wallet FIRO_ADDRESS --worker rig1
x-miner --algorithm progpow-epic --pool epic.pool-pay.com:3333 --wallet ADDRESS@epicbox.epiccash.com --worker rig1
x-miner --algorithm progpowz --pool us.zano.herominers.com:1110 --wallet ZANO_ADDRESS --worker rig1
x-miner --algorithm xelishashv3 --pool xel-us.kryptex.network:7019 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm qhash --pool qtc-us-east.lproute.com:8610 --wallet QTC_ADDRESS --worker rig1
x-miner --algorithm blake2b --pool us1.alphapool.tech:5555 --wallet ADDRESS --worker rig1
```

The ProgPoW coins (Ravencoin, Firo, Epic Cash, Zano) compile each new random program on the GPU's own compiler at run
time, and Etchash, Autolykos2, Octopus, FishHash, XelisHashV3 and QHash compile their kernels the same way once per start: the packages ship
NVIDIA's NVRTC (and, on Windows, AMD's hipRTC) next to `x-miner`; on Linux, AMD GPUs use the hipRTC that ROCm installs.
The first job of an epoch builds its light cache on the CPU and the DAG on the GPU, which takes 5-40 s. GPU memory
needed (late 2026): Conflux about 10 GB (8.8 GB DAG), Ergo about 8 GB (a 7.6 GB table, rebuilt at every block in 5-9
s), Firo about 7 GB, Ravencoin about 6.5 GB, Iron Fish and Karlsen about 5 GB (one fixed 4.5 GB dataset), Ethereum
Classic about 5 GB, Epic Cash and Zano about 2.5 GB. XelisHashV3 gives every hash in flight its own 544 KB scratchpad
and sizes the batch to the GPU (RX 9070 XT 4,096 hashes = 2.2 GB; RTX 3060 ~17,700 = 9.6 GB).

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

| Vendor | Pearl, Parano1d, Quantus, BLAKE2b (prebuilt kernels) | Run-time compiled algorithms (all others) | Requirement |
|---|---|---|---|
| AMD RDNA4 (RX 9070 / 9070 XT, gfx1201) | yes | yes | Windows: Adrenalin driver (provides the HIP runtime); Linux: ROCm 6.x |
| AMD RDNA4 (RX 9060 / 9060 XT, gfx1200) | built, untested | untested | same |
| NVIDIA RTX 30xx (sm_86) | yes | yes | driver 555+ |
| NVIDIA RTX 40xx (sm_89) | native build, untested | untested | driver 555+ |
| NVIDIA RTX 50xx (sm_120) | native build, untested | untested | driver 570+ |
| NVIDIA A100, H100 and other newer | via PTX JIT (untested) | untested | driver 555+ |
| Other AMD, NVIDIA Turing and older | no | no | |

`x-miner --list-devices` shows each GPU and which algorithms it can run.

## Measured hashrates

RX 9070 XT at a -30% power limit (~237 W), RTX 3060 at its stock 170 W, X-Miner's own benchmarks (2026-10-28):

| Algorithm | RX 9070 XT | RTX 3060 | Notes |
|---|---|---|---|
| Pearl | 101.1 TH/s (115.0 at stock power) | 44.4 TH/s | |
| Parano1d | 20.1 MH/s | 9.8 MH/s | |
| Quantus | 179.6 MH/s (195.8 at stock power) | 149.7 MH/s | |
| KawPow (Ravencoin) | 34.0 MH/s | 20.6 MH/s | WildRig 34.1 / T-Rex 20.85 on the same cards |
| KawPow (Neurai) | 39.4 MH/s | 20.6 MH/s | smaller DAG (3.1 GB) than Ravencoin's |
| FiroPow (Firo) | 33.1 MH/s | 20.5 MH/s | |
| ProgPoW 0.9.2 (Epic Cash) | 39.9 MH/s | 20.6 MH/s | see "Epic Cash" below |
| ProgPowZ (Zano) | 40.0 MH/s | 20.6 MH/s | WildRig: 41.1 MH/s on a 9070 XT at 314 W |
| Etchash (Ethereum Classic) | 35.5 MH/s | 41.0 MH/s | |
| Ethash (EthereumPoW) | 32.5 MH/s | 41.1 MH/s | 8.1 GB DAG |
| Autolykos2 (Ergo) | 64.5 MH/s | 96.3 MH/s @ 137 W | |
| Octopus (Conflux) | 42.7 MH/s | 24.2 MH/s | other miners: ~79 / ~42 (tensor-core method, not yet in X-Miner) |
| FishHash (Iron Fish) / KarlsenHashV2 | 23.5 MH/s (25.2 with `--gpu-mclock 2700`) | 27.5 MH/s @ 140 W | |
| XelisHashV3 (Xelis) | 5.0 kH/s @ ~170 W | 4.49 kH/s @ 112 W | published: RTX 3060 5.5 kH/s, RTX 4090 15 kH/s |
| QHash (Qubitcoin) | 2.51 GH/s | 0.96 GH/s @ 147 W | published: RTX 3060 0.70 GH/s, RTX 3070 1.14 GH/s |
| BLAKE2b | 4.63 GH/s | 1.97 GH/s | |

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
- KawPow (Ravencoin): Kryptex (`rvn-us.kryptex.network:7031`) and other pools that speak the kawpowminer stratum
  (`mining.subscribe` / `mining.notify` with header hash, seed hash, target and height).
- Neurai: KawPow unchanged, the same stratum, e.g. Kryptex (`xna-us.kryptex.network:7024`, 2-byte extranonce) or
  2Miners (`xna.2miners.com:6060`). It has its own `--algorithm neurai` so the dev fee mines Neurai, not Ravencoin.
- FiroPow (Firo): the same kawpowminer stratum, e.g. WoolyPooly (`pool.woolypooly.com:3104`). rplant
  (`na.rplant.xyz:7045`) rejected every share from every miner we tried, WildRig included, on 2026-10-28.
- Etchash (Ethereum Classic): EthereumStratum/1.0.0 pools, e.g. Kryptex (`etc-us.kryptex.network:7033`).
- Ethash (EthereumPoW): the same stratum, e.g. Kryptex (`ethw-us.kryptex.network:7034`).
- Quai: KawPow over Ravencoin-format donor headers, mined with the Ravencoin stratum and heights, e.g. Kryptex
  (`quai-kawpow-us.kryptex.network:7043`; log in with a Quai `0x...` address, not a `krx...` username).
- Autolykos2 (Ergo): Ergo stratum pools, e.g. Kryptex (`erg-us.kryptex.network:7021`).
- Octopus (Conflux): Conflux stratum pools, e.g. Kryptex (`cfx-us.kryptex.network:7027`).
- FishHash (Iron Fish): the native Iron Fish stratum, e.g. Kryptex (`iron-us.kryptex.network:7017`).
- KarlsenHashV2 (Karlsen): Kaspa-style stratum (karlsen-stratum-bridge), e.g. WoolyPooly (`pool.woolypooly.com:3132`).
- Epic Cash: the Epic node's stratum, e.g. epic.pool-pay.com (`epic.pool-pay.com:3333`). The wallet is your epicbox
  address with or without `@epicbox.epiccash.com`; X-Miner joins the worker with `+` (Epic pools read `.` as a
  deposit note). The pool credits every share at a fixed difficulty of 200,000 and answers about 60 shares per
  second per connection, so X-Miner sends at most 50 per second: the pool credits up to ~10 MH/s per connection
  whatever the GPU does.
- Zano: the ethproxy stratum (`eth_submitLogin` / `eth_getWork` with the block height), e.g. HeroMiners
  (`us.zano.herominers.com:1110`, `de.`, ...). The worker is sent separately (HeroMiners reads `address.x` as a
  payment id).
- XelisHashV3 (Xelis): the Xelis stratum (xel/v3), e.g. Kryptex (`xel-us.kryptex.network:7019`, share difficulty
  fixed at 1,048,576).
- QHash (Qubitcoin): Bitcoin stratum (coinbase + merkle branches), e.g. LuckyPool (`qtc-us-east.lproute.com:8610`,
  `qtc-eu.`, `qtc-hk.`, ...; starting difficulty 48 = 206 G hashes per share). Any Qubitcoin address type works;
  a native SegWit `bc1q...` address is the cheapest to spend from later.
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
