# X-Miner

Open-source GPU miner by SerpentX Tech for AMD and NVIDIA GPUs. MIT licensed. 1% dev fee (time-sliced: about 60 s
per 100 minutes of mining, shown in the console and API); none on BLAKE2b and FishHash (Iron Fish).

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
| Nexa (NEXA) | `nexapow` | RETRO BOOMER (TAPE) | `tape` |
| Evrmore (EVR) | `evrprogpow` | Meowcoin (MEWC) | `meowpow` |

Every algorithm is checked against its coin's published test vectors where they exist (ProgPowZ and Octopus have
none: GPU against the CPU reference), by a GPU-against-CPU `--selftest`, and with shares accepted on a live pool from
both an RX 9070 XT and an RTX 3060.

## Quick start

1. Unzip the release.
2. Open the launcher for your coin and pool in a text editor and set `WALLET` (and optionally `WORKER`):
   - `start-pearl-kryptex.bat` / `.sh`: Pearl on Kryptex (use your Kryptex `krx...` username)
   - `start-pearl-herominers.bat` / `.sh`: Pearl on HeroMiners (your PRL address)
   - `start-noid-rplant.bat` / `.sh`: Parano1d on rplant.xyz (your `o1...` NOID address)
   - `start-tape-rplant.bat` / `.sh`: RETRO BOOMER (TAPE) on rplant.xyz (your `tape1...` address)
   - `start-quantus-kryptex.bat` / `.sh`: Quantus on Kryptex (your Kryptex `krx...` username)
   - `start-kawpow-kryptex.bat` / `.sh`: Ravencoin on Kryptex (your Kryptex `krx...` username)
   - `start-neurai-kryptex.bat` / `.sh`: Neurai on Kryptex (your Kryptex `krx...` username)
   - `start-etchash-kryptex.bat` / `.sh`: Ethereum Classic on Kryptex (your Kryptex `krx...` username)
   - `start-ethw-kryptex.bat` / `.sh`: EthereumPoW on Kryptex (your Kryptex `krx...` username)
   - `start-evrmore-zpool.bat` / `.sh`: Evrmore on zpool (any payout address; set `PAYOUT` to its coin, e.g. BTC or LTC)
   - `start-meowcoin-zpool.bat` / `.sh`: Meowcoin on zpool (same; needs a 12 GB card)
   - `start-quai-kryptex.bat` / `.sh`: Quai on Kryptex (your Quai `0x...` address; Kryptex has no username login for Quai)
   - `start-ergo-kryptex.bat` / `.sh`: Ergo on Kryptex (your Kryptex `krx...` username)
   - `start-nexa-kryptex.bat` / `.sh`: Nexa on Kryptex (your Kryptex `krx...` username)
   - `start-octopus-kryptex.bat` / `.sh`: Conflux on Kryptex (your Kryptex `krx...` username)
   - `start-karlsen-woolypooly.bat` / `.sh`: Karlsen on WoolyPooly (your `karlsen:...` address)
   - `start-firo-woolypooly.bat` / `.sh`: Firo on WoolyPooly (your FIRO address)
   - `start-epic-poolpay.bat` / `.sh`: Epic Cash on epic.pool-pay.com (your epicbox address)
   - `start-zano-herominers.bat` / `.sh`: Zano on HeroMiners (your Zano address)
   - `start-xelis-kryptex.bat` / `.sh`: Xelis on Kryptex (your Kryptex `krx...` username)
   - `start-qubitcoin-luckypool.bat` / `.sh`: Qubitcoin on LuckyPool (your Qubitcoin address)
   - `start-blake2b-alphapool.bat` / `.sh`: Bitcoin Knots BLAKE2b (RDTS) on AlphaPool (your RDTS address)
3. Run it. The launcher restarts the miner if it ever exits.

The web dashboard is at <http://127.0.0.1:21550/> while the miner runs (`--api-host 0.0.0.0` for LAN access).

**HiveOS** (custom miner; tested on NVIDIA and AMD HiveOS rigs, AMD with the HIP runtime the package brings): put the release's
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
x-miner --algorithm fishhash --pool <iron-fish-pool>:<port> --wallet <IRON address> --worker rig1
x-miner --algorithm karlsenhashv2 --pool pool.woolypooly.com:3132 --wallet karlsen:qq... --worker rig1
x-miner --algorithm firopow --pool pool.woolypooly.com:3104 --wallet FIRO_ADDRESS --worker rig1
x-miner --algorithm progpow-epic --pool epic.pool-pay.com:3333 --wallet ADDRESS@epicbox.epiccash.com --worker rig1
x-miner --algorithm progpowz --pool us.zano.herominers.com:1110 --wallet ZANO_ADDRESS --worker rig1
x-miner --algorithm xelishashv3 --pool xel-us.kryptex.network:7019 --wallet krxXXXXXXX --worker rig1
x-miner --algorithm evrprogpow --pool evrprogpow.na.mine.zpool.ca:1330 --wallet BTC_ADDRESS --password c=BTC --worker rig1
x-miner --algorithm meowpow --pool meowpow.na.mine.zpool.ca:1327 --wallet BTC_ADDRESS --password c=BTC --worker rig1
x-miner --algorithm qhash --pool qtc-us-east.lproute.com:8610 --wallet QTC_ADDRESS --worker rig1
x-miner --algorithm blake2b --pool us1.alphapool.tech:5555 --wallet ADDRESS --worker rig1
```

The ProgPoW coins (Ravencoin, Firo, Epic Cash, Zano) compile each new random program on the GPU's own compiler at run
time, and Etchash, Autolykos2, Octopus, FishHash, XelisHashV3 and QHash compile their kernels the same way once per start: the packages ship
NVIDIA's NVRTC and AMD's hipRTC next to `x-miner` (on Linux in `hip/`, with AMD's HIP runtime; a system ROCm 6.x is
used instead when present).
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
| AMD RDNA4 (RX 9070 / 9070 XT, gfx1201) | yes | yes | Windows: Adrenalin driver (provides the HIP runtime); Linux / HiveOS: the HIP runtime ships in the package (`hip/`; Ubuntu 22.04-based systems with libnuma, libelf and libdrm-amdgpu, as on HiveOS), or a system ROCm 6.x |
| AMD RDNA4 (RX 9060 / 9060 XT, gfx1200) | built, untested | untested | same |
| NVIDIA RTX 30xx (sm_86) | yes (RTX 3060 / 3060 Ti / 3070 / 3080) | yes | driver 525+ (CUDA 12.0; tested with 550 and with a CUDA 13.3 driver); Parano1d's fastest kernel needs CUDA 13.3 |
| NVIDIA RTX 40xx (sm_89) | native build, untested | untested | driver 525+ |
| NVIDIA RTX 50xx (sm_120) | native build, untested | untested | driver 570+ |
| NVIDIA A100, H100 and other newer | via PTX JIT (untested) | untested | driver 525+ |
| AMD RDNA3 (RX 7000, gfx1100 / 1101 / 1102) | experimental, untested (Pearl: WMMA kernel, checked against the SIMT kernel at start-up) | experimental, untested | same as RDNA4 |
| AMD RDNA2 (RX 6000, gfx1030 / 1031 / 1032 / 1034) | yes: tested on RX 6800 / 6700 XT / 6600 XT (Pearl: padded SIMT kernel); gfx1034 untested | yes (same cards) | same as RDNA4 |
| AMD RDNA1 (RX 5000, BC-160, gfx1010 / 1011 / 1012) | yes: tested on RX 5700 XT (Pearl: SIMT kernel, slow without dot instructions); gfx1011 / 1012 untested | yes (RX 5700 XT) | same as RDNA4 (works with the bundled HIP runtime although ROCm does not list RDNA1) |
| NVIDIA Pascal (GTX 10, P102 / P104 / P106) | yes: tested on GTX 1080 Ti (PTX JIT; Pearl: dp4a SIMT kernel) | yes (GTX 1080 Ti) | a driver that still supports the GPU (NVIDIA's 580 branch is the last for Pascal); CUDA 11.1+ |
| NVIDIA Volta, Turing (GTX 16, RTX 20, Titan V, CMP 30HX-50HX) | experimental, untested (PTX JIT; Pearl: SIMT kernel, on Turing an int8 tensor-core kernel checked against it at start-up) | experimental, untested | same as Pascal (580 is also the last for Volta) |
| AMD RDNA1-3 chips without their own build: BC-250 (gfx1013), Steam Deck (gfx1033), Radeon 660M / 680M (gfx1035 / 1036), 780M (gfx1103), 880M / 890M / 8060S (gfx1150-1153) | experimental, untested (family-generic code objects; their loading was checked on an RX 9070 XT) | experimental, untested | same as RDNA4; on Linux a recent ROCm (generic code objects are code object v6) |
| AMD GCN (RX 400 / 500, Vega, Radeon VII), NVIDIA Maxwell and older | no | no | |

Overclocks: the CPU check catches GPUs that compute wrong results. A GTX 1080 Ti with a +150 MHz core offset (fine for
ethash-era miners) failed the DAG algorithms at random; at +100 or stock it passes. If `--selftest` reports wrong
results that change from run to run, test at stock clocks.

"Experimental" GPUs run kernels that were compiled for them and checked on our own cards (the Pearl SIMT kernels by
forcing them on an RX 9070 XT and an RTX 3060), but not on the GPUs themselves. Run `x-miner --selftest` first: it
checks every algorithm against the CPU reference on your GPU. Please report the result (and your hashrates) in an issue.

`x-miner --list-devices` shows each GPU and which algorithms it can run.

## Measured hashrates

RX 9070 XT at a -30% power limit (~237 W), RTX 3060 at its stock 170 W, X-Miner's own benchmarks (2026-10-28):

| Algorithm | RX 9070 XT | RTX 3060 | Notes |
|---|---|---|---|
| Pearl | 105.9 TH/s (121.4 at stock power) | 44.4 TH/s | |
| Parano1d | 20.1 MH/s | 20.9 MH/s at 99 W | RTX 30-50: carry-less multiply (CLMAD) kernel, needs a driver for CUDA 13.3+; with an older driver 10.8 MH/s |
| Quantus | 179.6 MH/s (195.8 at stock power) | 149.7 MH/s | |
| KawPow (Ravencoin) | 34.0 MH/s | 20.6 MH/s | WildRig 34.1 / T-Rex 20.85 on the same cards |
| KawPow (Neurai) | 39.4 MH/s | 20.6 MH/s | smaller DAG (3.1 GB) than Ravencoin's |
| FiroPow (Firo) | 33.1 MH/s | 20.5 MH/s | |
| ProgPoW 0.9.2 (Epic Cash) | 39.9 MH/s | 20.6 MH/s | see "Epic Cash" below |
| ProgPowZ (Zano) | 40.0 MH/s | 20.6 MH/s | WildRig: 41.1 MH/s on a 9070 XT at 314 W |
| Etchash (Ethereum Classic) | 35.5 MH/s | 41.0 MH/s | |
| Ethash (EthereumPoW) | 32.5 MH/s | 41.1 MH/s | 8.1 GB DAG |
| Autolykos2 (Ergo) | 64.5 MH/s | 96.3 MH/s @ 137 W | |
| Octopus (Conflux) | 58.5 MH/s (63.0 with `--gpu-mclock 2700`) | 31.4 MH/s | int8 matrix units (RDNA4 WMMA / NVIDIA sm_80+); other miners: ~79 / ~42 |
| FishHash (Iron Fish) / KarlsenHashV2 | 23.5 MH/s (25.2 with `--gpu-mclock 2700`) | 27.5 MH/s @ 140 W | |
| XelisHashV3 (Xelis) | 5.0 kH/s @ ~170 W | 4.49 kH/s @ 112 W | published: RTX 3060 5.5 kH/s, RTX 4090 15 kH/s |
| QHash (Qubitcoin) | 2.51 GH/s | 0.96 GH/s @ 147 W | published: RTX 3060 0.70 GH/s, RTX 3070 1.14 GH/s |
| NexaPow (Nexa) | 64.3 MH/s @ 339 W | 33.3 MH/s @ 168 W | lolMiner: ~95 / ~50; a 1.6 / 5.9 GB table is built on the GPU at start |
| BLAKE2b | 4.63 GH/s | 1.97 GH/s | |

On other GPUs at stock clocks and power (HiveOS rigs; NVIDIA driver 550 / CUDA 12.4, so Parano1d uses its
non-CLMAD kernel; 20 s benchmarks, every share CPU-checked, 2026-11-15). "-": the DAG, dataset or graph does not fit
in the GPU's memory (MeowPow 10.1 GB, Conflux 8.9 GB, Cuckaroo29 8.7 GB).

| Algorithm | RTX 3080 | RTX 3070 | RTX 3060 Ti | GTX 1080 Ti | RX 6800 | RX 6700 XT | RX 6600 XT | RX 5700 XT |
|---|---|---|---|---|---|---|---|---|
| Pearl (TH/s) | 101.0 | 71.9 | 66.3 | 16.6 | 11.8 | 7.7 | 6.2 | 1.9 |
| Parano1d (MH/s) | 24.9 | 17.0 | 15.1 | 5.8 | 9.4 | 6.2 | 5.0 | 6.2 |
| Quantus (MH/s) | 371 | 265 | 238 | 77 | 100 | 66 | 53 | 57 |
| BLAKE2b (GH/s) | 4.47 | 3.09 | 2.74 | 2.85 | 2.37 | 1.54 | 1.26 | 1.54 |
| KawPow / FiroPow / EvrProgPow (MH/s) | 43.4 | 25.5 | 25.6 | 22.1 | 28.2-28.8 | 18.3-18.6 | 14.9-15.2 | 18.5-18.7 |
| MeowPow (MH/s) | - | - | - | 22.2 | 30.1 | 22.2 | - | - |
| ProgPoW 0.9.2 / ProgPowZ (MH/s) | 43.4-43.7 | 25.6-25.7 | 25.7 | 22.3-22.4 | 25.8-26.3 | 16.7-17.0 | 13.7-14.0 | 16.7-17.0 |
| Etchash (MH/s) | 83.7 | 50.6 | 50.1 | 31.8 | 57.4 | 42.7 | 28.2 | 51.9 |
| Autolykos2 (MH/s) | 169.7 | 132.8 | 116.6 | 54.1 | 109.7 | 81.5 | 54.4 | 101.0 |
| Octopus (MH/s) | - | - | - | 14.4 | 22.3 | 14.5 | - | - |
| FishHash / KarlsenHashV2 (MH/s) | 56.3 | 33.9 | 33.7 | 21.5 | 38.5 | 28.7 | 18.9 | 34.9 |
| XelisHashV3 (kH/s) | 5.29 | 3.83 | 3.82 | 3.06 | 4.20 | 3.93 | 2.97 | 4.19 |
| QHash (GH/s) | 2.13 | 1.46 | 1.30 | 1.20 | 1.23 | 0.81 | 0.66 | 0.81 |
| NexaPow (MH/s) | 58.5 | 40.4 | 35.6 | 7.5 | 28.6 | 18.9 | 15.4 | 18.0 |
| Cuckaroo29 (g/s) | 8.83 | - | - | 3.77 | 4.01 | 2.82 | - | - |

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
  (`stratum+ssl://us.innovlab.cc:19601`) and other `parano1d-stratum-v1` pools. Some pools and trackers call the
  proof of work "poseidon2b"; it is `--algorithm noid` (or `tape`) here.
- RETRO BOOMER (TAPE): a Parano1d fork with the same proof of work and stratum. rplant (`stratum-na.rplant.xyz:7066`,
  TLS on 17066; also `stratum-eu` / `stratum-asia`). Its own `--algorithm tape` so the dev fee mines TAPE.
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
- FishHash (Iron Fish): the native Iron Fish stratum. Kryptex delisted Iron Fish on 2026-11-10 (its servers now close
  every connection), so use another Iron Fish pool.
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
- NexaPow (Nexa): the Miningcore Nexa stratum (4-byte extranonce, 8-byte miner nonce), e.g. Kryptex
  (`nexa-us.kryptex.network:7026`, `nexa.kryptex.network:7026`; share difficulty 1 = 4.3 G hashes per share).
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
