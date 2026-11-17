X-Miner on HiveOS (custom miner)

Flight sheet:
  Coin: PRL (or the coin you mine)   Wallet: your Kryptex krx... username, or your address for the coin's pool
  Pool: Configure in miner
  Miner: Custom
    Miner name:       x-miner-<version>, e.g. x-miner-0.3.0c (HiveOS derives it from the file name and refuses others)
    Installation URL: the x-miner-<version>-hiveos.tar.gz release link
    Hash algorithm:   pearlhash (or noid, tape, quantus, blake2b, kawpow, neurai, evrprogpow, meowpow, firopow,
                      progpow-epic, progpowz, etchash, ethash, quai, autolykos2, octopus, fishhash, karlsenhashv2,
                      xelishashv3, qhash, nexapow, cuckaroo29, beamhash3)
    Wallet and worker template: %WAL%.%WORKER_NAME%
    Pool URL:         prl.kryptex.network:7048   (noid: na.rplant.xyz:17185; tape: stratum-na.rplant.xyz:7066; quantus: qtc.kryptex.network:7049; blake2b: us1.alphapool.tech:5555; others: see the README); TLS: stratum+ssl://host:port
    Pass:             x
    Extra config arguments: optional, e.g. --gpu-id 0,1

AMD cards (RX 5000 / 6000 / 7000 / 9000): the package brings AMD's HIP runtime in hip/ (HiveOS installs only
ROCm's OpenCL), so nothing else is needed. NVIDIA cards (GTX 10 and newer): driver 525+; Parano1d's fastest
kernel needs a CUDA 13.3 driver. Run "./x-miner --selftest" in the miner folder to check every algorithm on
every GPU; wrong results that change from run to run mean the overclock is too high.
The miner's dashboard is on port 21550 of the rig.
