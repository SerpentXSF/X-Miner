X-Miner on HiveOS (custom miner)

Flight sheet:
  Coin: PRL (or NOID, QTC)     Wallet: your Kryptex krx... username / PRL address / o1... NOID address
  Pool: Configure in miner
  Miner: Custom
    Installation URL: the x-miner-<version>-hiveos.tar.gz release link
    Hash algorithm:   pearlhash   (or noid, quantus, blake2b)
    Wallet and worker template: %WAL%.%WORKER_NAME%
    Pool URL:         prl.kryptex.network:7048   (noid: na.rplant.xyz:17185; quantus: qtc.kryptex.network:7049; blake2b: us1.alphapool.tech:5555); TLS: stratum+ssl://host:port
    Pass:             x
    Extra config arguments: optional, e.g. --gpu-id 0,1

AMD cards need ROCm 6.x (libamdhip64.so, looked up in /opt/rocm/lib as well); NVIDIA cards need driver 555+.
The miner's dashboard is on port 21550 of the rig.
