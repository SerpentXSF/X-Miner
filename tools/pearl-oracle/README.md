# pearl-oracle

Development tool: checks shares produced by X-Miner with Pearl's official verifier (`zk-pow`, ISC), built from
source. It is not shipped with the miner.

    git clone https://github.com/pearl-research-labs/pearl ../pearl     # sibling of the X-Miner folder
    cd tools/pearl-oracle && cargo build --release
    x-miner --pearl-share-test out/         # writes header.hex, nbits.hex, share.b64
    target/release/pearl-oracle verify $(cat out/header.hex) $(cat out/nbits.hex) out/share.b64
