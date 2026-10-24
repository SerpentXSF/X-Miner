#!/bin/sh
# X-Miner - Bitcoin Knots BLAKE2b (RDTS) on AlphaPool. No X-Miner dev fee on blake2b.
# Edit WALLET (your RDTS payout address) and WORKER below, then run: ./start-blake2b-alphapool.sh
# AlphaPool's share difficulty is sized for ASICs: a GPU finds a share every few hours or days (lottery mining).
cd "$(dirname "$0")"

WALLET="YOUR_RDTS_ADDRESS"
WORKER="$(hostname)"
POOL="us1.alphapool.tech:5555"

while true; do
  ./x-miner --algorithm blake2b --pool "$POOL" --wallet "$WALLET" --worker "$WORKER"
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
