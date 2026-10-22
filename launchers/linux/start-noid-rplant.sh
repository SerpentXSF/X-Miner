#!/bin/sh
# X-Miner - Parano1d (NOID) on rplant.xyz (1% pool fee)
# Edit WALLET (your o1... NOID address) and WORKER below, then run: ./start-noid-rplant.sh
cd "$(dirname "$0")"

WALLET="YOUR_NOID_ADDRESS"
WORKER="$(hostname)"
POOL="na.rplant.xyz:17185"

while true; do
  ./x-miner --algorithm noid --pool "$POOL" --wallet "$WALLET" --worker "$WORKER"
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
