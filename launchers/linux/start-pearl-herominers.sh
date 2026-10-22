#!/bin/sh
# X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
# Edit WALLET and WORKER below, then run: ./start-pearl-herominers.sh
# Pools: us.pearl.herominers.com:1200
cd "$(dirname "$0")"

WALLET="YOUR_PRL_ADDRESS"
WORKER="$(hostname)"
POOL="us.pearl.herominers.com:1200"

while true; do
  ./x-miner --algorithm pearlhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER"
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
