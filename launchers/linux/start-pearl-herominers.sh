#!/bin/sh
# X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
# Edit WALLET and WORKER below, then run: ./start-pearl-herominers.sh
# Pools: us.pearl.herominers.com:1200
cd "$(dirname "$0")"

WALLET="YOUR_PRL_ADDRESS"
WORKER="$(hostname)"
POOL="us.pearl.herominers.com:1200"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Example:
# TUNING="--gpu-plimit 70% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70"
TUNING=""

while true; do
  ./x-miner --algorithm pearlhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
