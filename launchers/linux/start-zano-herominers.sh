#!/bin/sh
# X-Miner - Zano (ZANO, ProgPowZ) on HeroMiners (your Zano address)
# Edit WALLET and WORKER below, then run: ./start-zano-herominers.sh
# Pools: us.zano.herominers.com:1110!de.zano.herominers.com:1110
cd "$(dirname "$0")"

WALLET="YOUR_ZANO_ADDRESS"
WORKER="$(hostname)"
POOL="us.zano.herominers.com:1110!de.zano.herominers.com:1110"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   RX 9070 XT: 40 MH/s at the -30% power limit; ProgPowZ is compute and memory heavy.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm progpowz --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
