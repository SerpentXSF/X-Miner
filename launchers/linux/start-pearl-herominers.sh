#!/bin/sh
# X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
# Edit WALLET and WORKER below, then run: ./start-pearl-herominers.sh
# Pools: us.pearl.herominers.com:1200
cd "$(dirname "$0")"

WALLET="YOUR_PRL_ADDRESS"
WORKER="$(hostname)"
POOL="us.pearl.herominers.com:1200"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Measured examples:
#   RX 9070 XT, best TH/W: TUNING="--gpu-plimit 70%"
#   RTX 3060, best TH/W:   TUNING="--gpu-cclock 1500"
#   Pearl on NVIDIA: do not lock the memory clock (--gpu-mclock), it costs 8-28%.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm pearlhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
