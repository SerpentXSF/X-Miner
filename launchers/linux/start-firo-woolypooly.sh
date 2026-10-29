#!/bin/sh
# X-Miner - Firo (FIRO, FiroPow) on WoolyPooly (your FIRO address)
# Edit WALLET and WORKER below, then run: ./start-firo-woolypooly.sh
# Pools: pool.woolypooly.com:3104
cd "$(dirname "$0")"

WALLET="YOUR_FIRO_ADDRESS"
WORKER="$(hostname)"
POOL="pool.woolypooly.com:3104"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   FiroPow is limited by memory bandwidth: the memory clock / offset raise it, the core clock does not.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm firopow --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
