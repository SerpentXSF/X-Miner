#!/bin/sh
# X-Miner - Karlsen (KLS, KarlsenHashV2) on WoolyPooly (your karlsen:... address)
# Edit WALLET and WORKER below, then run: ./start-karlsen-woolypooly.sh
# Pools: pool.woolypooly.com:3132
cd "$(dirname "$0")"

WALLET="YOUR_KLS_ADDRESS"
WORKER="$(hostname)"
POOL="pool.woolypooly.com:3132"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   KarlsenHashV2 reads the same 4.5 GB dataset as FishHash: a memory overclock helps.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm karlsenhashv2 --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
