#!/bin/sh
# X-Miner - Parano1d (NOID) on rplant.xyz (1% pool fee)
# Edit WALLET (your o1... NOID address) and WORKER below, then run: ./start-noid-rplant.sh
cd "$(dirname "$0")"

WALLET="YOUR_NOID_ADDRESS"
WORKER="$(hostname)"
POOL="na.rplant.xyz:17185"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Measured examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70% --gpu-voffset -50"   (+3.8% at the same power)
#   RTX 3060:   TUNING="--gpu-mclock 810"   (same speed, 15 W less)
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm noid --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
