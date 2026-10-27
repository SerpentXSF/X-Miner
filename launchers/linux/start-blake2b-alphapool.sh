#!/bin/sh
# X-Miner - Bitcoin Knots BLAKE2b (RDTS) on AlphaPool. No X-Miner dev fee on blake2b.
# Edit WALLET (your RDTS payout address) and WORKER below, then run: ./start-blake2b-alphapool.sh
# AlphaPool's share difficulty is sized for ASICs: a GPU finds a share every few hours or days (lottery mining).
cd "$(dirname "$0")"

WALLET="YOUR_RDTS_ADDRESS"
WORKER="$(hostname)"
POOL="us1.alphapool.tech:5555"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Example:
# TUNING="--gpu-plimit 70% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70"
TUNING=""

while true; do
  ./x-miner --algorithm blake2b --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
