#!/bin/sh
# X-Miner - Xelis (XEL, XelisHashV3) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-xelis-kryptex.sh
# Pools: xel-us.kryptex.network:7019!xel.kryptex.network:7019
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="xel-us.kryptex.network:7019!xel.kryptex.network:7019"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   XelisHashV3 gives each hash a 544 KB scratchpad of random reads: memory latency bound, low power draw.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm xelishashv3 --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
