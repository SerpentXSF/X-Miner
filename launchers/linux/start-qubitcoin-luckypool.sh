#!/bin/sh
# X-Miner - Qubitcoin (QTC, QHash) on LuckyPool (your Qubitcoin address: bc1q..., 3... or 1...)
# Edit WALLET and WORKER below, then run: ./start-qubitcoin-luckypool.sh
# Pools: qtc-us-east.lproute.com:8610!qtc-eu.lproute.com:8610
cd "$(dirname "$0")"

WALLET="YOUR_QUBITCOIN_ADDRESS"
WORKER="$(hostname)"
POOL="qtc-us-east.lproute.com:8610!qtc-eu.lproute.com:8610"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   QHash is SHA-256 plus a small 16-qubit circuit: compute bound, runs at the power limit.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm qhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
