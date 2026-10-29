#!/bin/sh
# X-Miner - Epic Cash (EPIC, ProgPoW) on epic.pool-pay.com (your epicbox address; the worker is joined with +)
# Edit WALLET and WORKER below, then run: ./start-epic-poolpay.sh
# Pools: epic.pool-pay.com:3333
cd "$(dirname "$0")"

WALLET="YOUR_EPICBOX_ADDRESS@epicbox.epiccash.com"
WORKER="$(hostname)"
POOL="epic.pool-pay.com:3333"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   The pool credits every share at a fixed difficulty and answers about 60 shares per second, so X-Miner sends at most 50 per second.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm progpow-epic --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
