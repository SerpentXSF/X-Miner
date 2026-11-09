#!/bin/sh
# X-Miner - Meowcoin (MEWC, MeowPow) on zpool (paid in the coin you choose: BTC, LTC, DOGE, ...)
# Edit WALLET, PAYOUT and WORKER below, then run: ./start-meowcoin-zpool.sh   PAYOUT is the coin of WALLET.
# Pool: meowpow.na.mine.zpool.ca:1327 (zpool lists other regions on its site)
cd "$(dirname "$0")"

WALLET="YOUR_PAYOUT_ADDRESS"
PAYOUT="BTC"
WORKER="$(hostname)"
POOL="meowpow.na.mine.zpool.ca:1327"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   needs a 12 GB card (the DAG is ~10.6 GB); the memory clock / offset raises it most.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm meowpow --pool "$POOL" --wallet "$WALLET" --password "c=$PAYOUT" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
