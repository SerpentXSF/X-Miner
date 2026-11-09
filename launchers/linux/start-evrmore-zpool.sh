#!/bin/sh
# X-Miner - Evrmore (EVR, EvrProgPow) on zpool (paid in the coin you choose: BTC, LTC, DOGE, ...)
# Edit WALLET, PAYOUT and WORKER below, then run: ./start-evrmore-zpool.sh   PAYOUT is the coin of WALLET.
# Pool: evrprogpow.na.mine.zpool.ca:1330 (zpool lists other regions on its site)
cd "$(dirname "$0")"

WALLET="YOUR_PAYOUT_ADDRESS"
PAYOUT="BTC"
WORKER="$(hostname)"
POOL="evrprogpow.na.mine.zpool.ca:1330"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   KawPoW-like: the memory clock / offset raises it most; the core clock helps a little.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm evrprogpow --pool "$POOL" --wallet "$WALLET" --password "c=$PAYOUT" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
