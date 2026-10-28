#!/bin/sh
# X-Miner - Ethereum Classic (ETC, Etchash) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-etchash-kryptex.sh
# Pools: etc-us.kryptex.network:7033!etc.kryptex.network:7033
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="etc-us.kryptex.network:7033!etc.kryptex.network:7033"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   Etchash is limited by memory: raise the memory clock / offset; the core clock barely matters.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm etchash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
