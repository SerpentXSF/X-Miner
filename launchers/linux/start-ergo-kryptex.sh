#!/bin/sh
# X-Miner - Ergo (ERG, Autolykos v2) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-ergo-kryptex.sh
# Pools: erg-us.kryptex.network:7021!erg.kryptex.network:7021
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="erg-us.kryptex.network:7021!erg.kryptex.network:7021"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   Needs about 8 GB of GPU memory (a 7.6 GB table, rebuilt at every block).
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm autolykos2 --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
