#!/bin/sh
# X-Miner - Nexa (NEXA) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-nexa-kryptex.sh
# Pools: nexa-us.kryptex.network:7026!nexa.kryptex.network:7026
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="nexa-us.kryptex.network:7026!nexa.kryptex.network:7026"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root).
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm nexapow --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
