#!/bin/sh
# X-Miner - Iron Fish (IRON, FishHash) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-fishhash-kryptex.sh
# Pools: iron-us.kryptex.network:7017!iron.kryptex.network:7017
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="iron-us.kryptex.network:7017!iron.kryptex.network:7017"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   FishHash is limited by random memory reads (4.5 GB dataset): a memory overclock helps, e.g. --gpu-mclock 2700 on the RX 9070 XT (+7%).
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm fishhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
