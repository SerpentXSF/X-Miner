#!/bin/sh
# X-Miner - Conflux (CFX, Octopus) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-octopus-kryptex.sh
# Pools: cfx-us.kryptex.network:7027!cfx.kryptex.network:7027
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="cfx-us.kryptex.network:7027!cfx.kryptex.network:7027"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70%"
#   Octopus is limited by the DAG mix's memory bandwidth (it needs a GPU with 10 GB or more): a memory overclock
#   helps, e.g. --gpu-mclock 2700 on the RX 9070 XT (+7.5%).
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm octopus --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
