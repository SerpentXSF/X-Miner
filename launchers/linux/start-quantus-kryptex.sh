#!/bin/sh
# X-Miner - Quantus (QTC) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-quantus-kryptex.sh
# Pools: qtc-us.kryptex.network:7049!qtc.kryptex.network:7049
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="qtc-us.kryptex.network:7049!qtc.kryptex.network:7049"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Example:
# TUNING="--gpu-plimit 70% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70"
TUNING=""

while true; do
  ./x-miner --algorithm quantus --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
