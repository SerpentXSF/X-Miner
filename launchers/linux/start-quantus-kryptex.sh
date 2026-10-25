#!/bin/sh
# X-Miner - Quantus (QTC) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-quantus-kryptex.sh
# Pools: qtc-us.kryptex.network:7049!qtc.kryptex.network:7049
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="qtc-us.kryptex.network:7049!qtc.kryptex.network:7049"

while true; do
  ./x-miner --algorithm quantus --pool "$POOL" --wallet "$WALLET" --worker "$WORKER"
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
