#!/bin/sh
# X-Miner - Quantus (QTC) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-quantus-kryptex.sh
# Pools: qtc-us.kryptex.network:7049!qtc.kryptex.network:7049
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="qtc-us.kryptex.network:7049!qtc.kryptex.network:7049"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Measured examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70% --gpu-voffset -50"   (+4.5% at the same power)
#   RTX 3060:   TUNING="--gpu-mclock 810"   (+1%: memory power goes to the core)
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm quantus --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
