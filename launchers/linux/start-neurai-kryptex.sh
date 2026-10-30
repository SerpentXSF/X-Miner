#!/bin/sh
# X-Miner - Neurai (XNA, KawPow) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-neurai-kryptex.sh
# Pools: xna-us.kryptex.network:7024!xna.kryptex.network:7024
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="xna-us.kryptex.network:7024!xna.kryptex.network:7024"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Measured examples:
#   RX 9070 XT: TUNING="--gpu-plimit 70% --gpu-mclock 2700"   (+7%; 2800 was unstable on our card)
#   RTX 3060:   TUNING="--gpu-moffset 1000 --gpu-cclock 1500"   (+7% at 135 W instead of 169 W)
#   KawPow is limited by memory bandwidth: memory clock / offset raise it, core clock and voltage do not.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm neurai --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
