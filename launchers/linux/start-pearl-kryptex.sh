#!/bin/sh
# X-Miner - Pearl (PRL) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
# Edit WALLET and WORKER below, then run: ./start-pearl-kryptex.sh
# Pools: prl-us.kryptex.network:7048!prl.kryptex.network:7048
cd "$(dirname "$0")"

WALLET="YOUR_KRYPTEX_ADDRESS"
WORKER="$(hostname)"
POOL="prl-us.kryptex.network:7048!prl.kryptex.network:7048"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root). Measured examples:
#   RX 9070 XT, best TH/W: TUNING="--gpu-plimit 70%"
#   RTX 3060, best TH/W:   TUNING="--gpu-cclock 1500"
#   Pearl on NVIDIA: do not lock the memory clock (--gpu-mclock), it costs 8-28%.
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm pearlhash --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
