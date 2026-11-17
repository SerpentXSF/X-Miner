#!/bin/sh
# X-Miner - Beam (BEAM) on 2Miners (your Beam wallet address; TLS)
# Edit WALLET and WORKER below, then run: ./start-beam-2miners.sh
# Pool: stratum+ssl://beam.2miners.com:5252
cd "$(dirname "$0")"

WALLET="YOUR_BEAM_ADDRESS"
WORKER="$(hostname)"
POOL="stratum+ssl://beam.2miners.com:5252"

# Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"; needs root).
TUNING=""

trap 'exit 0' INT TERM   # Ctrl+C: the miner restores the GPU and exits, then the loop ends too
while true; do
  ./x-miner --algorithm beamhash3 --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" $TUNING
  echo "X-Miner exited (code $?). Restarting in 10 seconds... (Ctrl+C to stop)"
  sleep 10
done
