#!/usr/bin/env bash
# Builds the x-miner command line from the flight sheet:
#   Hash algorithm -> CUSTOM_ALGO (pearlhash | noid | quantus | blake2b | kawpow | etchash | autolykos2 | firopow | progpow-epic | progpowz | octopus | fishhash | karlsenhashv2), Pool URL -> CUSTOM_URL (several = failover),
#   Wallet and worker template -> CUSTOM_TEMPLATE, Pass -> CUSTOM_PASS, Extra config arguments -> CUSTOM_USER_CONFIG.
cd "$(dirname "$0")"
. h-manifest.conf

conf="--algorithm ${CUSTOM_ALGO:-pearlhash}"
for url in $CUSTOM_URL; do conf+=" --pool $url"; done
[[ -n $CUSTOM_TEMPLATE ]] && conf+=" --wallet $CUSTOM_TEMPLATE"
[[ -n $CUSTOM_PASS ]] && conf+=" --password $CUSTOM_PASS"
conf+=" --api-port $WEB_PORT --no-color $CUSTOM_USER_CONFIG"

mkdir -p "$(dirname "$CUSTOM_CONFIG_FILENAME")"
echo "$conf" > "$CUSTOM_CONFIG_FILENAME"
