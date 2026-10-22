#!/usr/bin/env bash
cd "$(dirname "$0")"
. h-manifest.conf

[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo "No config $CUSTOM_CONFIG_FILENAME (run h-config.sh)" && exit 1
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

./x-miner $(< "$CUSTOM_CONFIG_FILENAME") 2>&1 | tee --append "${CUSTOM_LOG_BASENAME}.log"
