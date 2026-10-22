#!/usr/bin/env bash
# HiveOS stats: reads the X-Miner API (/api/summary) and sets $khs and $stats for the agent.
# Sourced by the agent, so only variables are set; no exit.
. /hive/miners/custom/x-miner/h-manifest.conf

summary=$(curl -s --max-time 3 "http://127.0.0.1:${WEB_PORT}/api/summary")
if [[ -z $summary ]] || ! echo "$summary" | jq -e .x >/dev/null 2>&1; then
  khs=0
  stats=""
else
  # Hash units: Pearl reports MAC/s as H/s; HiveOS shows hs_units, so send kH/s for both algorithms.
  khs=$(echo "$summary" | jq '.x.hr["1m"] / 1000')
  stats=$(echo "$summary" | jq -c --arg ver "$CUSTOM_VERSION" '{
      hs: [.x.devices[].hr_1m / 1000],
      hs_units: "khs",
      temp: [.x.devices[].temp | if . < 0 then 0 else (. | floor) end],
      fan: [.x.devices[].fan | if . < 0 then 0 else (. | floor) end],
      bus_numbers: [.x.devices[].bus],
      uptime: (.x.uptime | floor),
      ar: [.x.shares.accepted, (.x.shares.rejected + .x.shares.stale)],
      algo: .x.algorithm,
      ver: $ver
    }')
fi
[[ -z $khs ]] && khs=0
