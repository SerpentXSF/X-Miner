// Builds the API summary and prints the periodic console stats table.
#pragma once

#include <string>

#include "core/stats.hpp"
#include "nlohmann/json.hpp"

namespace xm {

struct RunState {
  std::string algorithm;
  std::string pool;          // "host:port" currently connected
  bool connected = false;
  double fee_percent = 0;
  bool fee_window = false;
  double fee_next_s = 0;
};

// SRBMiner-compatible top-level fields (rig_name, miner_version, mining_time, algorithms[...]) so existing
// monitors can read X-Miner, plus an "x" object with everything the dashboard shows.
nlohmann::json build_summary(const Snapshot& s, const RunState& r);

void print_stats_table(const Snapshot& s, const RunState& r);

}  // namespace xm
