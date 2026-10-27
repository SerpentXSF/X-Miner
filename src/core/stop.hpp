// Graceful stop: Ctrl+C, SIGTERM and SIGHUP (closing the terminal, HiveOS "miner stop") set the stop flag; mining and
// benchmarks poll it and return, so main restores GPU tuning on the way out.
#pragma once

#include <atomic>

namespace xm {

inline std::atomic<bool>& stop_flag() {
  static std::atomic<bool> stop{false};
  return stop;
}
inline bool stop_requested() { return stop_flag().load(); }

}  // namespace xm
