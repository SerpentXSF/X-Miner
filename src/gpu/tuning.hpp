// GPU tuning from the command line: core/memory clock locks, clock offsets, power limits, fan speed and (AMD) core voltage
// offset, applied at start and restored on exit (Ctrl+C, closing the console window, SIGTERM / SIGHUP, watchdog exit).
//   NVIDIA (Windows, Linux): NVML. Needs administrator / root rights; without them X-Miner says so and mines anyway.
//   AMD Linux:               amdgpu sysfs (power1_cap, pp_od_clk_voltage). Needs root and the overdrive feature mask.
//   AMD Windows:             ADL Overdrive8 (Adrenalin's tuning settings).
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm::gpu {

struct TuneRequest {
  std::optional<int> core_lock_mhz;   // lock the core clock (min = max)
  std::optional<int> mem_lock_mhz;    // lock the memory clock
  std::optional<int> core_offset_mhz; // V/F curve offset
  std::optional<int> mem_offset_mhz;
  std::optional<double> power_w;      // board power limit in watts
  std::optional<double> power_pct;    // or as a percentage of the default limit (e.g. 70)
  std::optional<int> fan_pct;         // fixed fan speed
  std::optional<int> volt_offset_mv;  // core voltage offset (AMD Overdrive)
  bool any() const {
    return core_lock_mhz || mem_lock_mhz || core_offset_mhz || mem_offset_mhz || power_w || power_pct || fan_pct ||
           volt_offset_mv;
  }
};

class Tuner {
 public:
  explicit Tuner(const std::vector<DeviceInfo>& devices);
  ~Tuner();  // restores everything that was changed
  // Applies req to devices[index]; logs each change or why it could not be made. Returns false if nothing applied.
  bool apply(size_t index, const TuneRequest& req);
  // Safe to call more than once and from several threads. wait=false (exit paths with a possibly hung thread): give
  // up after a few seconds if another restore holds the lock.
  void restore(bool wait = true);
  // Prints what can be tuned on devices[index] and the current values (read-only).
  void report(size_t index);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xm::gpu
