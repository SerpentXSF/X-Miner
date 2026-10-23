// Hashrate and share statistics, per device and in total. Thread-safe.
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace xm {

using Clock = std::chrono::steady_clock;

struct DeviceSnapshot {
  int id = 0;
  int bus = -1;  // PCI bus (-1 = unknown)
  std::string name;
  double hr_now = 0, hr_1m = 0, hr_15m = 0, hr_1h = 0;  // hashes per second
  uint64_t accepted = 0, rejected = 0, stale = 0;
  double power_w = -1, temp_c = -1, fan_pct = -1, core_mhz = -1;  // -1 = unknown
};

struct Snapshot {
  std::vector<DeviceSnapshot> devices;
  double hr_now = 0, hr_1m = 0, hr_15m = 0, hr_1h = 0;
  uint64_t accepted = 0, rejected = 0, stale = 0;
  uint64_t fee_accepted = 0;
  double uptime_s = 0;
  // Pool-credited ("effective") hashrate: work of the user's accepted shares over the window (or the uptime if
  // shorter). Compare with the pool's dashboard; it is noisy until a few dozen shares are in.
  double eff_1h = 0, eff_24h = 0, eff_total = 0;
};

class Stats {
 public:
  explicit Stats(size_t devices);

  void set_name(int dev, std::string name, int bus = -1);
  // Report that `hashes` hashes were completed on device `dev` during the last `seconds` of work.
  void add_hashes(int dev, double hashes, double seconds);
  // work = hashes the pool credits for this share (2^256 / share target); fee-window shares are not counted.
  void share(int dev, bool accepted, bool stale, bool fee_window, double work = 0);
  void sensors(int dev, double power_w, double temp_c, double fan_pct, double core_mhz);
  Snapshot snapshot() const;

 private:
  struct Sample {
    Clock::time_point t;  // end of the interval
    double hashes;
    double seconds;       // length of the interval
  };
  struct Dev {
    std::string name;
    int bus = -1;
    std::deque<Sample> samples;  // up to 1 hour
    uint64_t accepted = 0, rejected = 0, stale = 0;
    double power_w = -1, temp_c = -1, fan_pct = -1, core_mhz = -1;
  };
  static double rate(const std::deque<Sample>& s, Clock::time_point now, double window_s);

  mutable std::mutex mu_;
  std::vector<Dev> devs_;
  uint64_t fee_accepted_ = 0;
  std::deque<std::pair<Clock::time_point, double>> credited_;  // accepted user shares, up to 24 h
  double credited_total_ = 0;
  Clock::time_point start_ = Clock::now();
};

std::string format_hashrate(double hps);  // "100.43 TH/s"
std::string format_duration(double seconds);  // "6h 12m"

}  // namespace xm
