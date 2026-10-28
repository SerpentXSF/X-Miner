// The dev fee, in one place.
//
// X-Miner mines for the developer for a short window in each cycle of *mining time* (time spent connected and
// hashing; reconnect gaps do not count). At 1.00% with a 100-minute cycle, that is 60 seconds per cycle, at the
// end of the cycle, so a short test run never pays a fee. The console, API and dashboard always show the fee,
// the time to the next window and the shares found during fee windows.
//
// This is open source: change kFeePercent or kFeeTargets in a fork if you want. Please keep it honest and visible.
#pragma once

#include <cstdint>
#include <string>

namespace xm {

constexpr double kFeePercent = 1.00;
constexpr double kCycleSeconds = 6000.0;  // 100 minutes of mining time per cycle

// Where fee windows mine. A target with an empty wallet disables the fee for that algorithm (and the console
// says so); release builds fill these in.
struct FeeTarget {
  const char* algorithm;
  const char* pool;     // host:port (stratum+ssl:// allowed)
  const char* wallet;   // developer address
  const char* worker;
  double percent;       // per-algorithm override (blake2b is 0)
};

inline constexpr FeeTarget kFeeTargets[] = {
    {"pearlhash", "prl.kryptex.network:7048", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"noid", "na.rplant.xyz:17185", "o1zzxrf4hkz33hq0lzpzxcmvmhv2990m5a8gg43ldxdlvzj2xqeqksk9j9y0", "xminer-fee", kFeePercent},
    {"blake2b", "", "", "", 0.0},
    {"quantus", "qtc-us.kryptex.network:7049", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"kawpow", "rvn-us.kryptex.network:7031", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"firopow", "", "", "", kFeePercent},  // no developer wallet yet: no fee
};

const FeeTarget* fee_target(const std::string& algorithm);

// Pure scheduling logic, driven by elapsed mining time. Not thread-safe; owned by the coordinator.
class FeeSchedule {
 public:
  explicit FeeSchedule(double percent, double cycle_s = kCycleSeconds)
      : window_s_(cycle_s * percent / 100.0), cycle_s_(cycle_s) {}

  // Advance by dt seconds of mining time. Returns true if the miner should now be on the fee pool.
  bool advance(double dt);

  bool in_window() const { return window_s_ > 0 && pos_ >= cycle_s_ - window_s_; }
  double seconds_to_window() const { return in_window() ? 0.0 : (cycle_s_ - window_s_) - pos_; }
  double window_seconds() const { return window_s_; }
  double fee_seconds_total() const { return fee_total_; }
  double mining_seconds_total() const { return mined_total_; }

 private:
  double window_s_;
  double cycle_s_;
  double pos_ = 0.0;        // position within the current cycle
  double fee_total_ = 0.0;
  double mined_total_ = 0.0;
};

}  // namespace xm
