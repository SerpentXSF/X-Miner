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
    {"neurai", "xna-us.kryptex.network:7024", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"quai", "quai-kawpow.kryptex.network:7043", "0x00498b94BA83D99043E592162F68e1B681835E77", "xminer-fee", kFeePercent},
    {"firopow", "pool.woolypooly.com:3104", "aHc8mHivZB9qM6MhKRRdPchpEBzoqWco8D", "xminer-fee", kFeePercent},
    {"etchash", "etc-us.kryptex.network:7033", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"ethash", "ethw.kryptex.network:7034", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"octopus", "cfx-us.kryptex.network:7027", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"fishhash", "iron-us.kryptex.network:7017", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"karlsenhashv2", "pool.woolypooly.com:3132", "karlsen:qqw709h9d6aagh8gvhrv63weq96tys6g89urtvfya4shfa54gc89sh8ux3nk4", "xminer-fee", kFeePercent},
    {"progpow-epic", "epic.pool-pay.com:3333", "esWj6iMYQV6TjhYphEMhLN9qe7BpKY9LUHtG4XZM6f7BJ6fYTxC3@epicbox.epiccash.com", "xminer-fee", kFeePercent},
    {"progpowz", "us.zano.herominers.com:1110", "ZxDHDLAjf3oQVk6nCF7Cjcaa4cMeZsMjybGjUwrMG5jFDJidCTSLQPQ7ixTmEYGnSbESR1vTQrFjr6yHGx7PUZ8U2HYcrgjjX", "xminer-fee", kFeePercent},
    {"autolykos2", "erg-us.kryptex.network:7021", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"xelishashv3", "xel-us.kryptex.network:7019", "krxY9VNKGD", "xminer-fee", kFeePercent},
    {"qhash", "qtc-us-east.lproute.com:8610", "bc1qrgqcl96ts6dwkx079qhlkuxjr7jn5gq2489q0l", "xminer-fee", kFeePercent},
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
