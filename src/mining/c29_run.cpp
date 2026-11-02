// Tari Cuckaroo29 mining: the shared coordinator with the Cuckaroo29 solver and the Kryptex c29 stratum pool.
#include "algo/c29/c29_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/c29_pool.hpp"

namespace xm {

namespace {

struct C29Traits {
  static constexpr const char* kName = "cuckaroo29";
  static constexpr bool kCpuRejectIsError = true;  // every proof is verified on the CPU before it is counted
  using Pool = C29Pool;
  using Job = c29::Job;
  using Share = c29::Share;
  using Engine = c29::Engine;
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int) {
    return std::make_unique<Engine>(dev, seed);
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) { return e.iterate(shares, err); }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("not found") != std::string::npos || err.find("nvalid job") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_c29(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) { return run_mining<C29Traits>(cfg, devs); }

}  // namespace xm
