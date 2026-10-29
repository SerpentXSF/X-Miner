// XelisHash v3 mining: the shared coordinator with the Xelis engine and the Xelis stratum pool.
#include "algo/xelis/xelis_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/xelis_pool.hpp"

namespace xm {

namespace {

struct XelisTraits {
  static constexpr const char* kName = "xelishashv3";
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = XelisPool;
  using Job = xelis::Job;
  using Share = xelis::Share;
  using Engine = xelis::Engine;
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    // --gpu-intensity N caps the hashes in flight at 2^N (each needs a 544 KB scratchpad).
    return std::make_unique<Engine>(dev, seed, intensity > 0 ? (1u << std::min(intensity, 20)) : 0u);
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) { return e.iterate(shares, err); }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("not found") != std::string::npos || err.find("\"code\":21") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_xelis(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) { return run_mining<XelisTraits>(cfg, devs); }

}  // namespace xm
