// Parano1d mining: the shared coordinator with the Parano1d engine and pool.
#include "algo/noid/noid_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/noid_pool.hpp"

namespace xm {

namespace {

struct NoidTraits {
  static constexpr const char* kName = "noid";
  // The GPU pre-filters on the top 64 digest bits, so a CPU reject is an expected near miss, not an error.
  static constexpr bool kCpuRejectIsError = false;
  using Pool = NoidPool;
  using Job = noid::Job;
  using Share = noid::Share;
  using Engine = noid::Engine;
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed) {
    return std::make_unique<Engine>(dev, seed);
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) {
    // ~0.25 s on an RX 9070 XT, ~0.45 s on an RTX 3060: a share found in a batch that straddles a new block is stale.
    return e.iterate(1u << 22, shares, err);
  }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("superseded") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_noid(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<NoidTraits>(cfg, devs);
}

}  // namespace xm
