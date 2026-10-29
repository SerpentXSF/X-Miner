// QHash (Qubitcoin) mining: the shared coordinator with the QHash engine and the Bitcoin stratum pool.
#include "algo/qhash/qhash_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/qhash_pool.hpp"

namespace xm {

namespace {

struct QhashTraits {
  static constexpr const char* kName = "qhash";
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = QhashPool;
  using Job = qhash::Job;
  using Share = qhash::Share;
  using Engine = qhash::Engine;
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    // --gpu-intensity N = 2^N nonces per launch.
    return std::make_unique<Engine>(dev, seed, intensity > 0 ? (1u << std::min(intensity, 31)) : 0u);
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) { return e.iterate(shares, err); }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("not found") != std::string::npos ||
           err.find("expired") != std::string::npos || err.find("[21") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_qhash(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) { return run_mining<QhashTraits>(cfg, devs); }

}  // namespace xm
