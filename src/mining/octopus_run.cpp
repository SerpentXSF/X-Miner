// Octopus mining: the shared coordinator with the Octopus engine and the Conflux stratum pool.
#include "algo/octopus/octopus_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/cfx_pool.hpp"

namespace xm {

namespace {

struct OctopusTraits {
  static constexpr const char* kName = "octopus";
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = CfxPool;
  using Job = octopus::Job;
  using Share = octopus::Share;
  struct Engine : octopus::Engine {
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : octopus::Engine(dev, seed), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, octopus::batch_nonces(intensity));
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) {
    return e.iterate(e.batch, shares, err);
  }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("not found") != std::string::npos || err.find("nvalid job") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_octopus(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<OctopusTraits>(cfg, devs);
}

}  // namespace xm
