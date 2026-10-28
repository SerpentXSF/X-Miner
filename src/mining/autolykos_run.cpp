// Ergo mining: the shared coordinator with the Autolykos v2 engine and the Ergo stratum pool.
#include "algo/autolykos/autolykos_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/ergo_pool.hpp"

namespace xm {

namespace {

struct ErgoTraits {
  static constexpr const char* kName = "autolykos2";
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = ErgoPool;
  using Job = autolykos::Job;
  using Share = autolykos::Share;
  struct Engine : autolykos::Engine {
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : autolykos::Engine(dev, seed), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, autolykos::batch_nonces(intensity));
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

int run_autolykos(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ErgoTraits>(cfg, devs);
}

}  // namespace xm
