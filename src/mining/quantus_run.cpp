// Quantus mining: the shared coordinator with the Quantus engine and pool.
#include "algo/quantus/quantus_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/quantus_pool.hpp"

namespace xm {

namespace {

struct QuantusTraits {
  static constexpr const char* kName = "quantus";
  // The GPU pre-filters on the top 64 hash bits, so a CPU reject is an expected near miss, not an error.
  static constexpr bool kCpuRejectIsError = false;
  using Pool = QuantusPool;
  using Job = qtc::Job;
  using Share = qtc::Share;
  struct Engine : qtc::Engine {  // the engine plus its launch size
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : qtc::Engine(dev, seed), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, qtc::batch_nonces(intensity));
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

int run_quantus(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<QuantusTraits>(cfg, devs);
}

}  // namespace xm
