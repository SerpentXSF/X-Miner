// BLAKE2b (RDTS) mining: the shared coordinator with the BLAKE2b engine and pool. No dev fee (devfee.hpp).
#include "algo/blake2b/blake2b_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/blake2b_pool.hpp"

namespace xm {

namespace {

struct Blake2bTraits {
  static constexpr const char* kName = "blake2b";
  // The GPU pre-filters on the top 64 digest bits, so a CPU reject is an expected near miss, not an error.
  static constexpr bool kCpuRejectIsError = false;
  using Pool = Blake2bPool;
  using Job = b2b::Job;
  using Share = b2b::Share;
  struct Engine : b2b::Engine {  // the engine plus its launch size
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : b2b::Engine(dev, seed), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, b2b::batch_nonces(intensity));
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) {
    return e.iterate(e.batch, shares, err);
  }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("not found") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_blake2b(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<Blake2bTraits>(cfg, devs);
}

}  // namespace xm
