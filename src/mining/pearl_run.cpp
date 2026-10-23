// Pearl mining: the shared coordinator with the Pearl engine and pool.
#include "algo/pearl/pearl_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/pearl_pool.hpp"

namespace xm {

namespace {

struct PearlTraits {
  static constexpr const char* kName = "pearlhash";
  static constexpr bool kCpuRejectIsError = true;  // every reported tile must match the CPU transcript
  using Pool = PearlPool;
  using Job = pearl::Job;
  using Share = pearl::Share;
  using Engine = pearl::Engine;
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed) {
    return std::make_unique<Engine>(dev, 16384, 32768, seed);  // 16384 x 32768 x 4096 MAC per pass
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) { return e.iterate(shares, err); }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("not found") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double rate) { p.submit(s, unit, rate); }
  static void report_rate(Pool&, double) {}  // the rate travels with each submit ("hs")
};

}  // namespace

int run_pearl(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<PearlTraits>(cfg, devs);
}

}  // namespace xm
