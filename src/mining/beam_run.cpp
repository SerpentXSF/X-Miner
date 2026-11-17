// Beam BeamHash III mining: the shared coordinator with the BeamHash III solver and the Beam stratum pool.
#include "algo/beam/beam_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/beam_pool.hpp"

namespace xm {

namespace {

struct BeamTraits {
  static constexpr const char* kName = "beamhash3";
  static constexpr bool kCpuRejectIsError = true;  // every proof is verified on the CPU before it is counted
  using Pool = BeamPool;
  using Job = beam::Job;
  using Share = beam::Share;
  using Engine = beam::Engine;
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

int run_beam(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) { return run_mining<BeamTraits>(cfg, devs); }

}  // namespace xm
