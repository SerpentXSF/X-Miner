// FishHash (Iron Fish, native Iron Fish stratum) and KarlsenHashV2 (Karlsen, Kaspa-style stratum) mining: the shared
// coordinator with the FishHash engine.
#include "algo/fishhash/fishhash_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/iron_pool.hpp"
#include "pool/kls_pool.hpp"

namespace xm {

namespace {

template <class P, fishhash::Variant V>
struct FishTraits {
  static constexpr const char* kName = V == fishhash::Variant::Iron ? "fishhash" : "karlsenhashv2";
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = P;
  using Job = fishhash::Job;
  using Share = fishhash::Share;
  struct Engine : fishhash::Engine {
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : fishhash::Engine(dev, seed, V), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, fishhash::batch_nonces(intensity));
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

int run_fishhash(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<FishTraits<IronPool, fishhash::Variant::Iron>>(cfg, devs);
}

int run_karlsen(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<FishTraits<KlsPool, fishhash::Variant::Karlsen>>(cfg, devs);
}

}  // namespace xm
