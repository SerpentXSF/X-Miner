// Etchash (ETC) and Ethash (ETHW) mining: the shared coordinator with the Etchash engine and the EthereumStratum/1.0.0
// pool.
#include "algo/etchash/etchash_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/ethstratum_pool.hpp"

namespace xm {

namespace {

struct EtcTag {
  static constexpr const char* kName = "etchash";
  static constexpr bool kEcip1099 = true;
};
struct EthwTag {
  static constexpr const char* kName = "ethash";
  static constexpr bool kEcip1099 = false;
};

template <class Tag>
struct EtchashTraits {
  static constexpr const char* kName = Tag::kName;
  static constexpr bool kCpuRejectIsError = false;  // the GPU compares the top 64 bits only
  using Pool = EthStratumPool;
  using Job = etchash::Job;
  using Share = etchash::Share;
  struct Engine : etchash::Engine {
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : etchash::Engine(dev, seed, Tag::kEcip1099), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, etchash::batch_nonces(intensity));
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

int run_etchash(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<EtchashTraits<EtcTag>>(cfg, devs);
}

int run_ethash(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<EtchashTraits<EthwTag>>(cfg, devs);
}

}  // namespace xm
