// KawPow and FiroPow mining: the shared coordinator with the ProgPoW engine and the kawpowminer-style pool.
#include "algo/kawpow/kawpow_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/kawpow_pool.hpp"

namespace xm {

namespace {

struct KawpowTag {
  static constexpr const char* kName = "kawpow";
  static const kawpow::Variant& variant() { return kawpow::kKawpow; }
};
struct FiropowTag {
  static constexpr const char* kName = "firopow";
  static const kawpow::Variant& variant() { return kawpow::kFiropow; }
};

template <class Tag>
struct ProgpowTraits {
  static constexpr const char* kName = Tag::kName;
  // The GPU pre-filters on the top 64 hash bits, so a CPU reject is an expected near miss, not an error.
  static constexpr bool kCpuRejectIsError = false;
  using Pool = KawpowPool;
  using Job = kawpow::Job;
  using Share = kawpow::Share;
  struct Engine : kawpow::Engine {  // the engine plus its launch size
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : kawpow::Engine(dev, seed, Tag::variant()), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    return std::make_unique<Engine>(dev, seed, kawpow::batch_nonces(intensity));
  }
  static double iterate(Engine& e, std::vector<Share>& shares, std::string& err) {
    return e.iterate(e.batch, shares, err);
  }
  static bool is_stale(const std::string& err) {
    return err.find("tale") != std::string::npos || err.find("expired") != std::string::npos ||
           err.find("not found") != std::string::npos || err.find("nvalid job") != std::string::npos ||
           err.find("ob not") != std::string::npos;
  }
  static void submit(Pool& p, const Share& s, int unit, double) { p.submit(s, unit); }
  static void report_rate(Pool& p, double total) { p.report_rate(total); }
};

}  // namespace

int run_kawpow(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ProgpowTraits<KawpowTag>>(cfg, devs);
}

int run_firopow(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ProgpowTraits<FiropowTag>>(cfg, devs);
}

}  // namespace xm
