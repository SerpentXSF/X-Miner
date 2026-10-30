// ProgPoW-family mining: the shared coordinator with the ProgPoW engine and each coin's pool (kawpowminer-style stratum
// for KawPow and FiroPow, Epic Cash stratum, Zano ethproxy).
#include "algo/kawpow/kawpow_gpu.hpp"
#include "mining/coordinator.hpp"
#include "pool/epic_pool.hpp"
#include "pool/kawpow_pool.hpp"
#include "pool/zano_pool.hpp"

namespace xm {

namespace {

struct KawpowTag {
  static constexpr const char* kName = "kawpow";
  static const kawpow::Variant& variant() { return kawpow::kKawpow; }
  using Pool = KawpowPool;
  static constexpr double kShareRateLimit = 0;
};
struct FiropowTag {
  static constexpr const char* kName = "firopow";
  static const kawpow::Variant& variant() { return kawpow::kFiropow; }
  using Pool = KawpowPool;
  static constexpr double kShareRateLimit = 0;
};
struct NeuraiTag {  // KawPow unchanged; its own name so the dev fee mines on a Neurai pool
  static constexpr const char* kName = "neurai";
  static const kawpow::Variant& variant() { return kawpow::kKawpow; }
  using Pool = KawpowPool;
  static constexpr double kShareRateLimit = 0;
};
struct EpicTag {
  static constexpr const char* kName = "progpow-epic";
  static const kawpow::Variant& variant() { return kawpow::kProgpow092; }
  using Pool = EpicPool;
  static constexpr double kShareRateLimit = 60;  // per GPU; the pool answers ~60/s per connection (EpicPool sends 50)
};
struct ZanoTag {
  static constexpr const char* kName = "progpowz";
  static const kawpow::Variant& variant() { return kawpow::kProgpowZ; }
  using Pool = ZanoPool;
  static constexpr double kShareRateLimit = 0;
};

template <class Tag>
struct ProgpowTraits {
  static constexpr const char* kName = Tag::kName;
  // The GPU pre-filters on the top 64 hash bits, so a CPU reject is an expected near miss, not an error.
  static constexpr bool kCpuRejectIsError = false;
  using Pool = typename Tag::Pool;
  using Job = kawpow::Job;
  using Share = kawpow::Share;
  struct Engine : kawpow::Engine {  // the engine plus its launch size
    Engine(gpu::Device& dev, uint64_t seed, uint32_t b) : kawpow::Engine(dev, seed, Tag::variant()), batch(b) {}
    uint32_t batch;
  };
  static std::unique_ptr<Engine> make_engine(gpu::Device& dev, uint64_t seed, int intensity) {
    auto e = std::make_unique<Engine>(dev, seed, kawpow::batch_nonces(intensity));
    e->set_share_rate_limit(Tag::kShareRateLimit);
    return e;
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

int run_neurai(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ProgpowTraits<NeuraiTag>>(cfg, devs);
}

int run_epic(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ProgpowTraits<EpicTag>>(cfg, devs);
}

int run_zano(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  return run_mining<ProgpowTraits<ZanoTag>>(cfg, devs);
}

}  // namespace xm
