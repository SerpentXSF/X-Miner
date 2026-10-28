// --benchmark for etchash: a fixed header at Ethereum Classic's late-2026 height (25.6 M, epoch 427, a 4.4 GB DAG)
// with an easy share target; every GPU candidate is re-checked with the CPU reference and the share count is compared
// with the expected rate.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <thread>

#include "algo/etchash/etchash_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::etchash {

Job bench_job() {
  Job job;
  job.id = "bench";
  job.height = 25624706;
  uint8_t* h = reinterpret_cast<uint8_t*>(job.header.data());
  for (int i = 0; i < 32; ++i) h[i] = static_cast<uint8_t>(i * 73 + 19);
  reinterpret_cast<uint8_t*>(job.target.data())[3] = 0x40;  // target 2^230: about one share per 2^26 nonces
  job.difficulty = 67108864.0;
  job.nonce_prefix = 0x0f57ull << 48;
  job.prefix_bits = 16;
  return job;
}

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  Job job = bench_job();
  if (const char* e = std::getenv("XM_BENCH_HEIGHT")) job.height = std::atoi(e);  // developer: other DAG sizes
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0xe7c0a0b0 + d);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": etchash (epoch " + std::to_string(epoch_for_height(job.height)) + "), " +
                      std::to_string(cfg.benchmark_seconds) + " s");
    const uint32_t batch = batch_nonces(cfg.intensity_for(d));
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    double total = 0, window = 0, watts = 0;
    int samples = 0;
    size_t shares = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds && !stop_requested()) {
      std::vector<Share> found;
      double n = eng.iterate(batch, found, err);
      if (n == 0) {
        error("GPU", err);
        ++failures;
        break;
      }
      total += n;
      window += n;
      shares += found.size();
      auto now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 5) {
        auto r = sensors.read(d);
        if (r.power_w > 0) {
          watts += r.power_w;
          ++samples;
        }
        log_line(Tone::Radar, "RATE", format_hashrate(window / dt) +
                                          (r.power_w > 0 ? "  " + std::to_string(int(r.power_w)) + " W" : "") +
                                          "  shares " + std::to_string(shares));
        window = 0;
        last = now;
      }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double expected = total / 67108864.0;  // 2^26 nonces per share at this target
    char line[220];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), all CPU-verified%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    bool plausible = std::abs(double(shares) - expected) <= 4 * std::sqrt(expected) + 2;  // Poisson, 4 sigma
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::qtc
