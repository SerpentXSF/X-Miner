// --benchmark for fishhash / karlsenhashv2: a fixed job with an easy share target (one share per 2^26 nonces); every
// GPU candidate is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "algo/fishhash/fishhash_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::fishhash {

Job bench_job(Variant v) {
  Job job;
  job.id = "bench";
  for (int i = 0; i < 180; ++i) job.header[i] = static_cast<uint8_t>(i * 73 + 19);
  for (int i = 0; i < 32; ++i) job.pre_pow[i] = static_cast<uint8_t>(i * 37 + 11);
  job.timestamp = 1761600000000ull;
  job.target[v == Variant::Iron ? 3 : 28] = 0x40;  // 2^230 in the variant's byte order
  job.difficulty = 67108864.0;
  job.nonce_prefix = 0x0f57ull << 48;
  job.prefix_bits = 16;
  return job;
}

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs, Variant v) {
  const Job job = bench_job(v);
  const char* name = v == Variant::Iron ? "fishhash" : "karlsenhashv2";
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0xf15a0b0 + d, v);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + name + ", " +
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

}  // namespace xm::fishhash
