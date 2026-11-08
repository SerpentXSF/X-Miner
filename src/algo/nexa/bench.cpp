// --benchmark for nexapow: a fixed job with an easy share difficulty (one share per ~2^24 hashes); every GPU
// candidate is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "algo/nexa/nexa_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::nexa {

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  Job job;
  job.id = "bench";
  for (int i = 0; i < 32; ++i) job.commitment[i] = static_cast<uint8_t>(i * 73 + 19);
  job.extranonce1 = {0x31, 0x56, 0xfb, 0xe5};
  job.nbits = 0x1b011c82;
  job.difficulty = 1.0 / 256;  // ~2^24 hashes per share
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    const int in = cfg.intensity_for(d);
    Engine eng(*dev, 0x4e455841 + d, in > 0 ? (1u << std::min(in, 31)) : 0u);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": nexapow (comb window " +
                      std::to_string(eng.window()) + "), " + std::to_string(eng.batch()) + " nonces per launch, " +
                      std::to_string(cfg.benchmark_seconds) + " s");
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    double total = 0, window = 0, watts = 0;
    int samples = 0;
    size_t shares = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds &&
           !stop_requested()) {
      std::vector<Share> found;
      const double n = eng.iterate(found, err);
      if (n == 0) {
        error("GPU", err);
        ++failures;
        break;
      }
      total += n;
      window += n;
      shares += found.size();
      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 5) {
        const auto r = sensors.read(d);
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
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double expected = total * 65535.0 / 281474976710656.0 / job.difficulty;  // P = 0xFFFF 2^-48 / d
    char line[240];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), %llu GPU candidates dropped by the CPU check%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  static_cast<unsigned long long>(eng.cpu_rejects()),
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    // The kernel hashes both candidate signatures (k and n - k; kernels/nexa/nexa.cu nexa_stage2_sig), so about as
    // many candidates as shares are dropped by the CPU check: only the share count decides.
    const bool plausible = std::abs(double(shares) - expected) <= 4 * std::sqrt(expected) + 2;
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::nexa
