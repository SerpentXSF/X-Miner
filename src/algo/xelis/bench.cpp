// --benchmark for xelishashv3: a fixed work with an easy share difficulty (one share per 2^10 hashes); every GPU
// candidate is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "algo/xelis/xelis_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::xelis {

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  Job job;
  job.id = "bench";
  for (int i = 0; i < kInputSize; ++i) job.work.bytes[i] = static_cast<uint8_t>(i * 73 + 19);
  job.difficulty = 1024;
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    const int in = cfg.intensity_for(d);
    Engine eng(*dev, 0x8e1150b0 + d, in > 0 ? (1u << std::min(in, 20)) : 0u);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": xelishashv3, " + std::to_string(eng.threads()) +
                      " hashes in flight, " + std::to_string(cfg.benchmark_seconds) + " s");
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    double total = 0, window = 0, watts = 0;
    int samples = 0;
    size_t shares = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds && !stop_requested()) {
      std::vector<Share> found;
      double n = eng.iterate(found, err);
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
    double expected = total / 1024.0;
    char line[240];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), %llu GPU candidates failed the CPU check%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  static_cast<unsigned long long>(eng.cpu_rejects()),
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    bool plausible = std::abs(double(shares) - expected) <= 4 * std::sqrt(expected) + 2;  // Poisson, 4 sigma
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::xelis
