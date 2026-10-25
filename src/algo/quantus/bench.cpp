// --benchmark for quantus: a fixed header with a 4-byte pool prefix and an easy share target; every GPU candidate
// is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cstdio>
#include <thread>

#include "algo/quantus/quantus_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "gpu/sensors.hpp"

namespace xm::qtc {

Job bench_job() {
  Job job;
  job.id = "bench";
  for (size_t i = 0; i < job.mining_hash.size(); ++i) job.mining_hash[i] = static_cast<uint8_t>(i * 73 + 19);
  job.extranonce = {0x26, 0x1a, 0x6f, 0x72};
  job.share_target[2] = 0x10;  // target 2^492: about one share per 2^20 nonces
  job.difficulty = 1048576.0;
  return job;
}

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  const Job job = bench_job();
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0x9757a0b0 + d);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": quantus, " +
                      std::to_string(cfg.benchmark_seconds) + " s");
    const uint32_t batch = batch_nonces(cfg.intensity_for(d));
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    double total = 0, window = 0, watts = 0;
    int samples = 0;
    size_t shares = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds) {
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
    double expected = total / 1048576.0;  // 2^20 nonces per share at this target
    char line[220];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), all CPU-verified%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    bool plausible = expected < 20 || (shares > expected * 0.9 && shares < expected * 1.1);
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::qtc
