// --benchmark for qhash: a fixed work with an easy share difficulty (one share per ~2^22 hashes); every GPU candidate
// is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "algo/qhash/qhash_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::qhash {

Job bench_job() {
  Job job;
  job.id = "bench";
  for (int i = 0; i < 60; ++i) job.tpl.coinb1.push_back(static_cast<uint8_t>(i * 37 + 11));
  for (int i = 0; i < 40; ++i) job.tpl.coinb2.push_back(static_cast<uint8_t>(i * 53 + 7));
  job.tpl.extranonce1 = {0x21, 0x65, 0xc7, 0xfe};
  job.tpl.extranonce2_size = 4;
  for (int i = 0; i < 32; ++i) job.tpl.prevhash[i] = static_cast<uint8_t>(i < 26 ? i * 91 + 5 : 0);
  job.tpl.version = 0x20000004;
  job.tpl.nbits = 0x1a0de1e9;
  job.tpl.ntime = 0x6ae35509;  // after the odd-angle and zero-byte switches
  job.ntime_hex = "6ae35509";
  job.difficulty = 1.0 / 1024;  // ~2^22 hashes per share
  return job;
}

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  const Job job = bench_job();
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    const int in = cfg.intensity_for(d);
    Engine eng(*dev, 0x9b17c0d3 + d, in > 0 ? (1u << std::min(in, 31)) : 0u);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": qhash (" + (eng.fp64() ? "fp64" : "fp32") +
                      "), " + std::to_string(eng.batch()) + " nonces per launch, " +
                      std::to_string(cfg.benchmark_seconds) + " s");
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
    // P(share) = 0xFFFF * 2^-48 / difficulty for a valid hash; ~95.2% of hashes pass the zero-byte rule.
    double expected = total * 65535.0 / 281474976710656.0 / job.difficulty * 0.952;
    char line[240];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), %llu GPU candidates failed the CPU check%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  static_cast<unsigned long long>(eng.cpu_rejects()),
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    bool plausible = std::abs(double(shares) - expected) <= 4 * std::sqrt(expected) + 2 + 0.01 * expected;  // fp32
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::qhash
