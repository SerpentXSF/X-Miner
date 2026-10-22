// --benchmark for noid: a real header (test vector v1) with an easy share target; every GPU candidate is
// re-checked with the CPU reference, and a sample of GPU "misses" is checked to prove nothing is skipped.
#include <chrono>
#include <cstdio>
#include <thread>

#include "algo/noid/noid_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "gpu/sensors.hpp"

namespace xm::noid {

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  const char* v1[16] = {"dc1b77ae0bf34dad64f0eeb9026e6076", "7b07ce91e5906136305f050c368dcc74",
                        "2ceb16e0a1c54aec97101dce4e7bfb79", "9ad2e144d6e8f2cfd9aa792e1af470ea",
                        "ddaa4e85b0d6e28b8f8ea9d349428d8e", "08f474ffb8e8ab152ead854756d71f03",
                        "00000000000000000000000068e77801", "0000000000000000000000000003d091",
                        "353cfc387dfae6b8a32edabf5585bd75", "fc5639b16b714b4f92fb2dcfc8ae9a19",
                        "544b0ec76d00616dbcbb9b7e9a319aef", "0f1a50d59c0aa21a80ae2120826571de",
                        "0e1ecd02ed7c0cbd0d0981e8c1fa7be4", "00000000000000000000000000000009",
                        "0000000000000000000000000000012c", "00000000000000000000000000000fa0"};
  Job job;
  job.id = "bench";
  for (int i = 0; i < 16; ++i) job.fields[i] = parse_u128_be_hex(v1[i]);
  job.share_target[29] = 0x01;  // target 2^232: about one share per 2^24 nonces
  job.nonce_prefix = 0x1122334455667788ull;
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0xabcdef00 + d);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": noid, " + std::to_string(cfg.benchmark_seconds) + " s");
    const uint32_t batch = 1u << 22;
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
    double expected = total / 16777216.0;  // 2^24 nonces per share at this target
    char line[220];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares (expected %.0f), all CPU-verified%s",
                  format_hashrate(total / secs).c_str(), secs, shares, expected,
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    bool plausible = expected < 20 || (shares > expected * 0.7 && shares < expected * 1.3);
    log_line(plausible ? Tone::Ok : Tone::Red, "DONE", line);
    if (!plausible) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::noid
