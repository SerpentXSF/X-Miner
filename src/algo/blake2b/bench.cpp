// --benchmark for blake2b: the stratum view of the Knots regtest activation block with an easy share target; every
// GPU candidate is re-checked with the CPU reference and the share count is compared with the expected rate.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "algo/blake2b/blake2b_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::b2b {

namespace {

void hex_into(const char* hex, uint8_t* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    unsigned v = 0;
    std::sscanf(hex + 2 * i, "%2x", &v);
    out[i] = static_cast<uint8_t>(v);
  }
}

}  // namespace

Job bench_job() {
  Job job;
  job.id = "bench";
  hex_into("0000000000004fee2e5613820a8fa7540fa030c58f0b363a28ee63af2ba84bc7", job.prevhash.data(), 32);
  hex_into("0000009650adc5f76b847f7d0a340d604917eea59e0a5d823214ca7974490984655f2700000000", job.coinb1.data(),
           kCoinb1Size);
  job.ntime_hex = "0000000000000000";
  job.extranonce2_hex = "0000000000000000";
  job.share_target[29] = 0x01;  // target 2^232: about one share per 2^24 nonces
  return job;
}

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  const Job job = bench_job();
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0xb1a4e2b0 + d);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": blake2b, " +
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
    double expected = total / 16777216.0;  // 2^24 nonces per share at this target
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

}  // namespace xm::b2b
