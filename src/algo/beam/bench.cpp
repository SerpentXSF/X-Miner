// --benchmark for beamhash3: runs of random inputs and nonces; reports verified solutions per second (Sol/s), runs per
// second and candidates per run. Every solution is checked by the CPU verifier.
#include <chrono>
#include <cstdio>

#include "algo/beam/beam_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::beam {

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0x4245414d + d);
    std::string err;
    if (!eng.init(err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": beamhash3, " +
                      std::to_string(cfg.benchmark_seconds) + " s");
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    uint64_t runs = 0, sols = 0, window_sols = 0, candidates = 0, overflow = 0;
    double gpu_ms = 0, watts = 0;
    int samples = 0;
    uint64_t seed = 0x8765431 + d;
    auto next = [&] {
      seed = seed * 6364136223846793005ull + 1442695040888963407ull;
      return seed ^ (seed >> 29);
    };
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds &&
           !stop_requested()) {
      uint8_t input[32], nonce[8];
      for (int i = 0; i < 32; i += 8) {
        const uint64_t r = next();
        for (int b = 0; b < 8; ++b) input[i + b] = uint8_t(r >> (8 * b));
      }
      const uint64_t r = next();
      for (int b = 0; b < 8; ++b) nonce[b] = uint8_t(r >> (8 * b));
      std::vector<Solution> found;
      SolveStats st;
      if (!eng.solve(input, nonce, found, st, err)) {
        error("GPU", err);
        ++failures;
        break;
      }
      ++runs;
      sols += found.size();
      window_sols += found.size();
      candidates += st.candidates;
      overflow += st.overflow;
      gpu_ms += st.gpu_ms;
      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 5) {
        const auto rd = sensors.read(d);
        if (rd.power_w > 0) {
          watts += rd.power_w;
          ++samples;
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%.2f Sol/s  %d W  %.2f runs/s", window_sols / dt, int(rd.power_w),
                      runs / std::chrono::duration<double>(now - t0).count());
        log_line(Tone::Radar, "RATE", buf);
        window_sols = 0;
        last = now;
      }
    }
    if (runs == 0) continue;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char line[320];
    std::snprintf(line, sizeof line,
                  "%.2f Sol/s over %.0f s (%llu runs, %.2f solutions per run, %.2f candidates per run, %.1f ms per "
                  "run, %llu elements over bucket capacity, %llu CPU rejects)%s",
                  sols / secs, secs, static_cast<unsigned long long>(runs), double(sols) / double(runs),
                  double(candidates) / double(runs), gpu_ms / double(runs), static_cast<unsigned long long>(overflow),
                  static_cast<unsigned long long>(eng.cpu_rejects()), samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    log_line(Tone::Ok, "DONE", line);
  }
  return failures ? 1 : 0;
}

}  // namespace xm::beam
