// --benchmark for cuckaroo29: graphs of random keys; reports graphs per second, the edges left after trimming and the
// 42-cycles found (one per ~42 graphs on average), every proof verified on the CPU.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "algo/c29/c29_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stop.hpp"
#include "gpu/sensors.hpp"

namespace xm::c29 {

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size() && !stop_requested(); ++d) {
    auto dev = gpu::open(devs[d]);
    Engine eng(*dev, 0x43323900 + d);
    std::string err;
    if (!eng.init(err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": cuckaroo29, " + std::to_string(eng.rounds()) +
                      " trimming rounds, " + std::to_string(cfg.benchmark_seconds) + " s");
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    uint64_t graphs = 0, window = 0, cycles = 0, proofs = 0, overflows = 0, remaining = 0;
    double gpu_ms = 0, cpu_ms = 0;
    double watts = 0;
    int samples = 0;
    uint64_t seed = 0x1234567 + d;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds &&
           !stop_requested()) {
      Keys k;
      for (auto& w : k) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        w = seed ^ (seed >> 29);
      }
      std::vector<Proof> found;
      SolveStats st;
      uint64_t prev = 0;
      bool have_prev = false;
      if (!eng.solve_pipelined(k, graphs, found, prev, have_prev, st, err)) {  // the CPU search overlaps the next graph
        error("GPU", err);
        ++failures;
        break;
      }
      ++graphs;
      ++window;
      cycles += st.cycles;
      proofs += found.size();
      overflows += st.overflow;
      remaining += st.remaining;
      gpu_ms += st.gpu_ms;
      cpu_ms += st.cpu_ms;
      const auto now = std::chrono::steady_clock::now();
      const double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 5) {
        const auto r = sensors.read(d);
        if (r.power_w > 0) {
          watts += r.power_w;
          ++samples;
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%.3f g/s  %d W  edges left %.0f  cycles %llu", window / dt, int(r.power_w),
                      double(remaining) / double(graphs), static_cast<unsigned long long>(cycles));
        log_line(Tone::Radar, "RATE", buf);
        window = 0;
        last = now;
      }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char line[320];
    std::snprintf(line, sizeof line,
                  "%.3f g/s over %.0f s (%llu graphs); %llu cycles (expected ~%.1f), %llu verified, %llu overflows, "
                  "%.0f edges left per graph, GPU %.1f ms + CPU %.1f ms per graph%s",
                  graphs / secs, secs, static_cast<unsigned long long>(graphs), static_cast<unsigned long long>(cycles),
                  graphs / 42.0, static_cast<unsigned long long>(proofs), static_cast<unsigned long long>(overflows),
                  graphs ? double(remaining) / double(graphs) : 0.0, graphs ? gpu_ms / graphs : 0.0,
                  graphs ? cpu_ms / graphs : 0.0, samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    const bool ok = proofs == cycles && eng.cpu_rejects() == 0;
    log_line(ok ? Tone::Ok : Tone::Red, "DONE", line);
    if (!ok) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::c29
