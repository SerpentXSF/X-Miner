// --benchmark for pearlhash: a synthetic job on each selected GPU, every hit re-verified on the CPU.
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <thread>

#include "algo/pearl/pearl_gpu.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/stats.hpp"
#include "gpu/sensors.hpp"

namespace xm::pearl {

namespace {

// A real 76-byte LuckyPool header (see share_test.cpp). Its own nbits makes every hit a non-block share.
const char* kHeaderHex =
    "00004020df2a4132ba5893f1954bf490d9852173e307dd71dec8f68533d27fc0c9ca31276b5380962696afc7352641624e7cb7565a"
    "776af3934f19b2784b5750141f20cacc35296a9cfe0018";

}  // namespace

int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  Job job;
  job.id = "bench";
  for (int i = 0; i < 76; ++i)
    job.header[i] = static_cast<uint8_t>(std::stoul(std::string(kHeaderHex + 2 * i, 2), nullptr, 16));
  // With --share-dump: an easy target (~1 share per 16384^2 x 4096 pass) to exercise the share path.
  // Otherwise a pool-like target (difficulty 2^21) so the run measures GPU throughput.
  job.share_nbits = cfg.share_dump.empty() ? 0x1a07fff8 : 0x1c010000;
  int failures = 0;
  gpu::Sensors sensors(devs);
  for (size_t d = 0; d < devs.size(); ++d) {
    auto dev = gpu::open(devs[d]);
    uint32_t bm = 16384, bn = 32768;
    if (const char* e = std::getenv("XM_PEARL_MN")) std::sscanf(e, "%ux%u", &bm, &bn);  // developer tuning
    Engine eng(*dev, bm, bn, 0x5eed0000 + d);
    std::string err;
    if (!eng.init(err) || !eng.set_job(job, err)) {
      error("GPU", "GPU " + std::to_string(d) + " " + devs[d].name + ": " + err);
      ++failures;
      continue;
    }
    info("BENCH", "GPU " + std::to_string(d) + " " + devs[d].name + ": pearlhash " + std::to_string(bm) + "x" + std::to_string(bn) + "x4096, " +
                      std::to_string(cfg.benchmark_seconds) + " s");
    auto t0 = std::chrono::steady_clock::now(), last = t0;
    double hashes = 0, window = 0, watts = 0;
    int samples = 0;
    size_t shares = 0;
    bool dumped = false;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < cfg.benchmark_seconds) {
      std::vector<Share> found;
      double h = eng.iterate(found, err);
      if (h == 0) {
        error("GPU", err);
        ++failures;
        break;
      }
      hashes += h;
      window += h;
      shares += found.size();
      if (!found.empty() && !cfg.share_dump.empty() && !dumped) {
        std::ofstream(cfg.share_dump + "/share.b64") << found[0].proof_b64;
        std::ofstream(cfg.share_dump + "/header.hex") << kHeaderHex;
        char nb[16];
        std::snprintf(nb, sizeof nb, "%08x", job.share_nbits);
        std::ofstream(cfg.share_dump + "/nbits.hex") << nb;
        dumped = true;
      }
      auto now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 5) {
        auto r = sensors.read(d);
        if (r.power_w > 0) {
          watts += r.power_w;
          ++samples;
        }
        log_line(Tone::Radar, "RATE",
                 format_hashrate(window / dt) + (r.power_w > 0 ? "  " + std::to_string(int(r.power_w)) + " W" : "") +
                     "  shares " + std::to_string(shares) + "  cpu-rejects " + std::to_string(eng.cpu_rejects()));
        window = 0;
        last = now;
      }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double rate = hashes / secs;
    char line[200];
    std::snprintf(line, sizeof line, "%s average over %.0f s; %zu shares, %llu CPU re-verify failures%s", format_hashrate(rate).c_str(),
                  secs, shares, static_cast<unsigned long long>(eng.cpu_rejects()),
                  samples ? (", " + std::to_string(int(watts / samples)) + " W").c_str() : "");
    log_line(eng.cpu_rejects() ? Tone::Red : Tone::Ok, "DONE", line);
    if (eng.cpu_rejects()) ++failures;
  }
  return failures ? 1 : 0;
}

}  // namespace xm::pearl
