#include <algorithm>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "algo/registry.hpp"
#include "gpu/tuning.hpp"
#include "algo/selftest.hpp"
#include "api/api.hpp"
#include "core/devfee.hpp"
#include "core/report.hpp"
#include "core/stats.hpp"
#include "core/banner.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/version.hpp"
#include "gpu/gpu.hpp"
#include "gpu/kernels.hpp"
#include "gpu/sensors.hpp"

using namespace xm;

namespace xm {
int run_pearl(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs);
int run_noid(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs);
}
namespace xm::noid {
int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs);
}
namespace xm::pearl {
int share_test(const std::string& dir);
int benchmark(const xm::Config& cfg, const std::vector<gpu::DeviceInfo>& devs);
}

namespace {

std::string gb(size_t bytes) {
  char b[16];
  std::snprintf(b, sizeof b, "%.0f GB", bytes / 1073741824.0);
  return b;
}

int list_devices() {
  auto& c = Console::get();
  print_section("SYSTEM CHECK");
  std::string amd = gpu::amd_runtime_name(), nv = gpu::nvidia_runtime_name();
  c.line({{Tone::Radar, "  [AMD   ] "}, {amd.empty() ? Tone::Dim : Tone::Plain, amd.empty() ? "HIP runtime not found" : amd}});
  c.line({{Tone::Radar, "  [NVIDIA] "}, {nv.empty() ? Tone::Dim : Tone::Plain, nv.empty() ? "CUDA driver not found" : nv}});
  auto devs = gpu::all_devices();
  if (devs.empty()) {
    c.line({{Tone::Red, "  no supported GPU found"}});
    return 1;
  }
  for (size_t i = 0; i < devs.size(); ++i) {
    const auto& d = devs[i];
    char id[32];
    std::snprintf(id, sizeof id, "  [GPU %zu ] ", i);
    std::string algos;
    for (const auto& a : algorithms())
      if (supports(a, d)) algos += std::string(algos.empty() ? "" : ", ") + a.name;
    c.line({{Tone::Radar, id},
            {Tone::Plain, d.name + "  "},
            {Tone::Dim, d.arch + " · " + gb(d.memory) + " · bus " + std::to_string(d.pci_bus) + " · " +
                            std::to_string(d.compute_units) +
                            (d.vendor == gpu::Vendor::Nvidia ? " SM  " : (d.arch >= "gfx10" ? " WGP  " : " CU  "))},
            {algos.empty() ? Tone::Dim : Tone::Ok, algos.empty() ? "no kernel yet" : algos}});
  }
  gpu::Sensors sensors(devs);
  for (size_t i = 0; i < devs.size(); ++i) {
    auto r = sensors.read(i);
    char b[160];
    std::snprintf(b, sizeof b, "power %.0f W · temp %.0f C · fan %.0f%% · core %.0f MHz", r.power_w, r.temp_c,
                  r.fan_pct, r.core_mhz);
    c.line({{Tone::Dim, "  [GPU " + std::to_string(i) + " ] sensors: "}, {Tone::Plain, b}});
  }
  return 0;
}

int list_algorithms() {
  auto& c = Console::get();
  print_section("ALGORITHMS");
  for (const auto& a : algorithms()) {
    std::string archs;
    for (const auto& s : gpu::kernel_archs(a.kernel)) archs += (archs.empty() ? "" : " ") + s;
    c.line({{Tone::Radar, "  " + std::string(a.name)},
            {Tone::Dim, "  " + std::string(a.coin) + "  "},
            {a.implemented ? Tone::Ok : Tone::Gold, a.implemented ? "ready: " + archs : "in development"}});
  }
  return 0;
}

// Developer tool: simulated devices and shares, to exercise the console table and the web dashboard.
// Everything it shows is labelled DEMO; it never connects to a pool.
int ui_demo(const Config& cfg) {
  Stats stats(2);
  stats.set_name(0, "DEMO RX 9070 XT");
  stats.set_name(1, "DEMO RTX 3060");
  FeeSchedule fee(kFeePercent, 600);  // short demo cycle so the fee window shows up quickly
  RunState rs;
  rs.algorithm = "pearlhash (DEMO)";
  rs.pool = "demo.invalid:0";
  rs.fee_percent = kFeePercent;
  std::mutex mu;
  std::unique_ptr<ApiServer> api;
  if (cfg.api_enable) {
    api = std::make_unique<ApiServer>(cfg.api_host, cfg.api_port, [&] {
      std::lock_guard<std::mutex> lk(mu);
      return build_summary(stats.snapshot(), rs);
    });
    std::string err;
    if (!api->start(err)) {
      error("API", err);
      api.reset();
    } else {
      info("API", "dashboard at http://" + cfg.api_host + ":" + std::to_string(cfg.api_port) + "/");
    }
  }
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(1.0, 0.01);
  info("DEMO", "simulated devices; nothing here is real mining");
  record_event("POOL", "DEMO connected to demo.invalid");
  for (int sec = 1; sec <= cfg.benchmark_seconds; ++sec) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::lock_guard<std::mutex> lk(mu);
    rs.connected = true;
    stats.add_hashes(0, 100.4e12 * noise(rng), 1.0);
    stats.add_hashes(1, 47.0e12 * noise(rng), 1.0);
    stats.sensors(0, 243 + 3 * noise(rng), 50, 55, 1883);
    stats.sensors(1, 119, 61, 66, 1702);
    bool was = rs.fee_window;
    rs.fee_window = fee.advance(1.0);
    rs.fee_next_s = fee.seconds_to_window();
    if (rs.fee_window != was) {
      log_line(Tone::Gold, "FEE", rs.fee_window ? "dev fee window started (DEMO)" : "dev fee window ended (DEMO)");
      record_event("FEE", rs.fee_window ? "dev window started" : "dev window ended");
    }
    if (sec % 7 == 0) {
      int dev = (sec / 7) % 2;
      bool ok = (sec % 91) != 0;
      stats.share(dev, ok, false, rs.fee_window);
      log_line(ok ? Tone::Ok : Tone::Red, ok ? "LOCK" : "MISS",
               std::string(ok ? "share accepted" : "share rejected") + "  unit " + std::to_string(dev) + " (DEMO)");
      record_event(ok ? "LOCK" : "MISS", std::string(ok ? "share accepted" : "share rejected") + " · unit " +
                                             std::to_string(dev));
    }
    if (sec % 30 == 0) record_event("JOB", "new job (DEMO)");
    if (sec % 20 == 0) print_stats_table(stats.snapshot(), rs);
  }
  return 0;
}

}  // namespace

namespace {
xm::gpu::Tuner* g_tuner = nullptr;
#ifdef _WIN32
// Closing the console window ends the process within a few seconds: restore GPU tuning first.
BOOL WINAPI on_console_event(DWORD type) {
  if ((type == CTRL_CLOSE_EVENT || type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) && g_tuner)
    g_tuner->restore();
  return FALSE;  // let the default handling (and Ctrl+C's signal path) continue
}
#endif
}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (auto err = parse_args(argc, argv, cfg)) {
    std::fprintf(stderr, "x-miner: %s\n", err->c_str());
    return 2;
  }
  Console::get().set_color(cfg.color && Console::get().color());
  if (!cfg.log_file.empty()) Console::get().set_log_file(cfg.log_file);
  if (cfg.help) {
    std::fputs(usage().c_str(), stdout);
    return 0;
  }
  if (cfg.version) {
    std::printf("%s %s (%s)\n", XM_NAME, XM_VERSION, XM_VENDOR);
    return 0;
  }
  print_banner(cfg.algorithm);
  if (cfg.list_devices) return list_devices();
  if (cfg.list_algorithms) return list_algorithms();
  if (cfg.ui_demo) return ui_demo(cfg);
  if (!cfg.pearl_share_test.empty()) return pearl::share_test(cfg.pearl_share_test);
  if (cfg.selftest) {
    print_section("SELF TEST");
    auto devs = gpu::all_devices();
    if (devs.empty()) {
      error("GPU", "no supported GPU found");
      return 1;
    }
    int failed = run_selftest(devs);
    return failed ? 1 : 0;
  }

  const AlgorithmInfo* algo = find_algorithm(cfg.algorithm);
  if (!algo) {
    error("ALGO", cfg.algorithm.empty() ? "no --algorithm given (see --list-algorithms)"
                                        : "unknown algorithm '" + cfg.algorithm + "' (see --list-algorithms)");
    return 2;
  }
  auto all = gpu::all_devices();
  std::vector<gpu::DeviceInfo> devs;
  for (size_t i = 0; i < all.size(); ++i)
    if (cfg.gpu_ids.empty() || std::find(cfg.gpu_ids.begin(), cfg.gpu_ids.end(), int(i)) != cfg.gpu_ids.end())
      devs.push_back(all[i]);
  if (devs.empty()) {
    error("GPU", "no GPU selected (see --list-devices)");
    return 1;
  }
  gpu::Tuner tuner(devs);
  g_tuner = &tuner;
#ifdef _WIN32
  SetConsoleCtrlHandler(on_console_event, TRUE);
#endif
  for (size_t i = 0; i < devs.size(); ++i) tuner.apply(i, cfg.tune_for(i));
  struct Restore {
    gpu::Tuner& t;
    ~Restore() {
      t.restore();
      g_tuner = nullptr;
    }
  } restore{tuner};
  if (std::string(algo->name) == "noid") {
    if (cfg.benchmark) {
      print_section("BENCHMARK");
      return noid::benchmark(cfg, devs);
    }
    print_section("MISSION START");
    return run_noid(cfg, devs);
  }
  if (std::string(algo->name) == "pearlhash") {
    if (cfg.benchmark) {
      print_section("BENCHMARK");
      return pearl::benchmark(cfg, devs);
    }
    print_section("MISSION START");
    return run_pearl(cfg, devs);
  }
  if (!algo->implemented) {
    error("ALGO", std::string(algo->name) + " is still in development in this build");
    return 3;
  }
  return 0;
}
