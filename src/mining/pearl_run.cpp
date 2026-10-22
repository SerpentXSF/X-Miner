// Pearl mining: one engine thread per GPU, one pool connection, stats, sensors, API/dashboard and console.
#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <mutex>
#include <thread>

#include "algo/pearl/pearl_gpu.hpp"
#include "api/api.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/devfee.hpp"
#include "core/report.hpp"
#include "core/stats.hpp"
#include "gpu/sensors.hpp"
#include "pool/pearl_pool.hpp"

namespace xm {

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

}  // namespace

int run_pearl(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  if (cfg.pools.empty() || cfg.wallet.empty()) {
    error("POOL", "pearlhash needs --pool and --wallet");
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  Stats stats(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) stats.set_name(static_cast<int>(i), devs[i].name);
  std::mutex rs_mu;
  RunState rs;
  rs.algorithm = "pearlhash";
  const FeeTarget* fee = fee_target("pearlhash");
  bool fee_configured = fee && fee->wallet[0] != '\0';
  rs.fee_percent = fee_configured ? fee->percent : 0.0;
  if (!fee_configured) warn("FEE", "no developer wallet in this build: the dev fee is off");

  PearlPool pool(cfg.pools, cfg.login(), cfg.password, [&](bool ok, const std::string& err, const std::string& job, int dev) {
    stats.share(dev, ok, err.find("tale") != std::string::npos || err.find("not found") != std::string::npos, false);
    if (ok) {
      log_line(Tone::Ok, "LOCK", "share accepted  unit " + std::to_string(dev) + "  job " + job);
      record_event("LOCK", "unit " + std::to_string(dev) + " · share accepted");
    } else {
      log_line(Tone::Red, "MISS", "share rejected  unit " + std::to_string(dev) + "  " + err, Tone::Red);
      record_event("MISS", "unit " + std::to_string(dev) + " · " + err);
    }
  });
  pool.start();

  std::unique_ptr<ApiServer> api;
  if (cfg.api_enable) {
    api = std::make_unique<ApiServer>(cfg.api_host, cfg.api_port, [&] {
      std::lock_guard<std::mutex> lk(rs_mu);
      return build_summary(stats.snapshot(), rs);
    });
    std::string err;
    if (api->start(err))
      info("API", "dashboard at http://" + cfg.api_host + ":" + std::to_string(cfg.api_port) + "/");
    else {
      warn("API", err);
      api.reset();
    }
  }

  std::vector<std::thread> workers;
  std::atomic<int> running{0};
  for (size_t i = 0; i < devs.size(); ++i) {
    workers.emplace_back([&, i] {
      auto dev = gpu::open(devs[i]);
      pearl::Engine eng(*dev, 16384, 32768, 0x9e3779b97f4a7c15ull ^ (uint64_t(i) << 32) ^
                                               static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
      std::string err;
      if (!eng.init(err)) {
        error("GPU", "unit " + std::to_string(i) + " " + devs[i].name + ": " + err);
        return;
      }
      ++running;
      uint64_t have = 0;
      double rate = 0;
      while (!g_stop) {
        pearl::Job job;
        uint64_t ver = 0;
        if (!pool.job(job, ver)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
        if (ver != have) {
          if (!eng.set_job(job, err)) {
            error("GPU", "unit " + std::to_string(i) + ": " + err);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
          }
          have = ver;
        }
        std::vector<pearl::Share> shares;
        auto t0 = std::chrono::steady_clock::now();
        double h = eng.iterate(shares, err);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (h == 0) {
          error("GPU", "unit " + std::to_string(i) + ": " + err);
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        stats.add_hashes(static_cast<int>(i), h, dt);
        rate = 0.9 * rate + 0.1 * (h / dt);
        for (const auto& s : shares) {
          if (s.is_block) log_line(Tone::Gold, "BLCK", "block candidate found by unit " + std::to_string(i) + "!");
          pool.submit(s, static_cast<int>(i), rate);
        }
        if (eng.cpu_rejects())
          error("GPU", "unit " + std::to_string(i) + ": GPU result failed CPU verification; check clocks");
      }
    });
  }

  gpu::Sensors sensors(devs);
  auto last_table = std::chrono::steady_clock::now();
  while (!g_stop) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (size_t i = 0; i < devs.size(); ++i) {
      auto r = sensors.read(i);
      stats.sensors(static_cast<int>(i), r.power_w, r.temp_c, r.fan_pct, r.core_mhz);
    }
    {
      std::lock_guard<std::mutex> lk(rs_mu);
      rs.connected = pool.connected();
      rs.pool = pool.endpoint();
    }
    if (std::chrono::steady_clock::now() - last_table > std::chrono::seconds(30)) {
      std::lock_guard<std::mutex> lk(rs_mu);
      print_stats_table(stats.snapshot(), rs);
      last_table = std::chrono::steady_clock::now();
    }
  }
  info("STOP", "shutting down");
  for (auto& w : workers) w.join();
  pool.stop();
  if (api) api->stop();
  return 0;
}

}  // namespace xm
