// Mining coordinator shared by all algorithms: one engine thread per GPU, the user's pool plus dev-fee routing,
// stats, sensors, API/dashboard and console. An algorithm plugs in through a traits type:
//
//   struct Traits {
//     static constexpr const char* kName;                 // --algorithm name, also the dev-fee key
//     static constexpr bool kCpuRejectIsError;            // a CPU re-check failure means a bad GPU result
//     using Pool; using Job; using Share; using Engine;   // Pool: ResultFn, start/stop/job/submit/connected/endpoint
//     static std::unique_ptr<Engine> make_engine(gpu::Device&, uint64_t seed);
//     static double iterate(Engine&, std::vector<Share>&, std::string& err);  // hashes done, 0 = error
//     static bool is_stale(const std::string& pool_error);
//     static void submit(Pool&, const Share&, int unit, double unit_rate);
//     static void report_rate(Pool&, double total_rate);  // pool-side hashrate report (may do nothing)
//   };
#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "api/api.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/report.hpp"
#include "core/stats.hpp"
#include "gpu/gpu.hpp"
#include "gpu/sensors.hpp"
#include "mining/fee_router.hpp"

namespace xm {

namespace detail {
inline std::atomic<bool>& stop_flag() {
  static std::atomic<bool> stop{false};
  return stop;
}
inline void on_signal(int) { stop_flag() = true; }
}  // namespace detail

template <class T>
int run_mining(const Config& cfg, const std::vector<gpu::DeviceInfo>& devs) {
  using Pool = typename T::Pool;
  using Job = typename T::Job;
  using Share = typename T::Share;
  auto& stop = detail::stop_flag();
  if (cfg.pools.empty() || cfg.wallet.empty()) {
    error("POOL", std::string(T::kName) + " needs --pool and --wallet");
    return 2;
  }
  std::signal(SIGINT, detail::on_signal);
  std::signal(SIGTERM, detail::on_signal);

  Stats stats(devs.size());
  for (size_t i = 0; i < devs.size(); ++i) stats.set_name(static_cast<int>(i), devs[i].name, devs[i].pci_bus);
  std::mutex rs_mu;
  RunState rs;
  rs.algorithm = T::kName;
  FeePlan plan = fee_plan(T::kName);
  if (!plan.enabled) warn("FEE", "no developer wallet in this build: the dev fee is off");

  auto on_result = [&](bool fee) {
    return [&, fee](bool ok, const std::string& err, const std::string& job, int dev) {
      const bool stale = !ok && T::is_stale(err);
      stats.share(dev, ok, stale, fee);
      const std::string who = fee ? "dev fee share" : "share", unit = "unit " + std::to_string(dev);
      if (ok) {
        log_line(Tone::Ok, "LOCK", who + " accepted  " + unit + "  job " + job);
        record_event("LOCK", unit + " · " + who + " accepted");
      } else if (stale) {
        log_line(Tone::Gold, "STAL", who + " stale  " + unit + "  " + err, Tone::Dim);
        record_event("STAL", unit + " · " + err);
      } else {
        log_line(Tone::Red, "MISS", who + " rejected  " + unit + "  " + err, Tone::Red);
        record_event("MISS", unit + " · " + err);
      }
    };
  };
  Pool pool(cfg.pools, cfg.login(), cfg.password, on_result(false));
  pool.start();
  FeeRouter<Pool, Job> router(pool, plan, on_result(true));

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
  std::vector<std::atomic<double>> rates(devs.size());
  for (auto& r : rates) r = 0;
  for (size_t i = 0; i < devs.size(); ++i) {
    workers.emplace_back([&, i] {
      const std::string unit = "unit " + std::to_string(i);
      auto dev = gpu::open(devs[i]);
      const uint64_t seed = 0x9e3779b97f4a7c15ull ^ (uint64_t(i) << 40) ^
                            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
      auto eng = T::make_engine(*dev, seed);
      std::string err;
      if (!eng->init(err)) {
        error("GPU", unit + " " + devs[i].name + ": " + err);
        return;
      }
      uint64_t have = 0, rejects_seen = 0;
      bool job_fee = false;
      double rate = 0;
      while (!stop) {
        Job job;
        uint64_t ver = 0;
        bool fee = false;
        if (!router.job(job, ver, fee)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
        if (ver != have) {
          if (!eng->set_job(job, err)) {
            error("GPU", unit + ": " + err);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
          }
          have = ver;
          job_fee = fee;
        }
        std::vector<Share> shares;
        auto t0 = std::chrono::steady_clock::now();
        double h = T::iterate(*eng, shares, err);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (h == 0) {
          error("GPU", unit + ": " + err);
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        stats.add_hashes(static_cast<int>(i), h, dt);
        rate = rate == 0 ? h / dt : 0.9 * rate + 0.1 * (h / dt);
        rates[i] = rate;
        for (const auto& s : shares) {
          if (s.is_block) log_line(Tone::Gold, "BLCK", "block candidate found by " + unit + "!");
          router.with_pool(job_fee, [&](Pool& p) { T::submit(p, s, static_cast<int>(i), rate); });
        }
        if constexpr (T::kCpuRejectIsError) {
          if (eng->cpu_rejects() != rejects_seen) {
            rejects_seen = eng->cpu_rejects();
            error("GPU", unit + ": GPU result failed CPU verification (" + std::to_string(rejects_seen) +
                             " so far); check clocks and temperatures");
          }
        }
      }
    });
  }

  gpu::Sensors sensors(devs);
  auto last_table = std::chrono::steady_clock::now(), last_tick = last_table;
  while (!stop) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto now = std::chrono::steady_clock::now();
    router.tick(std::chrono::duration<double>(now - last_tick).count());
    last_tick = now;
    double total = 0;
    for (size_t i = 0; i < devs.size(); ++i) {
      auto r = sensors.read(i);
      stats.sensors(static_cast<int>(i), r.power_w, r.temp_c, r.fan_pct, r.core_mhz);
      total += rates[i];
    }
    T::report_rate(pool, total);
    {
      std::lock_guard<std::mutex> lk(rs_mu);
      rs.connected = router.connected();
      rs.pool = router.endpoint();
      rs.fee_percent = router.percent();
      rs.fee_window = router.on_fee();
      rs.fee_next_s = router.seconds_to_window();
    }
    if (now - last_table > std::chrono::seconds(30)) {
      std::lock_guard<std::mutex> lk(rs_mu);
      print_stats_table(stats.snapshot(), rs);
      last_table = now;
    }
  }
  info("STOP", "shutting down");
  for (auto& w : workers) w.join();
  pool.stop();
  if (api) api->stop();
  return 0;
}

}  // namespace xm
