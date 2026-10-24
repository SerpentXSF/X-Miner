// Mining coordinator shared by all algorithms: one engine thread per GPU, the user's pool plus dev-fee routing,
// stats, sensors, API/dashboard and console. An algorithm plugs in through a traits type:
//
//   struct Traits {
//     static constexpr const char* kName;                 // --algorithm name, also the dev-fee key
//     static constexpr bool kCpuRejectIsError;            // a CPU re-check failure means a bad GPU result
//     using Pool; using Job; using Share; using Engine;   // Pool: ResultFn, start/stop/job/submit/connected/endpoint
//     static std::unique_ptr<Engine> make_engine(gpu::Device&, uint64_t seed, int intensity);  // 0 = auto
//     static double iterate(Engine&, std::vector<Share>&, std::string& err);  // hashes done, 0 = error
//     static bool is_stale(const std::string& pool_error);
//     static void submit(Pool&, const Share&, int unit, double unit_rate);
//     static void report_rate(Pool&, double total_rate);  // pool-side hashrate report (may do nothing)
//   };
#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
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
    return [&, fee](bool ok, const std::string& err, const std::string& job, int dev, double work) {
      const bool stale = !ok && T::is_stale(err);
      stats.share(dev, ok, stale, fee, work);
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

  // Watchdog: exit (so the launcher's restart loop recovers the GPU) after kMaxGpuErrors consecutive GPU errors, when
  // a GPU makes no progress for kStallSeconds while there is work, or when no GPU could be started.
  constexpr int kMaxGpuErrors = 10;
  constexpr int kStallSeconds = 120;
  std::atomic<int> exit_code{0}, init_failed{0};
  auto steady_s = [] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  std::vector<std::atomic<double>> last_progress(devs.size());  // steady seconds; 0 = not started / waiting
  for (auto& t : last_progress) t = 0;
  auto give_up = [&](int code, const std::string& why) {
    error("WDOG", why + "; exiting so the launcher can restart X-Miner");
    exit_code = code;
    stop = true;
  };

  // Thermal pause (--gpu-off-temperature): a GPU at or above its off temperature stops launching work until it
  // cools to its resume temperature. The sensor loop below sets the flags.
  std::vector<std::atomic<bool>> hot(devs.size());
  for (auto& h : hot) h = false;

  std::vector<std::thread> workers;
  std::vector<std::atomic<double>> rates(devs.size());
  for (auto& r : rates) r = 0;
  for (size_t i = 0; i < devs.size(); ++i) {
    workers.emplace_back([&, i] {
      const std::string unit = "unit " + std::to_string(i);
      auto dev = gpu::open(devs[i]);
      const uint64_t seed = 0x9e3779b97f4a7c15ull ^ (uint64_t(i) << 40) ^
                            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
      auto eng = T::make_engine(*dev, seed, cfg.intensity_for(i));
      std::string err;
      if (!eng->init(err)) {
        error("GPU", unit + " " + devs[i].name + ": " + err);
        ++init_failed;
        return;
      }
      int errors = 0;  // consecutive
      auto gpu_error = [&](const std::string& e) {
        error("GPU", unit + ": " + e);
        if (++errors >= kMaxGpuErrors) give_up(3, unit + ": " + std::to_string(errors) + " GPU errors in a row");
        std::this_thread::sleep_for(std::chrono::seconds(1));
      };
      uint64_t have = 0, rejects_seen = 0, wd_passes = 0;
      bool job_fee = false;
      double rate = 0;
      while (!stop) {
        if (hot[i]) {
          last_progress[i] = 0;  // paused, not stalled
          rates[i] = rate = 0;
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
          continue;
        }
        Job job;
        uint64_t ver = 0;
        bool fee = false;
        if (!router.job(job, ver, fee)) {
          last_progress[i] = 0;  // waiting for work is not a stall
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
        if (last_progress[i] == 0) last_progress[i] = steady_s();
        if (ver != have) {
          if (!eng->set_job(job, err)) {
            gpu_error(err);
            continue;
          }
          have = ver;
          job_fee = fee;
        }
        std::vector<Share> shares;
        auto t0 = std::chrono::steady_clock::now();
        double h = T::iterate(*eng, shares, err);
        static const char* wd_test = std::getenv("XM_WDOG_TEST");  // developer: "errors" or "stall" after 3 passes
        if (wd_test && ++wd_passes > 3) {
          if (wd_test[0] == 'e') h = 0, err = "simulated GPU error (XM_WDOG_TEST)";
          if (wd_test[0] == 's') std::this_thread::sleep_for(std::chrono::hours(1));
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (h == 0) {
          gpu_error(err);
          continue;
        }
        errors = 0;
        last_progress[i] = steady_s();
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
      const int off = cfg.off_temp_for(i), resume = cfg.resume_temp_for(i);
      const std::string unit = "unit " + std::to_string(i);
      if (off > 0 && r.temp_c >= off && !hot[i]) {
        hot[i] = true;
        warn("TEMP", unit + " at " + std::to_string(static_cast<int>(r.temp_c)) + " C: paused until " +
                         std::to_string(resume) + " C");
        record_event("TEMP", unit + " paused at " + std::to_string(static_cast<int>(r.temp_c)) + " C");
      } else if (hot[i] && r.temp_c >= 0 && r.temp_c <= resume) {
        hot[i] = false;
        info("TEMP", unit + " cooled to " + std::to_string(static_cast<int>(r.temp_c)) + " C: mining again");
        record_event("TEMP", unit + " resumed");
      }
    }
    T::report_rate(pool, total);
    if (init_failed == static_cast<int>(devs.size())) give_up(4, "no GPU could be started");
    for (size_t i = 0; i < devs.size(); ++i) {
      double t = last_progress[i];
      if (t > 0 && steady_s() - t > kStallSeconds) {
        // A kernel that never returns cannot be joined: restore GPU tuning (main's quick-exit handler) and exit now.
        error("WDOG", "unit " + std::to_string(i) + ": no GPU progress for " + std::to_string(kStallSeconds) +
                          " s; exiting so the launcher can restart X-Miner");
        std::quick_exit(3);
      }
    }
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
  return exit_code;
}

}  // namespace xm
