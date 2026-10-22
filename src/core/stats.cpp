#include "core/stats.hpp"

#include <algorithm>
#include <cstdio>

namespace xm {

Stats::Stats(size_t devices) : devs_(devices) {}

void Stats::set_name(int dev, std::string name, int bus) {
  std::lock_guard<std::mutex> lk(mu_);
  devs_.at(dev).name = std::move(name);
  devs_.at(dev).bus = bus;
}

void Stats::add_hashes(int dev, double hashes, double seconds) {
  auto now = Clock::now();
  std::lock_guard<std::mutex> lk(mu_);
  auto& s = devs_.at(dev).samples;
  s.push_back({now, hashes, seconds});
  while (!s.empty() && now - s.front().t > std::chrono::hours(1)) s.pop_front();
}

void Stats::share(int dev, bool accepted, bool stale, bool fee_window) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& d = devs_.at(dev);
  if (accepted) {
    ++d.accepted;
    if (fee_window) ++fee_accepted_;
  } else if (stale) {
    ++d.stale;
  } else {
    ++d.rejected;
  }
}

void Stats::sensors(int dev, double power_w, double temp_c, double fan_pct, double core_mhz) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& d = devs_.at(dev);
  d.power_w = power_w;
  d.temp_c = temp_c;
  d.fan_pct = fan_pct;
  d.core_mhz = core_mhz;
}

double Stats::rate(const std::deque<Sample>& s, Clock::time_point now, double window_s) {
  // Hashes divided by the work time those samples cover, using the samples that ended inside the window.
  // Each sample carries its own duration, so the result does not depend on where the window edge falls.
  double hashes = 0, secs = 0;
  auto cutoff = now - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(window_s));
  for (auto it = s.rbegin(); it != s.rend() && it->t > cutoff; ++it) {
    hashes += it->hashes;
    secs += it->seconds;
  }
  // A device that stopped reporting is idle: stretch the covered time to the window so its rate decays.
  if (!s.empty()) {
    double since_last = std::chrono::duration<double>(now - s.back().t).count();
    if (since_last > 2.0) secs += std::min(since_last, window_s);
  }
  return secs > 0 ? hashes / secs : 0.0;
}

Snapshot Stats::snapshot() const {
  auto now = Clock::now();
  std::lock_guard<std::mutex> lk(mu_);
  Snapshot out;
  out.uptime_s = std::chrono::duration<double>(now - start_).count();
  out.fee_accepted = fee_accepted_;
  for (size_t i = 0; i < devs_.size(); ++i) {
    const auto& d = devs_[i];
    DeviceSnapshot ds;
    ds.id = static_cast<int>(i);
    ds.name = d.name;
    ds.bus = d.bus;
    ds.hr_now = rate(d.samples, now, 10);
    ds.hr_1m = rate(d.samples, now, 60);
    ds.hr_15m = rate(d.samples, now, 900);
    ds.hr_1h = rate(d.samples, now, 3600);
    ds.accepted = d.accepted;
    ds.rejected = d.rejected;
    ds.stale = d.stale;
    ds.power_w = d.power_w;
    ds.temp_c = d.temp_c;
    ds.fan_pct = d.fan_pct;
    ds.core_mhz = d.core_mhz;
    out.hr_now += ds.hr_now;
    out.hr_1m += ds.hr_1m;
    out.hr_15m += ds.hr_15m;
    out.hr_1h += ds.hr_1h;
    out.accepted += ds.accepted;
    out.rejected += ds.rejected;
    out.stale += ds.stale;
    out.devices.push_back(ds);
  }
  return out;
}

std::string format_hashrate(double hps) {
  static const char* units[] = {"H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s"};
  int u = 0;
  while (hps >= 1000.0 && u < 6) {
    hps /= 1000.0;
    ++u;
  }
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f %s", hps, units[u]);
  return buf;
}

std::string format_duration(double seconds) {
  long s = static_cast<long>(seconds);
  char buf[32];
  if (s >= 86400) std::snprintf(buf, sizeof buf, "%ldd %ldh", s / 86400, (s % 86400) / 3600);
  else if (s >= 3600) std::snprintf(buf, sizeof buf, "%ldh %ldm", s / 3600, (s % 3600) / 60);
  else if (s >= 60) std::snprintf(buf, sizeof buf, "%ldm %lds", s / 60, s % 60);
  else std::snprintf(buf, sizeof buf, "%lds", s);
  return buf;
}

}  // namespace xm
