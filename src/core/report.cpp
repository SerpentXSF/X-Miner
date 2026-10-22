#include "core/report.hpp"

#include <cstdio>

#include "core/console.hpp"
#include "core/version.hpp"

namespace xm {

nlohmann::json build_summary(const Snapshot& s, const RunState& r) {
  using nlohmann::json;
  json gpu_hr = json::object();
  json devices = json::array();
  for (const auto& d : s.devices) {
    gpu_hr["gpu" + std::to_string(d.id)] = d.hr_1m;
    devices.push_back({{"id", d.id},
                       {"name", d.name},
                       {"hr", d.hr_now},
                       {"hr_1m", d.hr_1m},
                       {"power", d.power_w},
                       {"temp", d.temp_c},
                       {"fan", d.fan_pct},
                       {"core_mhz", d.core_mhz},
                       {"accepted", d.accepted},
                       {"rejected", d.rejected},
                       {"stale", d.stale}});
  }
  gpu_hr["total"] = s.hr_1m;
  json log = json::array();
  for (const auto& e : recent_events()) log.push_back({{"t", e.t}, {"tag", e.tag}, {"msg", e.msg}});

  json out;
  out["rig_name"] = "X-Miner";
  out["miner_version"] = XM_VERSION;
  out["mining_time"] = static_cast<long long>(s.uptime_s);
  out["total_gpu_workers"] = s.devices.size();
  out["algorithms"] = json::array({{{"name", r.algorithm},
                                    {"pool", {{"pool", r.pool}, {"connected", r.connected}}},
                                    {"hashrate", {{"1min", s.hr_1m}, {"15min", s.hr_15m}, {"1hr", s.hr_1h}, {"gpu", gpu_hr}}},
                                    {"shares",
                                     {{"total", s.accepted + s.rejected + s.stale},
                                      {"accepted", s.accepted},
                                      {"rejected", s.rejected + s.stale}}}}});
  out["x"] = {{"version", XM_VERSION},
              {"vendor", XM_VENDOR},
              {"algorithm", r.algorithm},
              {"pool", r.pool},
              {"connected", r.connected},
              {"uptime", s.uptime_s},
              {"fee", {{"percent", r.fee_percent}, {"in_window", r.fee_window}, {"next_window_s", r.fee_next_s},
                       {"shares_accepted", s.fee_accepted}}},
              {"hr", {{"now", s.hr_now}, {"1m", s.hr_1m}, {"15m", s.hr_15m}, {"1h", s.hr_1h}}},
              {"shares", {{"accepted", s.accepted}, {"rejected", s.rejected}, {"stale", s.stale}}},
              {"devices", devices},
              {"log", log}};
  return out;
}

namespace {

std::string pad(const std::string& s, size_t w, bool right = false) {
  if (s.size() >= w) return s;
  return right ? std::string(w - s.size(), ' ') + s : s + std::string(w - s.size(), ' ');
}

std::string num(double v, const char* fmt) {
  if (v < 0) return "-";
  char b[32];
  std::snprintf(b, sizeof b, fmt, v);
  return b;
}

}  // namespace

void print_stats_table(const Snapshot& s, const RunState& r) {
  auto& c = Console::get();
  c.line({{Tone::Hud, "  ┌─ UNIT ─┬─ DEVICE ─────────────────┬─── HASHRATE ──┬─ POWER ─┬─ EFF ────────┬ TEMP ┬ FAN ┬ A/R ──────┐"}});
  double watts = 0;
  for (const auto& d : s.devices) {
    if (d.power_w > 0) watts += d.power_w;
    std::string eff = d.power_w > 0 ? format_hashrate(d.hr_now / d.power_w) : "-";
    if (eff != "-") eff.replace(eff.size() - 2, 2, "/W");
    c.line({{Tone::Hud, "  │ "},
            {Tone::Radar, pad(std::to_string(d.id), 6, true) + " "},
            {Tone::Hud, "│ "},
            {Tone::Plain, pad(d.name.substr(0, 24), 24) + " "},
            {Tone::Hud, "│ "},
            {Tone::Radar, pad(format_hashrate(d.hr_now), 13, true) + " ", true},
            {Tone::Hud, "│ "},
            {Tone::Plain, pad(num(d.power_w, "%.0f W"), 7) + " "},
            {Tone::Hud, "│ "},
            {Tone::Gold, pad(eff, 12) + " "},
            {Tone::Hud, "│ "},
            {Tone::Plain, pad(num(d.temp_c, "%.0fC"), 4) + " "},
            {Tone::Hud, "│ "},
            {Tone::Plain, pad(num(d.fan_pct, "%.0f%%"), 3) + " "},
            {Tone::Hud, "│ "},
            {Tone::Ok, std::to_string(d.accepted)},
            {Tone::Dim, "/"},
            {Tone::Red, pad(std::to_string(d.rejected + d.stale), 9 - std::to_string(d.accepted).size() - 1)},
            {Tone::Hud, "│"}});
  }
  c.line({{Tone::Hud, "  ├────────┴──────────────────────────┼───────────────┼─────────┼──────────────┴──────┴─────┴───────────┤"}});
  char fee[96];
  if (r.fee_percent <= 0)
    std::snprintf(fee, sizeof fee, "no dev fee");
  else if (r.fee_window)
    std::snprintf(fee, sizeof fee, "dev fee %.2f%% · mining for the developer now", r.fee_percent);
  else
    std::snprintf(fee, sizeof fee, "dev fee %.2f%% · next window in %s", r.fee_percent,
                  format_duration(r.fee_next_s).c_str());
  c.line({{Tone::Hud, "  │ "},
          {Tone::Plain, pad("TOTAL", 34), true},
          {Tone::Hud, "│ "},
          {Tone::Radar, pad(format_hashrate(s.hr_now), 13, true) + " ", true},
          {Tone::Hud, "│ "},
          {Tone::Plain, pad(watts > 0 ? num(watts, "%.0f W") : "-", 7) + " "},
          {Tone::Hud, "│ "},
          {Tone::Dim, pad("1m " + format_hashrate(s.hr_1m) + " · up " + format_duration(s.uptime_s), 37)},
          {Tone::Hud, "│"}});
  c.line({{Tone::Hud, "  └───────────────────────────────────┴───────────────┴─────────┴───────────────────────────────────────┘"}});
  c.line({{Tone::Dim, "  " + std::string(fee) + " · shares " + std::to_string(s.accepted) + " accepted / " +
                          std::to_string(s.rejected + s.stale) + " rejected"}});
}

}  // namespace xm
