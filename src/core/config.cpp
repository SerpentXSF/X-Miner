#include "core/config.hpp"

#include <cstdlib>
#include <sstream>

namespace xm {

namespace {

std::vector<std::string> split(const std::string& s, const std::string& seps) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (seps.find(c) != std::string::npos) {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else if (c != ' ') {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool parse_int(const std::string& s, long lo, long hi, long& v) {
  if (s.empty()) return false;
  char* end = nullptr;
  v = std::strtol(s.c_str(), &end, 10);
  return end && *end == '\0' && v >= lo && v <= hi;
}

}  // namespace

std::string Config::login() const {
  if (worker.empty() || wallet.find('.') != std::string::npos) return wallet;
  return wallet + "." + worker;
}

std::optional<std::string> parse_pool_list(const std::string& text, std::vector<PoolEndpoint>& out) {
  for (const auto& item : split(text, "!,")) {
    PoolEndpoint p;
    std::string rest = item;
    auto scheme = rest.find("://");
    if (scheme != std::string::npos) {
      std::string s = rest.substr(0, scheme);
      p.tls = s.find("ssl") != std::string::npos || s.find("tls") != std::string::npos;
      rest = rest.substr(scheme + 3);
    }
    auto colon = rest.rfind(':');
    if (colon == std::string::npos || colon == 0) return "pool '" + item + "' needs host:port";
    long port = 0;
    if (!parse_int(rest.substr(colon + 1), 1, 65535, port)) return "pool '" + item + "' has a bad port";
    p.host = rest.substr(0, colon);
    p.port = static_cast<uint16_t>(port);
    out.push_back(p);
  }
  return std::nullopt;
}

// "v" or "v1,v2,-,v4": integers in [lo, hi], "-" or empty to skip a GPU, optionally "N%" (percent 10-200).
std::optional<std::string> parse_gpu_list(const std::string& text, long lo, long hi, bool allow_pct,
                                          std::vector<std::string>& out) {
  out.clear();
  size_t start = 0;
  while (true) {
    size_t comma = text.find(',', start);
    std::string item = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    long n = 0;
    if (!item.empty() && item != "-") {
      if (allow_pct && item.back() == '%') {
        if (!parse_int(item.substr(0, item.size() - 1), 10, 200, n)) return "bad percentage '" + item + "'";
      } else if (!parse_int(item, lo, hi, n)) {
        return "bad value '" + item + "' (expected " + std::to_string(lo) + " to " + std::to_string(hi) + ")";
      }
    }
    out.push_back(item == "-" ? std::string() : item);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return std::nullopt;
}

namespace {
// Value for GPU i: a single value applies to every GPU; a list gives one per GPU.
std::string pick(const std::vector<std::string>& v, size_t i) {
  if (v.empty()) return {};
  if (v.size() == 1) return v[0];
  return i < v.size() ? v[i] : std::string();
}
std::optional<int> pick_int(const std::vector<std::string>& v, size_t i) {
  std::string s = pick(v, i);
  if (s.empty()) return std::nullopt;
  return std::atoi(s.c_str());
}
}  // namespace

int Config::intensity_for(size_t gpu) const { return pick_int(intensity, gpu).value_or(0); }

gpu::TuneRequest Config::tune_for(size_t gpu) const {
  gpu::TuneRequest r;
  r.core_lock_mhz = pick_int(cclock, gpu);
  r.mem_lock_mhz = pick_int(mclock, gpu);
  r.core_offset_mhz = pick_int(coffset, gpu);
  r.mem_offset_mhz = pick_int(moffset, gpu);
  std::string pl = pick(plimit, gpu);
  if (!pl.empty()) {
    if (pl.back() == '%')
      r.power_pct = std::atof(pl.c_str());
    else
      r.power_w = std::atof(pl.c_str());
  }
  r.fan_pct = pick_int(fan, gpu);
  return r;
}

int Config::off_temp_for(size_t gpu) const { return pick_int(off_temp, gpu).value_or(0); }
int Config::resume_temp_for(size_t gpu) const {
  int off = off_temp_for(gpu);
  return pick_int(resume_temp, gpu).value_or(off > 10 ? off - 10 : 0);
}

std::optional<std::string> parse_args(int argc, char** argv, Config& c) {
  auto need = [&](int& i, const std::string& flag) -> std::optional<std::string> {
    if (i + 1 >= argc) return std::nullopt;
    (void)flag;
    return std::string(argv[++i]);
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    std::optional<std::string> v;
    auto value = [&]() -> std::optional<std::string> { return need(i, a); };
    if (a == "-h" || a == "--help") { c.help = true; }
    else if (a == "--version") { c.version = true; }
    else if (a == "--list-devices") { c.list_devices = true; }
    else if (a == "--list-algorithms") { c.list_algorithms = true; }
    else if (a == "--no-color" || a == "--disable-color") { c.color = false; }
    else if (a == "--disable-api") { c.api_enable = false; }
    else if (a == "--api-enable") { c.api_enable = true; }
    else if (a == "--benchmark") { c.benchmark = true; }
    else if (a == "--selftest") { c.selftest = true; }
    else if (a == "--ui-demo") { c.ui_demo = true; }
    else if (a == "--share-dump") {
      if (!(v = value())) return a + " needs a folder";
      c.share_dump = *v;
    } else if (a == "--pearl-share-test") {
      if (!(v = value())) return a + " needs a folder";
      c.pearl_share_test = *v;
    }
    else if (a == "--algorithm" || a == "-a" || a == "--algo" || a == "--algorithm-gpu") {
      if (!(v = value())) return a + " needs a value";
      c.algorithm = *v;
    } else if (a == "--pool" || a == "-o" || a == "--url") {
      if (!(v = value())) return a + " needs a value";
      if (auto err = parse_pool_list(*v, c.pools)) return err;
    } else if (a == "--wallet" || a == "-w" || a == "--user" || a == "-u") {
      if (!(v = value())) return a + " needs a value";
      c.wallet = *v;
    } else if (a == "--password" || a == "-p" || a == "--pass") {
      if (!(v = value())) return a + " needs a value";
      c.password = *v;
    } else if (a == "--worker") {
      if (!(v = value())) return a + " needs a value";
      c.worker = *v;
    } else if (a == "--gpu-id" || a == "-d" || a == "--devices") {
      if (!(v = value())) return a + " needs a value";
      for (const auto& s : split(*v, ",")) {
        long id = 0;
        if (!parse_int(s, 0, 255, id)) return "bad GPU id '" + s + "'";
        c.gpu_ids.push_back(static_cast<int>(id));
      }
    } else if (a == "--api-port") {
      long port = 0;
      if (!(v = value()) || !parse_int(*v, 1, 65535, port)) return "--api-port needs a port number";
      c.api_port = static_cast<uint16_t>(port);
      c.api_enable = true;
    } else if (a == "--api-host" || a == "--api-rig-ip") {
      if (!(v = value())) return a + " needs a value";
      c.api_host = *v;
    } else if (a == "--log-file") {
      if (!(v = value())) return a + " needs a value";
      c.log_file = *v;
    } else if (a == "--intensity" || a == "--gpu-intensity") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 0, 64, false, c.intensity)) return a + ": " + *e;
    } else if (a == "--gpu-tune-info") {
      c.tune_info = true;
    } else if (a == "--gpu-cclock" || a == "--cclk") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 100, 4000, false, c.cclock)) return a + ": " + *e;
    } else if (a == "--gpu-mclock" || a == "--mclk") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 100, 20000, false, c.mclock)) return a + ": " + *e;
    } else if (a == "--gpu-coffset" || a == "--coff") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, -1000, 1000, false, c.coffset)) return a + ": " + *e;
    } else if (a == "--gpu-moffset" || a == "--moff") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, -3000, 3000, false, c.moffset)) return a + ": " + *e;
    } else if (a == "--gpu-plimit" || a == "--pl") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 10, 1000, true, c.plimit)) return a + ": " + *e;
    } else if (a == "--gpu-fan" || a == "--fan") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 0, 100, false, c.fan)) return a + ": " + *e;
    } else if (a == "--gpu-off-temperature" || a == "--temp-stop") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 30, 120, false, c.off_temp)) return a + ": " + *e;
    } else if (a == "--gpu-resume-temperature" || a == "--temp-start") {
      if (!(v = value())) return a + " needs a value";
      if (auto e = parse_gpu_list(*v, 20, 115, false, c.resume_temp)) return a + ": " + *e;
    } else if (a == "--benchmark-time") {
      long n = 0;
      if (!(v = value()) || !parse_int(*v, 5, 86400, n)) return a + " needs seconds (5-86400)";
      c.benchmark_seconds = static_cast<int>(n);
    } else {
      return "unknown option '" + a + "' (see --help)";
    }
  }
  return std::nullopt;
}

std::string usage() {
  std::ostringstream o;
  o << "Usage: x-miner --algorithm <algo> --pool <host:port> --wallet <address> [options]\n\n"
       "Mining\n"
       "  --algorithm, -a <name>     pearlhash | noid   (--list-algorithms)\n"
       "  --pool, -o <url>           host:port, stratum+tcp://host:port or stratum+ssl://host:port;\n"
       "                             separate failover pools with ! or repeat --pool\n"
       "  --wallet, -w <address>     payout address or pool login\n"
       "  --worker <name>            worker name (sent as wallet.worker)\n"
       "  --password, -p <text>      pool password (default x)\n"
       "  --gpu-id, -d <list>        GPUs to use, e.g. 0,2 (default: all supported; see --list-devices)\n"
       "  --gpu-intensity <list>     work per launch (0 = auto). pearlhash: N x 1024 columns per pass (8-64,\n"
       "                             auto 32); noid: 2^N nonces per launch (18-26, auto 22)\n\n"
       "GPU tuning (applied at start, restored on exit; one value for all GPUs or a comma list, '-' = skip)\n"
       "  --gpu-cclock <MHz>         lock the core clock             (NVIDIA; AMD Linux: Overdrive max core)\n"
       "  --gpu-mclock <MHz>         lock / cap the memory clock     (NVIDIA lock; AMD: Overdrive max memory)\n"
       "  --gpu-coffset <MHz>        core clock offset               (NVIDIA; AMD Windows: max clock offset)\n"
       "  --gpu-moffset <MHz>        memory clock offset             (NVIDIA)\n"
       "  --gpu-plimit <W|N%>        power limit in watts or % of default (AMD Windows: % only, e.g. 70%)\n"
       "  --gpu-fan <N>              fixed fan speed in % (NVIDIA; AMD: flat fan curve at N%)\n"
       "                             NVIDIA and AMD Linux need administrator / root.\n"
       "  --gpu-off-temperature <C>  pause a GPU when it reaches C degrees ...\n"
       "  --gpu-resume-temperature <C>  ... and resume at or below C (default: off temperature - 10)\n"
       "  --gpu-tune-info           show tunable ranges and current values for each GPU, then exit\n\n"
       "Monitoring\n"
       "  --api-port <port>          HTTP API and web dashboard port (default 21550)\n"
       "  --api-host <ip>            bind address (default 127.0.0.1; 0.0.0.0 for LAN access)\n"
       "  --disable-api              no HTTP API or dashboard\n"
       "  --log-file <path>          also write the log to a file\n"
       "  --no-color                 plain text output\n\n"
       "Tools\n"
       "  --list-devices             show GPUs and whether each algorithm supports them\n"
       "  --list-algorithms          show supported algorithms\n"
       "  --benchmark                hash locally without a pool for --benchmark-time seconds (default 60)\n"
       "  --selftest                 check every GPU kernel against the CPU reference and exit\n"
       "  --version, --help\n\n"
       "Dev fee: 1.00% (blake2b 0%). It is time-sliced, shown in the console and API, and set in\n"
       "src/core/devfee.hpp. X-Miner is open source (MIT).\n";
  return o.str();
}

}  // namespace xm
