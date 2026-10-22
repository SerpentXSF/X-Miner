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
      long n = 0;
      if (!(v = value()) || !parse_int(*v, 0, 64, n)) return a + " needs a number 0-64";
      c.intensity = static_cast<int>(n);
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
       "  --intensity <0-64>         work size per launch (0 = auto)\n\n"
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
