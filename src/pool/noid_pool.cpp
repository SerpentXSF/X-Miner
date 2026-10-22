#include "pool/noid_pool.hpp"

#include <chrono>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

bool hex_bytes(const std::string& hex, uint8_t* out, size_t n) {
  if (hex.size() != 2 * n) return false;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < n; ++i) {
    int h = nib(hex[2 * i]), l = nib(hex[2 * i + 1]);
    if (h < 0 || l < 0) return false;
    out[i] = static_cast<uint8_t>(h << 4 | l);
  }
  return true;
}

}  // namespace

bool noid_parse_fields(const std::string& hex, noid::Fields& out) {
  uint8_t b[256];
  if (!hex_bytes(hex, b, sizeof b)) return false;
  for (int i = 0; i < 16; ++i) out[i] = noid::u128_from_le_bytes(b + 16 * i);
  return true;
}

bool noid_parse_target(const std::string& hex, std::array<uint8_t, 32>& out) {
  return hex_bytes(hex, out.data(), out.size());
}

bool noid_parse_prefix(const std::string& hex, uint64_t& out) {
  uint8_t b[8];
  if (!hex_bytes(hex, b, sizeof b)) return false;
  out = 0;
  for (int i = 0; i < 8; ++i) out |= uint64_t(b[i]) << (8 * i);
  return true;
}

std::string noid_nonce_hex(const noid::U128& n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 16; ++i) {
    uint8_t b = static_cast<uint8_t>((i < 8 ? n.lo >> (8 * i) : n.hi >> (8 * (i - 8))) & 0xff);
    s += d[b >> 4];
    s += d[b & 15];
  }
  return s;
}

NoidPool::NoidPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

NoidPool::~NoidPool() { stop(); }

void NoidPool::start() { thread_ = std::thread([this] { run(); }); }

void NoidPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string NoidPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool NoidPool::job(noid::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void NoidPool::submit(const noid::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  auto wd = work_domain_.find(s.job_id);
  if (wd == work_domain_.end()) {
    on_result_(false, "job expired before submit", s.job_id, tag);
    return;
  }
  int id = next_id_++;
  json m = {{"id", id}, {"method", "mining.submit"}, {"params", {s.job_id, noid_nonce_hex(s.nonce), wd->second}}};
  pending_[id] = {s.job_id, tag};
  outbox_.push_back(m.dump());
}

void NoidPool::report_rate(double hashrate) {
  std::lock_guard<std::mutex> lk(mu_);
  rate_ = hashrate;
}

void NoidPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  std::string method = m.value("method", "");
  if (method == "mining.notify") {
    const json& ps = m["params"];
    const json& p = ps.is_array() && !ps.empty() ? ps[0] : ps;
    if (!p.is_object()) {
      warn("POOL", "unrecognised mining.notify");
      return;
    }
    noid::Job j;
    j.id = p.value("job_id", "");
    std::string wd = p.value("work_domain_id", "");
    if (p.value("nonce_field_index", 10) != noid::kNonceField || p.value("nonce_bits", 64) != 64 ||
        !noid_parse_fields(p.value("pow_fields_hex", ""), j.fields) ||
        !noid_parse_target(p.value("share_target_hex", ""), j.share_target) ||
        !noid_parse_prefix(p.value("nonce_prefix_hex", ""), j.nonce_prefix) || j.id.empty() || wd.empty()) {
      warn("POOL", "ignoring malformed job " + j.id);
      return;
    }
    noid_parse_target(p.value("block_target_hex", ""), j.block_target);  // optional
    std::lock_guard<std::mutex> lk(mu_);
    // A clean job supersedes the older ones: rplant rejects their shares ("stale share (template superseded)"),
    // so they are dropped here and counted as stale instead of being sent.
    if (p.value("clean", true)) work_domain_.clear();
    while (work_domain_.size() >= 16) work_domain_.erase(work_domain_.begin());
    work_domain_[j.id] = wd;
    job_ = j;
    have_job_ = true;
    ++version_;
    record_event("JOB", "height " + std::to_string(p.value("height", 0)) + " · " + j.id);
    return;
  }
  if (method == "mining.pause") {
    std::lock_guard<std::mutex> lk(mu_);
    have_job_ = false;
    info("POOL", "pool paused mining");
    return;
  }
  if (method == "client.reconnect") {
    info("POOL", "pool asked us to reconnect");
    reconnect_ = true;
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  if (id == 1) {  // subscribe
    const json& r = m.contains("result") ? m["result"] : json();
    std::string proto = r.is_object() ? r.value("protocol", "") : "";
    if (proto != "parano1d-stratum-v1") {
      error("POOL", "pool does not speak parano1d-stratum-v1 (protocol '" + proto + "')");
      reconnect_ = true;
    }
    return;
  }
  bool ok = m.contains("result") && m["result"].is_boolean() && m["result"].get<bool>();
  if (id == 2) {
    if (ok)
      info("POOL", "authorized as " + login_);
    else
      error("POOL", "authorization failed: " + (err.empty() ? std::string("rejected") : err));
    return;
  }
  if (id == 3) {  // miner.stats
    if (!err.empty()) stats_supported_ = false;
    return;
  }
  Pending pd;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;
    pd = it->second;
    pending_.erase(it);
  }
  on_result_(ok, err, pd.job_id, pd.tag);
}

bool NoidPool::session(const PoolEndpoint& ep) {
  if (!conn_.connect(ep.host, ep.port, 10000, ep.tls)) {
    warn("POOL", conn_.error());
    return false;
  }
  info("POOL", "connected to " + ep.host + ":" + std::to_string(ep.port) + (ep.tls ? " over TLS" : "") + " (" +
                   std::to_string(conn_.latency_ms()) + " ms)");
  record_event("POOL", "connected to " + ep.host);
  reconnect_ = false;
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {"X-Miner/" XM_VERSION}}}).dump());
  conn_.send_line(json({{"id", 2}, {"method", "mining.authorize"}, {"params", {login_, password_}}}).dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now(), last_stats = last_rx;
  std::string worker = login_.find('.') != std::string::npos ? login_.substr(login_.find('.') + 1) : "default";
  while (!stop_ && !reconnect_) {
    std::vector<std::string> out;
    {
      std::lock_guard<std::mutex> lk(mu_);
      out.swap(outbox_);
      if (stats_supported_ && rate_ > 0 && std::chrono::steady_clock::now() - last_stats > std::chrono::seconds(60)) {
        out.push_back(json({{"id", 3},
                            {"method", "miner.stats"},
                            {"params", {{"nonce_rate", static_cast<uint64_t>(rate_)}, {"worker", worker}}}})
                          .dump());
        last_stats = std::chrono::steady_clock::now();
      }
    }
    for (const auto& o : out)
      if (!conn_.send_line(o)) break;
    std::string line;
    int r = conn_.read_line(line, 200);
    if (r < 0) {
      warn("POOL", conn_.error());
      break;
    }
    if (r > 0) {
      last_rx = std::chrono::steady_clock::now();
      handle(line);
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) {
      warn("POOL", "no data from the pool for 3 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = false;  // the nonce prefix belongs to this session
  work_domain_.clear();
  for (auto& [id, p] : pending_) on_result_(false, "connection lost", p.job_id, p.tag);
  pending_.clear();
  outbox_.clear();
  return true;
}

void NoidPool::run() {
  int failures = 0;
  while (!stop_) {
    PoolEndpoint ep;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ep = pools_[pool_index_];
    }
    bool had_session = session(ep);
    if (stop_) break;
    failures = had_session ? 0 : failures + 1;
    if (failures >= 3 && pools_.size() > 1) {
      std::lock_guard<std::mutex> lk(mu_);
      pool_index_ = (pool_index_ + 1) % pools_.size();
      failures = 0;
      warn("POOL", "switching to " + pools_[pool_index_].host + ":" + std::to_string(pools_[pool_index_].port));
    }
    for (int i = 0; i < 50 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace xm
