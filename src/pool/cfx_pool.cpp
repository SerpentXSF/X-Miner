#include "pool/cfx_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "core/console.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

std::string strip0x(std::string s) { return s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0 ? s.substr(2) : s; }

bool all_hex(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

}  // namespace

double cfx_difficulty(const octopus::Hash256& boundary) {
  const uint8_t* b = reinterpret_cast<const uint8_t*>(boundary.data());
  double v = 0;
  for (int i = 0; i < 32; ++i) v = v * 256 + b[i];
  return v > 0 ? std::ldexp(1.0, 256) / v : 0;
}

CfxPool::CfxPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

CfxPool::~CfxPool() { stop(); }

void CfxPool::start() { thread_ = std::thread([this] { run(); }); }

void CfxPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string CfxPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool CfxPool::job(octopus::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void CfxPool::submit(const octopus::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "0x%016llx", static_cast<unsigned long long>(s.nonce));
  int id = next_id_++;
  json m = {{"id", id},
            {"method", "mining.submit"},
            {"params", {login_, s.job_id, nonce, "0x" + kawpow::hash_to_hex(s.header)}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void CfxPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.notify") {
    if (!p.is_array() || p.size() < 4 || !p[0].is_string() || !p[1].is_string() || !p[2].is_string() ||
        !p[3].is_string())
      return;
    const std::string hash = strip0x(p[2].get<std::string>()), bound = strip0x(p[3].get<std::string>());
    const std::string height = p[1].get<std::string>();
    if (hash.size() != 64 || !all_hex(hash) || bound.size() > 64 || !all_hex(bound) ||
        height.empty() || height.size() > 19 || !std::all_of(height.begin(), height.end(), ::isdigit))
      return;
    octopus::Job j;
    j.id = p[0].get<std::string>();
    j.height = std::stoull(height);
    j.header = kawpow::hash_from_hex(hash);
    j.boundary = kawpow::hash_from_hex(std::string(64 - bound.size(), '0') + bound);
    j.difficulty = cfx_difficulty(j.boundary);
    if (j.difficulty <= 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
    while (live_jobs_.size() > 8) live_jobs_.pop_front();
    job_ = j;
    have_job_ = true;
    ++version_;
    if (j.difficulty != last_difficulty_) {
      last_difficulty_ = j.difficulty;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %.6g (hashes per share)", j.difficulty);
      info("POOL", buf);
    }
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {
    if (!err.empty() || !(r.is_boolean() && r.get<bool>())) {
      error("POOL", "login failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    authorized_ = true;
    info("POOL", "authorized as " + login_);
    return;
  }
  // Kryptex answers a submit with [true]; conflux-rust's own stratum with true.
  const bool ok = err.empty() && ((r.is_boolean() && r.get<bool>()) ||
                                  (r.is_array() && !r.empty() && r[0].is_boolean() && r[0].get<bool>()));
  Pending pd;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;
    pd = it->second;
    pending_.erase(it);
  }
  on_result_(ok, ok ? std::string() : (err.empty() ? line : err), pd.job_id, pd.tag, pd.work);
}

bool CfxPool::session(const PoolEndpoint& ep) {
  if (!conn_.connect(ep.host, ep.port, 10000, ep.tls)) {
    warn("POOL", conn_.error());
    return false;
  }
  info("POOL", "connected to " + ep.host + ":" + std::to_string(ep.port) + (ep.tls ? " over TLS" : "") + " (" +
                   std::to_string(conn_.latency_ms()) + " ms)");
  record_event("POOL", "connected to " + ep.host);
  reconnect_ = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    authorized_ = false;
    have_job_ = false;
    last_difficulty_ = 0;
  }
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {login_, password_}}}).dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now();
  while (!stop_ && !reconnect_) {
    std::vector<std::string> out;
    {
      std::lock_guard<std::mutex> lk(mu_);
      out.swap(outbox_);
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
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(2)) {
      warn("POOL", "no data from the pool for 2 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = false;
  authorized_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void CfxPool::run() {
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
