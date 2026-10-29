#include "pool/xelis_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

std::string strip0x(std::string s) { return s.rfind("0x", 0) == 0 ? s.substr(2) : s; }

// Up to `max` bytes of hex into out (zero-padded); false if not hex or too long.
bool hex_to(const std::string& s, uint8_t* out, size_t max) {
  if (s.size() % 2 || s.size() / 2 > max) return false;
  std::memset(out, 0, max);
  for (size_t i = 0; i < s.size() / 2; ++i) {
    unsigned v = 0;
    for (int k = 0; k < 2; ++k) {
      const char c = s[2 * i + k];
      if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
      v = v * 16 + unsigned(std::isdigit(static_cast<unsigned char>(c)) ? c - '0' : std::tolower(c) - 'a' + 10);
    }
    out[i] = static_cast<uint8_t>(v);
  }
  return true;
}

}  // namespace

XelisPool::XelisPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

XelisPool::~XelisPool() { stop(); }

void XelisPool::start() { thread_ = std::thread([this] { run(); }); }

void XelisPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string XelisPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool XelisPool::job(xelis::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_ || !subscribed_ || difficulty_ == 0) return false;
  out = job_;
  out.difficulty = difficulty_;
  std::memcpy(out.work.bytes + 48, extra_nonce_, 32);
  std::memcpy(out.work.bytes + 80, public_key_, 32);
  version = version_.load();
  return true;
}

void XelisPool::submit(const xelis::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "%016llx", static_cast<unsigned long long>(s.nonce));
  const int id = next_id_++;
  json m = {{"id", id}, {"method", "mining.submit"}, {"params", {login_, s.job_id, nonce}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void XelisPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.contains("method") && m["method"].is_string() ? m["method"].get<std::string>() : "";
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.set_difficulty") {
    if (p.is_array() && !p.empty() && p[0].is_number() && p[0].get<double>() >= 1 && p[0].get<double>() < 1e19) {
      std::lock_guard<std::mutex> lk(mu_);
      difficulty_ = static_cast<uint64_t>(p[0].get<double>());
      ++version_;
    }
    return;
  }
  if (method == "mining.set_extranonce") {
    if (p.is_array() && !p.empty() && p[0].is_string()) {
      std::lock_guard<std::mutex> lk(mu_);
      if (!hex_to(strip0x(p[0].get<std::string>()), extra_nonce_, 32)) warn("POOL", "unusable extra nonce from the pool");
      ++version_;
    }
    return;
  }
  if (method == "mining.notify") {
    if (!p.is_array() || p.size() < 4 || !p[0].is_string() || !p[1].is_string() || !p[2].is_string() || !p[3].is_string())
      return;
    const std::string algo = p[3].get<std::string>();
    if (algo != "xel/v3") {
      warn("POOL", "the pool asks for " + algo + "; X-Miner mines xel/v3");
      return;
    }
    const std::string ts = strip0x(p[1].get<std::string>());
    xelis::Job j;
    if (ts.empty() || ts.size() > 16 || !std::all_of(ts.begin(), ts.end(), ::isxdigit) ||
        !hex_to(strip0x(p[2].get<std::string>()), j.work.bytes, 32))
      return;
    const uint64_t t = std::stoull(ts, nullptr, 16);
    for (int i = 0; i < 8; ++i) j.work.bytes[32 + i] = static_cast<uint8_t>(t >> (56 - 8 * i));
    j.id = p[0].get<std::string>();
    std::lock_guard<std::mutex> lk(mu_);
    const bool clean = p.size() > 4 && p[4].is_boolean() && p[4].get<bool>();
    if (clean) live_jobs_.clear();
    if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
    while (live_jobs_.size() > 8) live_jobs_.pop_front();
    job_ = j;
    have_job_ = true;
    ++version_;
    if (difficulty_ != last_difficulty_) {
      last_difficulty_ = difficulty_;
      info("POOL", "share difficulty " + std::to_string(difficulty_) + " (hashes per share)");
    }
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // subscribe: [session, extra nonce, size, public key]
    if (!err.empty() || !r.is_array() || r.size() < 4 || !r[1].is_string() || !r[3].is_string()) {
      error("POOL", "subscribe failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!hex_to(strip0x(r[1].get<std::string>()), extra_nonce_, 32) ||
        !hex_to(strip0x(r[3].get<std::string>()), public_key_, 32)) {
      error("POOL", "unusable subscribe reply: " + line);
      reconnect_ = true;
      return;
    }
    subscribed_ = true;
    return;
  }
  if (id == 2) {
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
  const bool ok = err.empty() && r.is_boolean() && r.get<bool>();
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

bool XelisPool::session(const PoolEndpoint& ep) {
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
    authorized_ = subscribed_ = have_job_ = false;
    difficulty_ = last_difficulty_ = 0;
  }
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {"X-Miner/" XM_VERSION, {"xel/v3"}}}}).dump());
  conn_.send_line(json({{"id", 2}, {"method", "mining.authorize"}, {"params", {login_, password_}}}).dump());
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
      try {
        handle(line);
      } catch (const std::exception& e) {  // malformed pool data (types, numbers): drop the message
        warn("POOL", std::string("ignored a malformed message from the pool (") + e.what() + ")");
      }
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(5)) {
      warn("POOL", "no data from the pool for 5 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = authorized_ = subscribed_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void XelisPool::run() {
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
