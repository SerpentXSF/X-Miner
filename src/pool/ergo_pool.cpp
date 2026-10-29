#include "pool/ergo_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdio>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

std::string strip0x(std::string s) { return s.rfind("0x", 0) == 0 ? s.substr(2) : s; }

bool hex_ok(const std::string& s, size_t n) {
  if (s.size() != n) return false;
  for (char c : s)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

}  // namespace

ErgoPool::ErgoPool(std::vector<PoolEndpoint> pools, std::string login, std::string password,
                               ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

ErgoPool::~ErgoPool() { stop(); }

void ErgoPool::start() { thread_ = std::thread([this] { run(); }); }

void ErgoPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string ErgoPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool ErgoPool::job(autolykos::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void ErgoPool::submit(const autolykos::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "%016llx", static_cast<unsigned long long>(s.nonce));
  const std::string en2 = std::string(nonce).substr(std::min<size_t>(size_t(s.prefix_bits / 4), 16));
  int id = next_id_++;
  json m = {{"id", id}, {"method", "mining.submit"}, {"params", {login_, s.job_id, en2, "", std::string(nonce)}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void ErgoPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.set_extranonce") {
    if (p.is_array() && !p.empty() && p[0].is_string()) {
      std::lock_guard<std::mutex> lk(mu_);
      extranonce1_ = p[0].get<std::string>();
    }
    return;
  }
  if (method == "mining.notify") {
    if (!p.is_array() || p.size() < 7 || !p[0].is_string() || !p[1].is_number_integer() || !p[2].is_string()) return;
    const std::string msg = strip0x(p[2].get<std::string>());
    if (!hex_ok(msg, 64)) return;
    autolykos::Job j;
    j.id = p[0].get<std::string>();
    j.height = p[1].get<uint32_t>();
    for (int i = 0; i < 32; ++i) j.msg[i] = static_cast<uint8_t>(std::stoul(msg.substr(2 * i, 2), nullptr, 16));
    j.b = autolykos::from_decimal(p[6].is_string() ? p[6].get<std::string>() : p[6].dump());
    if (j.b == autolykos::Bytes32{}) {
      warn("POOL", "job without a usable share target");
      return;
    }
    j.difficulty = autolykos::work_for(j.b);
    std::lock_guard<std::mutex> lk(mu_);
    j.prefix_bits = extranonce1_.size() <= 16 ? int(extranonce1_.size()) * 4 : 0;
    j.nonce_prefix = j.prefix_bits ? std::stoull(extranonce1_, nullptr, 16) << (64 - j.prefix_bits) : 0;
    const bool clean = p.size() > 8 && p[8].is_boolean() ? p[8].get<bool>() : true;
    if (clean) live_jobs_.clear();
    if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
    while (live_jobs_.size() > 16) live_jobs_.pop_front();
    job_ = j;
    have_job_ = true;
    ++version_;
    if (j.difficulty != last_difficulty_) {
      last_difficulty_ = j.difficulty;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %.6g (hashes per share)", j.difficulty);
      info("POOL", buf);
    }
    record_event("JOB", j.id + " height " + std::to_string(j.height));
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // subscribe: [..., extranonce1, extranonce2 size]
    if (r.is_array() && r.size() >= 2 && r[1].is_string()) {
      std::lock_guard<std::mutex> lk(mu_);
      extranonce1_ = r[1].get<std::string>();
    }
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

bool ErgoPool::session(const PoolEndpoint& ep) {
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
    extranonce1_.clear();
  }
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {"X-Miner/" XM_VERSION}}}).dump());
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
  have_job_ = false;
  authorized_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void ErgoPool::run() {
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
