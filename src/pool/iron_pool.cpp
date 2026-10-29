#include "pool/iron_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

bool from_hex(const std::string& s, uint8_t* out, size_t n) {
  if (s.size() != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) {
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

IronPool::IronPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

IronPool::~IronPool() { stop(); }

void IronPool::start() { thread_ = std::thread([this] { run(); }); }

void IronPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string IronPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool IronPool::job(fishhash::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_header_ || !have_target_ || !subscribed_ || waiting_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void IronPool::submit(const fishhash::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !subscribed_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "%016llx", static_cast<unsigned long long>(s.nonce));
  const int id = next_id_++;
  json m = {{"id", id},
            {"method", "mining.submit"},
            {"body", {{"miningRequestId", std::stoll(s.job_id)}, {"randomness", nonce}}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void IronPool::publish_locked() {
  if (!have_header_ || !have_target_) return;
  double t = 0;
  for (uint8_t b : job_.target) t = t * 256 + b;
  job_.difficulty = t > 0 ? std::ldexp(1.0, 256) / t : 0;
  job_.prefix_bits = int(xn_.size()) * 4;
  job_.nonce_prefix = xn_.empty() ? 0 : std::stoull(xn_, nullptr, 16) << (64 - job_.prefix_bits);
  ++version_;
  if (job_.difficulty != last_difficulty_) {
    last_difficulty_ = job_.difficulty;
    char buf[96];
    std::snprintf(buf, sizeof buf, "share difficulty %.6g (hashes per share)", job_.difficulty);
    info("POOL", buf);
  }
}

void IronPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const json& b = m.contains("body") && m["body"].is_object() ? m["body"] : json::object();
  if (method == "mining.subscribed") {
    const std::string xn = b.value("xn", "");
    if (xn.size() > 14 || (!xn.empty() && xn.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)) {
      error("POOL", "unusable extranonce from the pool: " + xn);
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    xn_ = xn;
    subscribed_ = true;
    info("POOL", "subscribed as " + login_);
    publish_locked();
    return;
  }
  if (method == "mining.set_target") {
    fishhash::Bytes32 t;
    if (!from_hex(b.value("target", ""), t.data(), 32)) return;
    std::lock_guard<std::mutex> lk(mu_);
    job_.target = t;
    have_target_ = true;
    publish_locked();
    return;
  }
  if (method == "mining.notify") {
    if (!b.contains("miningRequestId") || !b["miningRequestId"].is_number_integer()) return;
    uint8_t header[180];
    if (!from_hex(b.value("header", ""), header, 180)) return;
    std::lock_guard<std::mutex> lk(mu_);
    job_.id = std::to_string(b["miningRequestId"].get<long long>());
    std::memcpy(job_.header, header, 180);
    have_header_ = true;
    waiting_ = false;
    if (std::find(live_jobs_.begin(), live_jobs_.end(), job_.id) == live_jobs_.end()) live_jobs_.push_back(job_.id);
    while (live_jobs_.size() > 8) live_jobs_.pop_front();
    publish_locked();
    return;
  }
  if (method == "mining.wait_for_work") {
    std::lock_guard<std::mutex> lk(mu_);
    waiting_ = true;
    return;
  }
  if (method == "mining.disconnect") {
    warn("POOL", "the pool disconnected us: " + b.dump());
    reconnect_ = true;
    return;
  }
  if (method == "mining.submitted" || m.contains("error")) {
    int id = -1;
    if (b.contains("id") && b["id"].is_number_integer()) id = b["id"].get<int>();
    else if (m.contains("id") && m["id"].is_number_integer()) id = m["id"].get<int>();
    const bool ok = b.value("result", false);
    std::string msg = b.value("message", "");
    if (m.contains("error")) msg = m["error"].dump();
    Pending pd;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = pending_.find(id);
      if (it == pending_.end()) {
        if (!msg.empty()) warn("POOL", msg);
        return;
      }
      pd = it->second;
      pending_.erase(it);
    }
    on_result_(ok, ok ? std::string() : (msg.empty() ? line : msg), pd.job_id, pd.tag, pd.work);
  }
}

bool IronPool::session(const PoolEndpoint& ep) {
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
    have_header_ = have_target_ = subscribed_ = waiting_ = false;
    last_difficulty_ = 0;
    xn_.clear();
  }
  conn_.send_line(json({{"id", 0},
                        {"method", "mining.subscribe"},
                        {"body", {{"version", 3}, {"publicAddress", login_}, {"agent", "X-Miner/" XM_VERSION}}}})
                      .dump());
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
  have_header_ = subscribed_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void IronPool::run() {
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
