#include "pool/nexa_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

bool hex_vec(const std::string& s, std::vector<uint8_t>& out, size_t max = 1 << 20) {
  if (s.size() % 2 || s.size() / 2 > max) return false;
  out.clear();
  for (size_t i = 0; i < s.size(); i += 2) {
    unsigned v = 0;
    for (int k = 0; k < 2; ++k) {
      const char c = s[i + k];
      if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
      v = v * 16 + unsigned(std::isdigit(static_cast<unsigned char>(c)) ? c - '0' : std::tolower(c) - 'a' + 10);
    }
    out.push_back(static_cast<uint8_t>(v));
  }
  return true;
}

bool hex_u32(const std::string& s, uint32_t& v) {
  if (s.empty() || s.size() > 8 || !std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); }))
    return false;
  v = static_cast<uint32_t>(std::stoul(s, nullptr, 16));
  return true;
}

}  // namespace

bool nexa_parse_notify(const std::string& params_json, nexa::Job& out) {
  json p = json::parse(params_json, nullptr, false);
  if (p.is_discarded() || !p.is_array() || p.size() < 4) return false;
  if (!p[0].is_string() || !p[1].is_string() || !p[2].is_number_unsigned() || !p[3].is_string()) return false;
  nexa::Job j;
  j.id = p[0].get<std::string>();
  std::vector<uint8_t> hc;
  if (j.id.empty() || !hex_vec(p[1].get<std::string>(), hc) || hc.size() != 32 || !hex_u32(p[3].get<std::string>(), j.nbits))
    return false;
  std::copy(hc.begin(), hc.end(), j.commitment.begin());
  j.height = p[2].get<uint64_t>();
  out = j;
  return true;
}

NexaPool::NexaPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

NexaPool::~NexaPool() { stop(); }

void NexaPool::start() { thread_ = std::thread([this] { run(); }); }

void NexaPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string NexaPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool NexaPool::job(nexa::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_ || !subscribed_ || difficulty_ <= 0) return false;
  out = job_;
  out.extranonce1 = extranonce1_;
  out.difficulty = difficulty_;
  version = version_.load();
  return true;
}

void NexaPool::submit(const nexa::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char ntime[16];
  std::snprintf(ntime, sizeof ntime, "%08llx", static_cast<unsigned long long>(std::time(nullptr)) & 0xffffffffull);
  const int id = next_id_++;
  json m = {{"id", id}, {"method", "mining.submit"}, {"params", {login_, s.job_id, s.extranonce1_hex, ntime, s.nonce_hex}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void NexaPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.contains("method") && m["method"].is_string() ? m["method"].get<std::string>() : "";
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.set_difficulty") {
    if (p.is_array() && !p.empty() && p[0].is_number() && p[0].get<double>() > 0 && p[0].get<double>() < 4e9) {
      std::lock_guard<std::mutex> lk(mu_);
      difficulty_ = p[0].get<double>();
      ++version_;
    }
    return;
  }
  if (method == "mining.set_extranonce") {
    std::vector<uint8_t> en1;
    if (p.is_array() && p.size() >= 2 && p[0].is_string() && p[1].is_number_integer() &&
        hex_vec(p[0].get<std::string>(), en1, 32) && en1.size() == 4) {
      std::lock_guard<std::mutex> lk(mu_);
      extranonce1_ = en1;
      ++version_;
    } else {
      warn("POOL", "ignoring malformed mining.set_extranonce");
    }
    return;
  }
  if (method == "client.reconnect") {
    info("POOL", "the pool asked for a reconnect");
    reconnect_ = true;
    return;
  }
  if (method == "mining.notify") {
    nexa::Job j;
    if (!nexa_parse_notify(p.dump(), j)) {
      warn("POOL", "ignoring a malformed mining.notify");
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (j.height != height_) live_jobs_.clear();  // a new block: the earlier jobs are stale
    height_ = j.height;
    if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
    while (live_jobs_.size() > 8) live_jobs_.pop_front();
    job_ = j;
    have_job_ = true;
    ++version_;
    if (difficulty_ != last_difficulty_) {
      last_difficulty_ = difficulty_;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %g (%.3g G hashes per share)", difficulty_, difficulty_ * 4.294967296);
      info("POOL", buf);
    }
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // subscribe: [subscriptions, extranonce1 (4 bytes), 4]
    std::vector<uint8_t> en1;
    if (!err.empty() || !r.is_array() || r.size() < 2 || !r[1].is_string() || !hex_vec(r[1].get<std::string>(), en1, 32) ||
        en1.size() != 4) {
      error("POOL", "subscribe failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    extranonce1_ = en1;
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

bool NexaPool::session(const PoolEndpoint& ep) {
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
  have_job_ = authorized_ = subscribed_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void NexaPool::run() {
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
