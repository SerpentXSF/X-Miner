#include "pool/kls_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdio>

#include <cstring>

#include "algo/blake2b/blake2b.hpp"
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

// Share target for a difficulty: floor((2^224 - 1) / d) as the bridge computes it; little-endian bytes.
fishhash::Bytes32 target_for(double d) {
  const b2b::Target le = b2b::target_for_difficulty(d);
  fishhash::Bytes32 t{};
  for (int i = 0; i < 32; ++i) t[i] = le[i];
  return t;
}

bool hex_bytes(const std::string& s, uint8_t* out, size_t n) {
  if (!hex_ok(s, 2 * n)) return false;
  for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(std::stoul(s.substr(2 * i, 2), nullptr, 16));
  return true;
}

// mining.notify params: [job id, [4 x u64], timestamp], [job id, "hash hex" + "timestamp LE hex"] or
// [job id, "hash hex", timestamp].
bool parse_notify(const json& p, fishhash::Job& j) {
  if (!p.is_array() || p.size() < 2 || !p[0].is_string()) return false;
  j.id = p[0].get<std::string>();
  if (p[1].is_array() && p[1].size() == 4 && p.size() >= 3 && p[2].is_number_unsigned()) {
    for (int i = 0; i < 4; ++i) {
      if (!p[1][i].is_number_unsigned()) return false;
      const uint64_t w = p[1][i].get<uint64_t>();
      std::memcpy(j.pre_pow.data() + 8 * i, &w, 8);  // little-endian words
    }
    j.timestamp = p[2].get<uint64_t>();
  } else if (p[1].is_string()) {
    const std::string h = strip0x(p[1].get<std::string>());
    if (h.size() == 80) {
      uint8_t ts[8];
      if (!hex_bytes(h.substr(0, 64), j.pre_pow.data(), 32) || !hex_bytes(h.substr(64), ts, 8)) return false;
      std::memcpy(&j.timestamp, ts, 8);
    } else if (h.size() == 64 && p.size() >= 3 && p[2].is_number_unsigned()) {
      if (!hex_bytes(h, j.pre_pow.data(), 32)) return false;
      j.timestamp = p[2].get<uint64_t>();
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

}  // namespace

bool kls_parse_notify(const std::string& params_json, fishhash::Job& j) {
  return parse_notify(json::parse(params_json, nullptr, false), j);
}

KlsPool::KlsPool(std::vector<PoolEndpoint> pools, std::string login, std::string password,
                               ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

KlsPool::~KlsPool() { stop(); }

void KlsPool::start() { thread_ = std::thread([this] { run(); }); }

void KlsPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string KlsPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool KlsPool::job(fishhash::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void KlsPool::submit(const fishhash::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "0x%016llx", static_cast<unsigned long long>(s.nonce));
  int id = next_id_++;
  json m = {{"id", id}, {"method", "mining.submit"}, {"params", {login_, s.job_id, nonce}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void KlsPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.set_difficulty") {
    if (p.is_array() && !p.empty() && p[0].is_number()) {
      std::lock_guard<std::mutex> lk(mu_);
      difficulty_ = p[0].get<double>();
    }
    return;
  }
  if (method == "mining.set_extranonce" || method == "set_extranonce") {
    if (p.is_array() && !p.empty() && p[0].is_string()) {
      std::lock_guard<std::mutex> lk(mu_);
      extranonce_ = p[0].get<std::string>();
    }
    return;
  }
  if (method == "mining.notify") {
    fishhash::Job j;
    if (!parse_notify(p, j)) return;
    std::lock_guard<std::mutex> lk(mu_);
    j.target = target_for(difficulty_);
    j.difficulty = difficulty_ * 4294967296.0;
    j.prefix_bits = extranonce_.size() <= 16 ? int(extranonce_.size()) * 4 : 0;
    j.nonce_prefix = j.prefix_bits ? std::stoull(extranonce_, nullptr, 16) << (64 - j.prefix_bits) : 0;
    const bool clean = p.size() > 3 && p[3].is_boolean() && p[3].get<bool>();  // blocks every ~1 s: keep recent jobs
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
    record_event("JOB", j.id);
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // subscribe: [true, "EthereumStratum/1.0.0"] (or [[...], extranonce] on some pools)
    if (r.is_array() && r.size() >= 2 && r[0].is_array() && r[1].is_string()) {
      std::lock_guard<std::mutex> lk(mu_);
      extranonce_ = r[1].get<std::string>();
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

bool KlsPool::session(const PoolEndpoint& ep) {
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
    extranonce_.clear();
    difficulty_ = 1;
  }
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {"X-Miner/" XM_VERSION, "EthereumStratum/1.0.0"}}}).dump());
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

void KlsPool::run() {
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
