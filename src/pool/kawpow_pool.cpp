#include "pool/kawpow_pool.hpp"

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

double kawpow_difficulty(const kawpow::Hash256& target) {
  const uint8_t* t = reinterpret_cast<const uint8_t*>(target.data());
  int shift = 0;
  while (shift < 24 && t[shift] == 0) ++shift;
  double top = 0;
  for (int i = 0; i < 8; ++i) top = top * 256 + t[shift + i];
  return top > 0 ? std::ldexp(1.0, 64 + 8 * shift) / top : 0;  // 2^256 / target
}

KawpowPool::KawpowPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

KawpowPool::~KawpowPool() { stop(); }

void KawpowPool::start() { thread_ = std::thread([this] { run(); }); }

void KawpowPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string KawpowPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool KawpowPool::job(kawpow::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void KawpowPool::submit(const kawpow::Share& s, int tag) {
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
            {"params", {login_, s.job_id, nonce, "0x" + kawpow::hash_to_hex(s.header),
                        "0x" + kawpow::hash_to_hex(s.mix)}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void KawpowPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const json& p = m.contains("params") ? m["params"] : json();
  if (method == "mining.set_target") {
    if (p.is_array() && !p.empty() && p[0].is_string() && hex_ok(strip0x(p[0].get<std::string>()), 64)) {
      std::lock_guard<std::mutex> lk(mu_);
      job_.target = kawpow::hash_from_hex(strip0x(p[0].get<std::string>()));
      job_.difficulty = kawpow_difficulty(job_.target);
    }
    return;
  }
  if (method == "mining.notify") {
    if (!p.is_array() || p.size() < 6 || !p[0].is_string() || !p[1].is_string() || !p[2].is_string()) return;
    const std::string header = strip0x(p[1].get<std::string>()), seed = strip0x(p[2].get<std::string>());
    if (!hex_ok(header, 64) || !hex_ok(seed, 64)) return;
    kawpow::Job j;
    j.id = p[0].get<std::string>();
    j.header = kawpow::hash_from_hex(header);
    j.seed = kawpow::hash_from_hex(seed);
    j.height = p[5].is_number_integer() ? p[5].get<int>() : -1;
    std::lock_guard<std::mutex> lk(mu_);
    j.target = job_.target;
    if (p[3].is_string() && hex_ok(strip0x(p[3].get<std::string>()), 64))
      j.target = kawpow::hash_from_hex(strip0x(p[3].get<std::string>()));
    if (j.height < 0) {  // pools without the height: the epoch from the seed, any block of it (the period is then off)
      warn("POOL", "job without a block height");
      return;
    }
    j.difficulty = kawpow_difficulty(j.target);
    if (p.size() > 6 && p[6].is_string() && hex_ok(p[6].get<std::string>(), 8)) {  // compact nbits -> 256-bit target
      const uint32_t nbits = static_cast<uint32_t>(std::stoul(p[6].get<std::string>(), nullptr, 16));
      const int exp = int(nbits >> 24);
      uint8_t* t = reinterpret_cast<uint8_t*>(j.block_target.data());
      for (int k = 0; k < 3; ++k) {
        const int pos = 32 - exp + k;
        if (pos >= 0 && pos < 32) t[pos] = static_cast<uint8_t>(nbits >> (16 - 8 * k));
      }
    }
    j.nonce_prefix = prefix_;
    j.prefix_bits = prefix_bits_;
    const bool clean = p.size() > 4 && p[4].is_boolean() ? p[4].get<bool>() : true;
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
  if (method == "mining.set_extranonce" && p.is_array() && !p.empty() && p[0].is_string()) {
    std::lock_guard<std::mutex> lk(mu_);
    const std::string en = p[0].get<std::string>();
    prefix_bits_ = en.size() <= 16 ? int(en.size()) * 4 : 0;  // longer: unusable, ignored
    prefix_ = prefix_bits_ ? std::stoull(en, nullptr, 16) << (64 - prefix_bits_) : 0;
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // subscribe: [null, extranonce]
    if (r.is_array() && r.size() >= 2 && r[1].is_string()) {
      const std::string en = r[1].get<std::string>();
      std::lock_guard<std::mutex> lk(mu_);
      prefix_bits_ = en.size() <= 16 ? int(en.size()) * 4 : 0;
      prefix_ = prefix_bits_ ? std::stoull(en, nullptr, 16) << (64 - prefix_bits_) : 0;
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

bool KawpowPool::session(const PoolEndpoint& ep) {
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
    prefix_ = 0;
    prefix_bits_ = 0;
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
  for (auto& [id, p] : pending_) on_result_(false, "connection lost", p.job_id, p.tag, p.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void KawpowPool::run() {
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
