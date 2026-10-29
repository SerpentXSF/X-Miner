#include "pool/zano_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdio>

#include "core/console.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

std::string strip0x(std::string s) { return s.rfind("0x", 0) == 0 ? s.substr(2) : s; }

bool all_hex(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); });
}

bool parse_work(const json& r, kawpow::Job& j) {
  if (!r.is_array() || r.size() < 4) return false;
  for (int i = 0; i < 4; ++i)
    if (!r[i].is_string()) return false;
  const std::string header = strip0x(r[0].get<std::string>()), seed = strip0x(r[1].get<std::string>());
  const std::string bound = strip0x(r[2].get<std::string>()), height = strip0x(r[3].get<std::string>());
  if (header.size() != 64 || !all_hex(header) || seed.size() != 64 || !all_hex(seed) || bound.size() > 64 ||
      !all_hex(bound) || height.size() > 16 || !all_hex(height))
    return false;
  const uint64_t h = std::stoull(height, nullptr, 16);
  if (h > 0x7fffffff) return false;
  j.id = header;
  j.header = kawpow::hash_from_hex(header);
  j.seed = kawpow::hash_from_hex(seed);
  j.height = static_cast<int>(h);
  j.target = kawpow::hash_from_hex(std::string(64 - bound.size(), '0') + bound);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(j.target.data());
  double t = 0;
  for (int i = 0; i < 32; ++i) t = t * 256 + b[i];
  j.difficulty = t > 0 ? std::ldexp(1.0, 256) / t : 0;
  j.nonce_prefix = 0;
  j.prefix_bits = 0;
  return j.difficulty > 0;
}

}  // namespace

bool zano_parse_work(const std::string& result_json, kawpow::Job& j) {
  return parse_work(json::parse(result_json, nullptr, false), j);
}

ZanoPool::ZanoPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), password_(std::move(password)), on_result_(std::move(on_result)) {
  const size_t dot = login.rfind('.');
  address_ = dot == std::string::npos ? login : login.substr(0, dot);
  worker_ = dot == std::string::npos ? "" : login.substr(dot + 1);
}

ZanoPool::~ZanoPool() { stop(); }

void ZanoPool::start() { thread_ = std::thread([this] { run(); }); }

void ZanoPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string ZanoPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool ZanoPool::job(kawpow::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void ZanoPool::submit(const kawpow::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce[24];
  std::snprintf(nonce, sizeof nonce, "0x%016llx", static_cast<unsigned long long>(s.nonce));
  const int id = next_id_++;
  json m = {{"id", id},
            {"jsonrpc", "2.0"},
            {"method", "eth_submitWork"},
            {"params", {nonce, "0x" + kawpow::hash_to_hex(s.header), "0x" + kawpow::hash_to_hex(s.mix)}}};
  if (!worker_.empty()) m["worker"] = worker_;
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void ZanoPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  const bool has_id = m.contains("id") && m["id"].is_number_integer();
  const int id = has_id ? m["id"].get<int>() : -1;
  if (r.is_array()) {  // work: the getWork reply or a push (no id / id 0)
    if (r.empty()) return;  // the node is not synced
    kawpow::Job j;
    if (!parse_work(r, j)) {
      if (r.size() < 4 && !warned_height_) {
        warned_height_ = true;
        error("POOL", "the pool's work has no block height, which ProgPowZ needs; use a pool that sends it (HeroMiners)");
      }
      return;
    }
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
  if (id == 1) {
    if (!err.empty() || !(r.is_boolean() && r.get<bool>())) {
      error("POOL", "login failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    authorized_ = true;
    info("POOL", "authorized as " + address_ + (worker_.empty() ? "" : " (worker " + worker_ + ")"));
    return;
  }
  if (!has_id) return;
  // true (the node, open-zano-pool) or {"status": "OK"} (HeroMiners)
  const bool ok = err.empty() && ((r.is_boolean() && r.get<bool>()) ||
                                  (r.is_object() && r.value("status", "") == "OK"));
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

bool ZanoPool::session(const PoolEndpoint& ep) {
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
    authorized_ = have_job_ = false;
    last_difficulty_ = 0;
  }
  json login = {{"id", 1}, {"jsonrpc", "2.0"}, {"method", "eth_submitLogin"}, {"params", {address_, password_}}};
  if (!worker_.empty()) login["worker"] = worker_;
  conn_.send_line(login.dump());
  conn_.send_line(json({{"id", 2}, {"jsonrpc", "2.0"}, {"method", "eth_getWork"}, {"params", json::array()}}).dump());
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
  have_job_ = authorized_ = false;
  live_jobs_.clear();
  for (auto& [id, pd] : pending_) on_result_(false, "connection lost", pd.job_id, pd.tag, pd.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void ZanoPool::run() {
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
