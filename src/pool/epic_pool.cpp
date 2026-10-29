#include "pool/epic_pool.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstring>

#include "core/console.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

bool parse_job(const json& p, kawpow::Job& j) {
  if (!p.is_object() || !p.contains("height") || !p["height"].is_number_unsigned() || !p.contains("job_id") ||
      !p["job_id"].is_number_unsigned() || !p.contains("pre_pow") || !p["pre_pow"].is_string() ||
      !p.contains("difficulty") || !p["difficulty"].is_array())
    return false;
  uint64_t diff = 0;
  for (const auto& d : p["difficulty"])
    if (d.is_array() && d.size() == 2 && d[0] == "progpow" && d[1].is_number_unsigned()) diff = d[1].get<uint64_t>();
  const std::string hex = p["pre_pow"].get<std::string>();
  if (diff == 0 || hex.empty() || hex.size() % 2 || hex.size() > 4096) return false;
  std::vector<uint8_t> pre(hex.size() / 2);
  for (size_t i = 0; i < pre.size(); ++i) {
    if (!std::isxdigit(static_cast<unsigned char>(hex[2 * i])) || !std::isxdigit(static_cast<unsigned char>(hex[2 * i + 1])))
      return false;
    pre[i] = static_cast<uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
  }
  const uint64_t height = p["height"].get<uint64_t>();
  if (height > 0x7fffffff) return false;
  const auto h = kawpow::keccak256(pre.data(), pre.size());
  std::memcpy(j.header.data(), h.data(), 32);
  j.height = static_cast<int>(height);
  j.seed = kawpow::epoch_seed(j.height / kawpow::kProgpow092.epoch_length);
  j.id = std::to_string(height) + ":" + std::to_string(p["job_id"].get<uint64_t>());
  j.target = epic_target(diff);
  j.difficulty = double(diff);
  j.nonce_prefix = 0;
  j.prefix_bits = 0;
  return true;
}

}  // namespace

kawpow::Hash256 epic_target(uint64_t d) {
  kawpow::Hash256 t{};
  uint8_t* q = reinterpret_cast<uint8_t*>(t.data());
  if (d == 0) d = 1;
  uint64_t rem = 0;
  for (int bit = 255; bit >= 0; --bit) {  // binary long division of 2^256 - 1 (all ones)
    const bool carry = rem >> 63;
    rem = rem << 1 | 1;
    if (carry || rem >= d) {
      rem -= d;
      q[31 - bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
    }
  }
  return t;
}

bool epic_parse_job(const std::string& job_json, kawpow::Job& j) {
  return parse_job(json::parse(job_json, nullptr, false), j);
}

EpicPool::EpicPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {
  // A bare address (no @domain) arrives as address.worker: Epic pools take "." as a deposit note, "+" as the worker.
  const size_t dot = login_.rfind('.');
  if (login_.find('@') == std::string::npos && login_.find('+') == std::string::npos && dot != std::string::npos)
    login_[dot] = '+';
}

EpicPool::~EpicPool() { stop(); }

void EpicPool::start() { thread_ = std::thread([this] { run(); }); }

void EpicPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string EpicPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool EpicPool::job(kawpow::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !authorized_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void EpicPool::submit(const kawpow::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || !authorized_) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  tokens_ = std::min(kMaxSharesPerSecond,
                     tokens_ + kMaxSharesPerSecond * std::chrono::duration<double>(now - last_refill_).count());
  last_refill_ = now;
  if (tokens_ < 1) {  // above what the pool answers: it would drop the share anyway
    if (dropped_++ == 0)
      info("POOL", "more shares than the pool accepts per second; sending at most 50 per second");
    return;
  }
  tokens_ -= 1;
  const size_t colon = s.job_id.find(':');
  const uint64_t height = std::stoull(s.job_id.substr(0, colon)), job_id = std::stoull(s.job_id.substr(colon + 1));
  json mix = json::array();
  const uint8_t* m = reinterpret_cast<const uint8_t*>(s.mix.data());  // the [u32; 8] mix as little-endian memory
  for (int i = 0; i < 32; ++i) mix.push_back(int(m[i]));
  const int id = next_id_++;
  json msg = {{"id", id},
              {"jsonrpc", "2.0"},
              {"method", "submit"},
              {"params", {{"height", height}, {"job_id", job_id}, {"nonce", s.nonce}, {"pow", {{"ProgPow", mix}}}}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(msg.dump());
}

void EpicPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (method == "job" || (method == "getjobtemplate" && err.empty())) {
    kawpow::Job j;
    if (!parse_job(method == "job" ? m.value("params", json()) : r, j)) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (j.height != live_height_) live_jobs_.clear();  // shares for an older height are "too late"
    live_height_ = j.height;
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
    return;
  }
  if (method == "login") {
    if (!err.empty() || r != "ok") {
      error("POOL", "login failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    authorized_ = true;
    info("POOL", "authorized as " + login_);
    return;
  }
  if (method == "keepalive" || method == "getjobtemplate") {
    if (!err.empty()) warn("POOL", method + ": " + err);
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  const bool ok = err.empty() && r.is_string() && (r == "ok" || r.get<std::string>().rfind("blockfound", 0) == 0);
  if (ok && r != "ok") log_line(Tone::Gold, "BLCK", "the pool reports a block: " + r.get<std::string>());
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

bool EpicPool::session(const PoolEndpoint& ep) {
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
  conn_.send_line(json({{"id", 1},
                        {"jsonrpc", "2.0"},
                        {"method", "login"},
                        {"params", {{"login", login_}, {"pass", password_}, {"agent", "X-Miner/" XM_VERSION}}}})
                      .dump());
  conn_.send_line(
      json({{"id", 2}, {"jsonrpc", "2.0"}, {"method", "getjobtemplate"}, {"params", {{"algorithm", "progpow"}}}}).dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now(), last_ka = last_rx;
  while (!stop_ && !reconnect_) {
    std::vector<std::string> out;
    {
      std::lock_guard<std::mutex> lk(mu_);
      out.swap(outbox_);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_ka > std::chrono::seconds(60)) {
      out.push_back(json({{"id", 3}, {"jsonrpc", "2.0"}, {"method", "keepalive"}, {"params", nullptr}}).dump());
      last_ka = now;
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

void EpicPool::run() {
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
