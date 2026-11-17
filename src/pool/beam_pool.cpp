#include "pool/beam_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/console.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

namespace {

int nib(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool hex_vec(const std::string& hex, std::vector<uint8_t>& out) {
  if (hex.size() % 2) return false;
  out.clear();
  for (size_t i = 0; i < hex.size(); i += 2) {
    const int h = nib(hex[i]), l = nib(hex[i + 1]);
    if (h < 0 || l < 0) return false;
    out.push_back(static_cast<uint8_t>(h << 4 | l));
  }
  return true;
}

std::string to_hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

// Job ids arrive as strings or numbers.
std::string id_string(const json& m) {
  if (!m.contains("id")) return "";
  if (m["id"].is_string()) return m["id"].get<std::string>();
  if (m["id"].is_number_unsigned()) return std::to_string(m["id"].get<uint64_t>());
  if (m["id"].is_number_integer()) return std::to_string(m["id"].get<int64_t>());
  return "";
}

}  // namespace

bool beam_parse_job(const std::string& json_line, beam::Job& out) {
  json j = json::parse(json_line, nullptr, false);
  if (!j.is_object() || !j.contains("input") || !j["input"].is_string() || !j.contains("difficulty") ||
      !j["difficulty"].is_number_unsigned())
    return false;
  out.id = id_string(j);
  std::vector<uint8_t> in;
  if (out.id.empty() || !hex_vec(j["input"].get<std::string>(), in) || in.size() != 32) return false;
  std::copy(in.begin(), in.end(), out.input.begin());
  const uint64_t d = j["difficulty"].get<uint64_t>();
  if (d > 0xffffffffull || (d >> 24) > 232) return false;
  out.difficulty = static_cast<uint32_t>(d);
  out.height = j.contains("height") && j["height"].is_number_unsigned() ? j["height"].get<uint64_t>() : 0;
  return true;
}

BeamPool::BeamPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

BeamPool::~BeamPool() { stop(); }

void BeamPool::start() { thread_ = std::thread([this] { run(); }); }

void BeamPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string BeamPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool BeamPool::job(beam::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_ || !logged_in_) return false;  // no work without the session's nonce prefix
  out = job_;
  version = version_.load();
  return true;
}

void BeamPool::submit(const beam::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!logged_in_ || std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() ||
      s.nonce.size() < nonce_prefix_.size() || !std::equal(nonce_prefix_.begin(), nonce_prefix_.end(), s.nonce.begin())) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  const json m = {{"method", "solution"},
                  {"id", s.job_id},
                  {"nonce", to_hex(s.nonce.data(), s.nonce.size())},
                  {"output", to_hex(s.solution.data(), s.solution.size())},
                  {"jsonrpc", "2.0"}};
  pending_.push_back({s.job_id, tag, s.work});
  if (std::getenv("XM_BEAM_WIRE")) info("WIRE", "-> " + m.dump());  // developer
  outbox_.push_back(m.dump());
}

void BeamPool::handle(const std::string& line) {
  if (std::getenv("XM_BEAM_WIRE")) info("WIRE", "<- " + line.substr(0, 600));  // developer
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  const std::string id = id_string(m);
  if (method == "job") {
    beam::Job j;
    if (!beam_parse_job(line, j)) {
      warn("POOL", "ignoring a job that is not BeamHash III work");
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    j.nonce_prefix = nonce_prefix_;
    if (j.height && j.height != height_) live_jobs_.clear();
    height_ = j.height;
    if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
    while (live_jobs_.size() > 16) live_jobs_.pop_front();
    job_ = j;
    have_job_ = true;
    ++version_;
    if (j.difficulty != last_difficulty_) {
      last_difficulty_ = j.difficulty;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %.0f (solutions per share)", beam::difficulty_value(j.difficulty));
      info("POOL", buf);
    }
    record_event("JOB", j.id);
    return;
  }
  if (method == "cancel") {
    std::lock_guard<std::mutex> lk(mu_);
    live_jobs_.erase(std::remove(live_jobs_.begin(), live_jobs_.end(), id), live_jobs_.end());
    if (have_job_ && job_.id == id) have_job_ = false;  // stop mining a cancelled job
    return;
  }
  if (method != "result") return;
  const int code = m.contains("code") && m["code"].is_number_integer() ? m["code"].get<int>() : 0;
  const std::string desc = m.value("description", "");
  if (id == "login") {
    if (code < 0) {
      error("POOL", "login failed: " + (desc.empty() ? line : desc));
      reconnect_ = true;
      return;
    }
    std::vector<uint8_t> prefix;
    if (m.contains("nonceprefix") && m["nonceprefix"].is_string() &&
        (!hex_vec(m["nonceprefix"].get<std::string>(), prefix) || prefix.size() > 6)) {
      error("POOL", "unusable nonce prefix from the pool: " + line);
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    nonce_prefix_ = prefix;
    job_.nonce_prefix = prefix;
    if (have_job_) ++version_;  // a job that arrived before the login reply gets the prefix
    logged_in_ = true;
    info("POOL", "authorized as " + login_);
    return;
  }
  Pending pd;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::find_if(pending_.begin(), pending_.end(), [&](const Pending& p) { return p.job_id == id; });
    if (it == pending_.end()) return;
    pd = *it;
    pending_.erase(it);
  }
  const bool ok = code == 1;
  on_result_(ok, ok ? std::string() : code == 3 ? "job expired" : (desc.empty() ? line : desc), pd.job_id, pd.tag, pd.work);
}

bool BeamPool::session(const PoolEndpoint& ep) {
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
    logged_in_ = false;
    last_difficulty_ = 0;
  }
  conn_.send_line(json({{"method", "login"}, {"api_key", login_}, {"id", "login"}, {"jsonrpc", "2.0"}}).dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now();
  const auto t_login = last_rx;
  bool authorized = false;
  while (!stop_ && !reconnect_) {
    std::vector<std::string> out;
    {
      std::lock_guard<std::mutex> lk(mu_);
      out.swap(outbox_);
    }
    for (const auto& o : out)
      if (!conn_.send_line(o)) break;
    std::string line;
    const int r = conn_.read_line(line, 200);
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
    }
    if (!authorized) {
      std::lock_guard<std::mutex> lk(mu_);
      authorized = logged_in_;
    }
    if (!authorized && std::chrono::steady_clock::now() - t_login > std::chrono::seconds(20)) {
      warn("POOL", "no login reply from the pool in 20 s; reconnecting");
      break;
    }
    if (r == 0 && std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) {
      warn("POOL", "no data from the pool for 3 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = false;  // the nonce prefix belongs to this session
  logged_in_ = false;
  live_jobs_.clear();
  for (auto& p : pending_) on_result_(false, "connection lost", p.job_id, p.tag, p.work);
  pending_.clear();
  outbox_.clear();
  return authorized;  // a session that never logged in counts as a failure (failover to the next pool)
}

void BeamPool::run() {
  int failures = 0;
  while (!stop_) {
    PoolEndpoint ep;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ep = pools_[pool_index_];
    }
    const bool had_session = session(ep);
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
