#include "pool/c29_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "core/console.hpp"
#include "core/version.hpp"
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

bool hex_bytes(const std::string& hex, uint8_t* out, size_t n) {
  if (hex.size() != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) {
    int h = nib(hex[2 * i]), l = nib(hex[2 * i + 1]);
    if (h < 0 || l < 0) return false;
    out[i] = static_cast<uint8_t>(h << 4 | l);
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

}  // namespace

bool c29_parse_job(const std::string& job_json, c29::Job& out) {
  json j = json::parse(job_json, nullptr, false);
  if (!j.is_object() || !j.contains("job_id") || !j["job_id"].is_string() || !j.contains("blob") ||
      !j["blob"].is_string() || !j.contains("target") || !j["target"].is_string())
    return false;
  if (j.contains("algo") && j["algo"].is_string() && j["algo"].get<std::string>() != "cuckaroo") return false;
  out.id = j["job_id"].get<std::string>();
  const std::string target = j["target"].get<std::string>();
  if (out.id.empty() || !hex_bytes(j["blob"].get<std::string>(), out.mining_hash.data(), 32) || target.size() != 16)
    return false;
  out.difficulty = c29::target_difficulty(target.c_str());
  if (out.difficulty == 0) return false;
  out.height = j.contains("height") && j["height"].is_number_unsigned() ? j["height"].get<uint64_t>() : 0;
  out.nonce_prefix = out.prefix_mask = 0;
  if (j.contains("xn") && j["xn"].is_string()) {  // leading nonce bytes
    const std::string xn = j["xn"].get<std::string>();
    uint8_t b[8];
    if (xn.size() % 2 || xn.size() > 16 || !hex_bytes(xn, b, xn.size() / 2)) return false;
    for (size_t k = 0; k < xn.size() / 2; ++k) {
      out.nonce_prefix |= uint64_t(b[k]) << (56 - 8 * k);
      out.prefix_mask |= uint64_t(0xff) << (56 - 8 * k);
    }
  }
  return true;
}

C29Pool::C29Pool(std::vector<PoolEndpoint> pools, std::string login, std::string password,
                         ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

C29Pool::~C29Pool() { stop(); }

void C29Pool::start() { thread_ = std::thread([this] { run(); }); }

void C29Pool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string C29Pool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool C29Pool::job(c29::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void C29Pool::submit(const c29::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (std::find(live_jobs_.begin(), live_jobs_.end(), s.job_id) == live_jobs_.end() || session_id_.empty()) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  char nonce_hex[24];
  std::snprintf(nonce_hex, sizeof nonce_hex, "%016llx", static_cast<unsigned long long>(s.nonce));
  json pow = json::array();
  for (uint32_t e : s.proof) pow.push_back(e);
  int id = next_id_++;
  json m = {{"id", id},
            {"method", "submit"},
            {"params",
             {{"id", session_id_},
              {"job_id", s.job_id},
              {"nonce", nonce_hex},
              {"pow", pow},
              {"result", to_hex(s.result.data(), 32)}}}};
  pending_[id] = {s.job_id, tag, s.work};
  if (std::getenv("XM_C29_WIRE")) info("WIRE", "-> " + m.dump());  // developer
  outbox_.push_back(m.dump());
}

void C29Pool::take_job(const std::string& job_json, bool clean) {
  c29::Job j;
  if (!c29_parse_job(job_json, j)) {
    warn("POOL", "ignoring a job that is not Cuckaroo29 work");
    return;
  }
  if (clean || (j.height && j.height != height_)) live_jobs_.clear();
  height_ = j.height;
  if (std::find(live_jobs_.begin(), live_jobs_.end(), j.id) == live_jobs_.end()) live_jobs_.push_back(j.id);
  while (live_jobs_.size() > 16) live_jobs_.pop_front();
  job_ = j;
  have_job_ = true;
  ++version_;
  if (j.difficulty != last_difficulty_) {
    last_difficulty_ = j.difficulty;
    char buf[96];
    std::snprintf(buf, sizeof buf, "share difficulty %llu (42-cycles of that difficulty)", static_cast<unsigned long long>(j.difficulty));
    info("POOL", buf);
  }
  record_event("JOB", j.id);
}

void C29Pool::handle(const std::string& line) {
  if (std::getenv("XM_C29_WIRE")) info("WIRE", "<- " + line.substr(0, 600));  // developer
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  const std::string method = m.value("method", "");
  if (method == "job") {
    const json& p = m.contains("params") ? m["params"] : json();
    if (!p.is_object()) return;
    const bool clean = p.contains("clean_jobs") && p["clean_jobs"].is_boolean() && p["clean_jobs"].get<bool>();
    std::lock_guard<std::mutex> lk(mu_);
    take_job(p.contains("job") ? p["job"].dump() : p.dump(), clean);
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  const int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null())
    err = m["error"].is_object() && m["error"].contains("message") ? m["error"]["message"].dump() : m["error"].dump();
  const json& r = m.contains("result") ? m["result"] : json();
  if (id == 1) {  // login
    if (!err.empty() || !r.is_object() || !r.contains("id") || !r["id"].is_string()) {
      error("POOL", "login failed: " + (err.empty() ? line : err));
      reconnect_ = true;
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    session_id_ = r["id"].get<std::string>();
    keepalive_ = false;
    if (r.contains("extensions") && r["extensions"].is_array())
      for (const auto& e : r["extensions"])
        if (e.is_string() && e.get<std::string>() == "keepalive") keepalive_ = true;
    info("POOL", "authorized as " + login_);
    if (r.contains("job")) take_job(r["job"].dump(), true);
    return;
  }
  const bool ok = err.empty() && ((r.is_object() && r.value("status", "") == "OK") || (r.is_boolean() && r.get<bool>()));
  Pending pd;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;  // keepalive replies
    pd = it->second;
    pending_.erase(it);
  }
  on_result_(ok, ok ? std::string() : (err.empty() ? line : err), pd.job_id, pd.tag, pd.work);
}

bool C29Pool::session(const PoolEndpoint& ep) {
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
    session_id_.clear();
    last_difficulty_ = 0;
  }
  conn_.send_line(json({{"id", 1},
                        {"method", "login"},
                        {"params", {{"agent", "X-Miner/" XM_VERSION}, {"login", login_}, {"pass", password_}}}})
                      .dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now(), last_tx = last_rx;
  while (!stop_ && !reconnect_) {
    std::vector<std::string> out;
    {
      std::lock_guard<std::mutex> lk(mu_);
      out.swap(outbox_);
      if (out.empty() && keepalive_ && !session_id_.empty() &&
          std::chrono::steady_clock::now() - last_tx > std::chrono::seconds(60))
        out.push_back(json({{"id", next_id_++}, {"method", "keepalived"}, {"params", {{"id", session_id_}}}}).dump());
    }
    for (const auto& o : out) {
      if (!conn_.send_line(o)) break;
      last_tx = std::chrono::steady_clock::now();
    }
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
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) {
      warn("POOL", "no data from the pool for 3 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = false;  // the extranonce belongs to this session
  live_jobs_.clear();
  session_id_.clear();
  for (auto& [id, p] : pending_) on_result_(false, "connection lost", p.job_id, p.tag, p.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void C29Pool::run() {
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
