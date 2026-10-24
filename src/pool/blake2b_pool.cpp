#include "pool/blake2b_pool.hpp"

#include <chrono>
#include <cstdio>

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

bool b2b_parse_wire8(const std::string& hex, std::array<uint8_t, 8>& out) {
  out.fill(0);
  if (hex.size() == 16) return hex_bytes(hex, out.data(), 8);
  uint8_t b[4];
  if (hex.size() != 8 || !hex_bytes(hex, b, 4)) return false;
  for (int i = 0; i < 4; ++i) out[i] = b[3 - i];  // u32 "0000abcd" -> cd ab 00 00
  return true;
}

std::string b2b_nonce_hex(uint64_t nonce) {
  uint8_t b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(nonce >> (8 * i));
  return to_hex(b, 8);
}

bool b2b_parse_notify(const std::string& params_json, b2b::Job& out, uint32_t& nbits, bool& clean) {
  json p = json::parse(params_json, nullptr, false);
  if (!p.is_array() || p.size() < 9 || !p[0].is_string() || !p[1].is_string() || !p[2].is_string() ||
      !p[6].is_string() || !p[7].is_string())
    return false;
  // An empty coinb2 and no merkle branches mark a BLAKE2b job; anything else is SHA-256d work.
  if (!(p[3].is_null() || (p[3].is_string() && p[3].get<std::string>().empty()))) return false;
  if (!(p[4].is_null() || (p[4].is_array() && p[4].empty()))) return false;
  out.id = p[0].get<std::string>();
  out.ntime_hex = p[7].get<std::string>();
  const std::string nb = p[6].get<std::string>();
  if (out.id.empty() || !hex_bytes(p[1].get<std::string>(), out.prevhash.data(), 32) ||
      !hex_bytes(p[2].get<std::string>(), out.coinb1.data(), b2b::kCoinb1Size) ||
      !b2b_parse_wire8(out.ntime_hex, out.ntime) || nb.size() != 8)
    return false;
  uint8_t b[4];
  if (!hex_bytes(nb, b, 4)) return false;
  nbits = uint32_t(b[0]) << 24 | uint32_t(b[1]) << 16 | uint32_t(b[2]) << 8 | b[3];
  clean = p[8].is_boolean() ? p[8].get<bool>() : true;
  return true;
}

Blake2bPool::Blake2bPool(std::vector<PoolEndpoint> pools, std::string login, std::string password,
                         ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

Blake2bPool::~Blake2bPool() { stop(); }

void Blake2bPool::start() { thread_ = std::thread([this] { run(); }); }

void Blake2bPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string Blake2bPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool Blake2bPool::job(b2b::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void Blake2bPool::submit(const b2b::Share& s, int tag) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!live_jobs_.count(s.job_id)) {
    on_result_(false, "job expired before submit", s.job_id, tag, s.work);
    return;
  }
  int id = next_id_++;
  json m = {{"id", id},
            {"method", "mining.submit"},
            {"params", {login_, s.job_id, s.extranonce2_hex, s.ntime_hex, b2b_nonce_hex(s.nonce)}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
  if (s.is_block) log_line(Tone::Gold, "BLOK", "this share also meets the network target: block candidate");
}

void Blake2bPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  std::string method = m.value("method", "");
  if (method == "mining.notify") {
    b2b::Job j;
    uint32_t nbits = 0;
    bool clean = true;
    if (!m.contains("params") || !b2b_parse_notify(m["params"].dump(), j, nbits, clean)) {
      warn("POOL", "ignoring a job that is not BLAKE2b work");
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (extranonce1_.empty() && extranonce2_size_ == 0) {
      warn("POOL", "job before subscription; ignored");
      return;
    }
    std::copy(extranonce1_.begin(), extranonce1_.end(), j.extranonce.begin());  // extranonce2 = zeros
    j.extranonce2_hex = std::string(2 * static_cast<size_t>(extranonce2_size_), '0');
    j.share_target = b2b::target_for_difficulty(difficulty_);
    j.block_target = b2b::target_from_nbits(nbits);
    if (clean) live_jobs_.clear();
    while (live_jobs_.size() >= 16) live_jobs_.erase(live_jobs_.begin());
    live_jobs_.insert(j.id);
    job_ = j;
    have_job_ = true;
    ++version_;
    if (difficulty_ != last_difficulty_) {
      last_difficulty_ = difficulty_;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %.6g (%.4g hashes per share)", difficulty_,
                    b2b::target_work(j.share_target));
      info("POOL", buf);
    }
    record_event("JOB", j.id);
    return;
  }
  if (method == "mining.set_difficulty") {
    const json& p = m["params"];
    if (p.is_array() && !p.empty() && p[0].is_number() && p[0].get<double>() > 0) {
      std::lock_guard<std::mutex> lk(mu_);
      difficulty_ = p[0].get<double>();
    }
    return;
  }
  if (method == "mining.set_extranonce") {
    const json& p = m["params"];
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint8_t> en1;
    if (p.is_array() && p.size() >= 2 && p[0].is_string() && p[1].is_number_integer()) {
      const std::string h = p[0].get<std::string>();
      en1.resize(h.size() / 2);
      if (hex_bytes(h, en1.data(), en1.size()) && en1.size() + p[1].get<size_t>() == b2b::kExtranonceSize) {
        extranonce1_ = en1;
        extranonce2_size_ = p[1].get<int>();
        have_job_ = false;  // work roots change; wait for the next notify
        return;
      }
    }
    warn("POOL", "ignoring malformed mining.set_extranonce");
    return;
  }
  if (method == "client.reconnect") {
    info("POOL", "pool asked us to reconnect");
    reconnect_ = true;
    return;
  }
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  int id = m["id"].get<int>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  if (id == 1) {  // subscribe
    const json& r = m.contains("result") ? m["result"] : json();
    if (r.is_array() && r.size() >= 3 && r[1].is_string() && r[2].is_number_integer()) {
      const std::string h = r[1].get<std::string>();
      std::vector<uint8_t> en1(h.size() / 2);
      const int en2 = r[2].get<int>();
      if (hex_bytes(h, en1.data(), en1.size()) && en2 >= 0 && en1.size() + size_t(en2) == b2b::kExtranonceSize) {
        std::lock_guard<std::mutex> lk(mu_);
        extranonce1_ = en1;
        extranonce2_size_ = en2;
        return;
      }
    }
    error("POOL", "unexpected subscribe reply (not a BLAKE2b gateway?): " + (err.empty() ? line : err));
    reconnect_ = true;
    return;
  }
  bool ok = m.contains("result") && m["result"].is_boolean() && m["result"].get<bool>();
  if (id == 2) {
    if (ok)
      info("POOL", "authorized as " + login_);
    else
      error("POOL", "authorization failed: " + (err.empty() ? std::string("rejected") : err));
    return;
  }
  Pending pd;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;
    pd = it->second;
    pending_.erase(it);
  }
  on_result_(ok, err, pd.job_id, pd.tag, pd.work);
}

bool Blake2bPool::session(const PoolEndpoint& ep) {
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
    extranonce1_.clear();
    extranonce2_size_ = 0;
    difficulty_ = 1;
    last_difficulty_ = 0;
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
      handle(line);
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) {
      warn("POOL", "no data from the pool for 3 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  std::lock_guard<std::mutex> lk(mu_);
  have_job_ = false;  // extranonce1 belongs to this session
  live_jobs_.clear();
  for (auto& [id, p] : pending_) on_result_(false, "connection lost", p.job_id, p.tag, p.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void Blake2bPool::run() {
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
