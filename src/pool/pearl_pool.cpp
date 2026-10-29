#include "pool/pearl_pool.hpp"

#include <chrono>
#include <exception>
#include <cmath>
#include <cstdio>

#include "core/console.hpp"
#include "core/gzip.hpp"
#include "core/version.hpp"
#include "nlohmann/json.hpp"

namespace xm {

using nlohmann::json;

uint32_t nbits_from_target_hex(const std::string& hex) {
  return pearl::target_to_nbits(pearl::U256::from_be_hex(hex));
}

double difficulty_from_nbits(uint32_t nbits) {
  const double mant = nbits & 0x7fffff;
  const int e = static_cast<int>(nbits >> 24);
  if (mant == 0) return 0;
  return 65535.0 * std::pow(2.0, 208) / (mant * std::pow(2.0, 8 * (e - 3)));
}

uint32_t nbits_from_difficulty(double d) {
  if (d <= 0) return 0;
  // target = 0xFFFF * 2^208 / d. Represent as mantissa * 256^(e-3) with a 3-byte mantissa (rounded down).
  long double t = 65535.0L * std::pow(2.0L, 208) / static_cast<long double>(d);
  int e = 3;
  while (t >= 8388608.0L) {  // keep mantissa < 0x800000 (sign bit clear)
    t /= 256.0L;
    ++e;
  }
  while (t < 32768.0L && e > 3) {
    t *= 256.0L;
    --e;
  }
  uint32_t mant = static_cast<uint32_t>(std::floor(t));
  return static_cast<uint32_t>(e) << 24 | mant;
}

PearlPool::PearlPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
    : pools_(std::move(pools)), login_(std::move(login)), password_(std::move(password)),
      on_result_(std::move(on_result)) {}

PearlPool::~PearlPool() { stop(); }

void PearlPool::start() { thread_ = std::thread([this] { run(); }); }

void PearlPool::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

std::string PearlPool::endpoint() const {
  std::lock_guard<std::mutex> lk(mu_);
  const auto& p = pools_[pool_index_];
  return p.host + ":" + std::to_string(p.port);
}

bool PearlPool::job(pearl::Job& out, uint64_t& version) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (!have_job_) return false;
  out = job_;
  version = version_.load();
  return true;
}

void PearlPool::submit(const pearl::Share& s, int tag, double hashrate) {
  std::lock_guard<std::mutex> lk(mu_);
  int id = next_id_++;
  json m = {{"id", id},
            {"method", "mining.submit"},
            {"params", {{"worker", login_}, {"job_id", s.job_id},
                        {"plain_proof", gzip_ ? pearl::base64(gzip_compress(s.proof.data(), s.proof.size())) : s.proof_b64},
                        {"hs", static_cast<uint64_t>(hashrate)}}}};
  pending_[id] = {s.job_id, tag, s.work};
  outbox_.push_back(m.dump());
}

void PearlPool::set_job_locked(const std::string& id, const std::string& header_hex, uint32_t nbits, uint32_t cert) {
  if (header_hex.size() != 152) {
    warn("POOL", "ignoring job " + id + ": header is " + std::to_string(header_hex.size()) + " hex chars, not 152");
    return;
  }
  pearl::Job j;
  j.id = id;
  for (int i = 0; i < 76; ++i) j.header[i] = static_cast<uint8_t>(std::stoul(header_hex.substr(2 * i, 2), nullptr, 16));
  j.share_nbits = nbits;
  j.salted = cert >= 3;
  job_ = j;
  have_job_ = true;
  ++version_;
}

void PearlPool::handle(const std::string& line) {
  json m = json::parse(line, nullptr, false);
  if (m.is_discarded() || !m.is_object()) return;
  std::string method = m.value("method", "");
  if (method == "mining.notify") {
    const json& p = m["params"];
    if (!p.is_object()) {
      warn("POOL", "array-style mining.notify is not supported for Pearl on this pool yet");
      return;
    }
    std::string id = p.value("job_id", ""), header = p.value("header", ""), target = p.value("target", "");
    uint32_t cert = p.value("cert_version", 3u);
    std::lock_guard<std::mutex> lk(mu_);
    uint32_t nbits = !target.empty() ? nbits_from_target_hex(target) : diff_nbits_;
    if (!nbits) {
      warn("POOL", "job " + id + " has no share target yet");
      return;
    }
    set_job_locked(id, header, nbits, cert);
    record_event("JOB", "height " + std::to_string(p.value("height", 0)) + " · " + id);
    if (nbits != last_nbits_) {
      last_nbits_ = nbits;
      char buf[96];
      std::snprintf(buf, sizeof buf, "share difficulty %.4g (one share per %.3g MAC)", difficulty_from_nbits(nbits),
                    difficulty_from_nbits(nbits) * 4294967296.0);
      info("POOL", buf);
    }
    return;
  }
  if (method == "mining.set_difficulty") {
    const json& p = m["params"];
    if (p.is_array() && !p.empty() && p[0].is_number()) {
      std::lock_guard<std::mutex> lk(mu_);
      diff_nbits_ = nbits_from_difficulty(p[0].get<double>());
    }
    return;
  }
  // A response to one of our requests.
  if (!m.contains("id") || !m["id"].is_number_integer()) return;
  int id = m["id"].get<int>();
  bool ok = m.contains("result") && m["result"].is_boolean() && m["result"].get<bool>();
  std::string err;
  if (m.contains("error") && !m["error"].is_null()) err = m["error"].dump();
  if (id == 2) {  // authorize
    if (ok) {
      // Kryptex answers a v2 authorize with "type":"v2" (top level or in the result) and then expects gzip proofs.
      auto is_v2 = [](const json& o) { return o.is_object() && o.value("type", "") == "v2"; };
      bool v2 = is_v2(m) || (m.contains("result") && is_v2(m["result"]));
      {
        std::lock_guard<std::mutex> lk(mu_);
        gzip_ = v2;
      }
      info("POOL", "authorized as " + login_ + (v2 ? " (gzip shares)" : ""));
    } else {
      error("POOL", "authorization failed: " + (err.empty() ? std::string("rejected") : err));
    }
    return;
  }
  Pending p;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;
    p = it->second;
    pending_.erase(it);
  }
  on_result_(ok, err, p.job_id, p.tag, p.work);
}

bool PearlPool::session(const PoolEndpoint& ep) {
  if (!conn_.connect(ep.host, ep.port, 10000, ep.tls)) {
    warn("POOL", conn_.error());
    return false;
  }
  info("POOL", "connected to " + ep.host + ":" + std::to_string(ep.port) + (ep.tls ? " over TLS" : "") + " (" +
                   std::to_string(conn_.latency_ms()) + " ms)");
  record_event("POOL", "connected to " + ep.host);
  conn_.send_line(json({{"id", 1}, {"method", "mining.subscribe"}, {"params", {"X-Miner/" XM_VERSION}}}).dump());
  // Object-form authorize: required by HeroMiners ("wallet" field), accepted by Kryptex, LuckyPool, k1pool.
  // "type":"v2" asks for gzip proofs; only a pool that echoes it switches the session (see handle()).
  {
    std::lock_guard<std::mutex> lk(mu_);
    gzip_ = false;
  }
  conn_.send_line(json({{"id", 2},
                        {"method", "mining.authorize"},
                        {"params",
                         {{"wallet", login_}, {"password", password_}, {"agent", "X-Miner/" XM_VERSION}, {"type", "v2"}}}})
                      .dump());
  connected_ = true;
  auto last_rx = std::chrono::steady_clock::now();
  while (!stop_) {
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
    } else if (std::chrono::steady_clock::now() - last_rx > std::chrono::minutes(3)) {
      warn("POOL", "no data from the pool for 3 minutes; reconnecting");
      break;
    }
  }
  connected_ = false;
  conn_.close();
  // Shares in flight are lost with the connection.
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& [id, p] : pending_) on_result_(false, "connection lost", p.job_id, p.tag, p.work);
  pending_.clear();
  outbox_.clear();
  return true;
}

void PearlPool::run() {
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
    if (failures >= 3 && pools_.size() > 1) {  // fail over to the next pool
      std::lock_guard<std::mutex> lk(mu_);
      pool_index_ = (pool_index_ + 1) % pools_.size();
      failures = 0;
      warn("POOL", "switching to " + pools_[pool_index_].host + ":" + std::to_string(pools_[pool_index_].port));
    }
    for (int i = 0; i < 50 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace xm
