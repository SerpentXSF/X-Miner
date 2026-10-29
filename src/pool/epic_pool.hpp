// Epic Cash stratum (the node's servers/src/mining/stratumserver.rs, relayed 1:1 by epic.pool-pay.com:3333, probed
// 2026-10-28) for ProgPoW 0.9.2. Newline-delimited JSON-RPC with named params; replies echo the method:
//   -> login {login, pass, agent}            <- "ok"      login: address[@epicbox.epiccash.com][+worker]
//   -> getjobtemplate {algorithm: progpow}   <- the job (also pushed as method "job", id "Stratum")
//      job: {height, job_id, pre_pow (hex), difficulty: [[algo, share difficulty], ...], algorithm, ...}
//      ProgPoW header hash = Keccak-256(pre_pow); share target = (2^256 - 1) / the "progpow" difficulty
//   -> submit {height, job_id, nonce (JSON integer), pow: {ProgPow: [32 mix bytes]}}   <- "ok" / "blockfound - <hash>"
//   -> keepalive
// The node's ProgPoW share difficulty is tiny for a GPU (200,000: ~150 shares/s on an RX 9070 XT) and every share is
// credited at it, but pool-pay answers only ~60 per second per connection and drops the rest, so a backlog builds up and
// goes stale at each block. Shares are therefore sent at most kMaxSharesPerSecond, and only for the current height
// (the node rejects every other height as "too late").
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

// floor((2^256 - 1) / d) as big-endian bytes (d >= 1).
kawpow::Hash256 epic_target(uint64_t d);

// Parses an Epic job object (JSON text) into a ProgPoW job; false for malformed jobs or a zero ProgPoW difficulty.
bool epic_parse_job(const std::string& job_json, kawpow::Job& j);

class EpicPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  EpicPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~EpicPool();
  void start();
  void stop();

  bool job(kawpow::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const kawpow::Share& s, int tag);
  void report_rate(double) {}

 private:
  void run();
  bool session(const PoolEndpoint& ep);
  void handle(const std::string& line);

  std::vector<PoolEndpoint> pools_;
  std::string login_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false}, reconnect_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  kawpow::Job job_;
  bool have_job_ = false, authorized_ = false;
  std::atomic<uint64_t> version_{0};
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // "height:job_id" of the current height, in arrival order
  int live_height_ = -1;
  static constexpr double kMaxSharesPerSecond = 50;
  double tokens_ = kMaxSharesPerSecond;
  std::chrono::steady_clock::time_point last_refill_ = std::chrono::steady_clock::now();
  uint64_t dropped_ = 0;
  size_t pool_index_ = 0;
  int next_id_ = 10;
  struct Pending {
    std::string job_id;
    int tag;
    double work;
  };
  std::map<int, Pending> pending_;
  std::vector<std::string> outbox_;
};

}  // namespace xm
