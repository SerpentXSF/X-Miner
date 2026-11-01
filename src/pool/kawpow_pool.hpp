// KawPow pool client: the stratum dialect Ravencoin pools speak (kawpowminer), probed on Kryptex
// rvn-*.kryptex.network:7031 on 2026-10-28.
//   -> mining.subscribe ["X-Miner/v"]            <- result [null, EXTRANONCE (hex, the top nonce bits)]
//   -> mining.authorize [login, pass]             <- result true
//   <- mining.set_target [TARGET (64 hex, big-endian)]
//   <- mining.notify [job id, header hash, seed hash, target, clean, height, nbits]
//   -> mining.submit [login, job id, "0x" nonce (16 hex), "0x" header hash, "0x" mix hash]   <- result true
#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class KawpowPool {
 public:
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  KawpowPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~KawpowPool();
  void start();
  void stop();

  bool job(kawpow::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void set_job_grace_ms(int ms) { grace_ms_ = ms; }
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
  uint64_t prefix_ = 0;
  int prefix_bits_ = 0;
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // in arrival order
  // Jobs retired by a clean notify, still submitted for grace_ms_ (QuaiPool; developer override XM_JOB_GRACE_MS).
  std::deque<std::pair<std::string, std::chrono::steady_clock::time_point>> retired_;
  int grace_ms_ = 0;
  uint64_t grace_sent_ = 0, grace_ok_ = 0;
  size_t pool_index_ = 0;
  int next_id_ = 10;
  struct Pending {
    std::string job_id;
    int tag;
    double work;
    bool grace = false;
  };
  std::map<int, Pending> pending_;
  std::vector<std::string> outbox_;
};

// Hashes per share for a 256-bit big-endian target: 2^256 / (target + 1).
double kawpow_difficulty(const kawpow::Hash256& target);

// Quai on Kryptex sends a clean job every few seconds; a share found on the job before is still accepted
// (2026-11-01: 1 of 1 such shares accepted, 22 / 0 in 20 minutes), so it is sent within 3 s instead of dropped.
class QuaiPool : public KawpowPool {
 public:
  QuaiPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result)
      : KawpowPool(std::move(pools), std::move(login), std::move(password), std::move(on_result)) {
    if (!std::getenv("XM_JOB_GRACE_MS")) set_job_grace_ms(3000);
  }
};

}  // namespace xm
