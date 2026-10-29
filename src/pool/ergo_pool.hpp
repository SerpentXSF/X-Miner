// Ergo (Autolykos v2) pool client: the Bitcoin-shaped stratum Ergo pools speak (ErgoStratumServer), probed on Kryptex
// erg-*.kryptex.network:7021 on 2026-10-28.
//   -> mining.subscribe ["X-Miner/v"]   <- result [null | [...], EXTRANONCE1 (hex), EXTRANONCE2 size (bytes)]
//   -> mining.authorize [login, pass]   <- result true
//   <- mining.notify [job id, height, msg (32 B hex), "", "", version, b (decimal, the share target), "", clean]
//   <- mining.set_difficulty [d]        (b is then b / d when the pool does not fold d into b; Kryptex sends 1 / none)
//   -> mining.submit [login, job id, EXTRANONCE2 (hex), "", NONCE (16 hex)]   <- result true
// nonce = extranonce1 || extranonce2 (8 bytes, big-endian); a share is valid when hit < b.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "algo/autolykos/autolykos_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class ErgoPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  ErgoPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~ErgoPool();
  void start();
  void stop();

  bool job(autolykos::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const autolykos::Share& s, int tag);
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
  autolykos::Job job_;
  bool have_job_ = false, authorized_ = false;
  std::atomic<uint64_t> version_{0};
  std::string extranonce1_;
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // in arrival order
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
