// Conflux stratum pool client for Octopus, the dialect of conflux-rust's stratum and Kryptex
// cfx-*.kryptex.network:7027 (probed 2026-10-28):
//   -> mining.subscribe [login, pass]               <- result true
//   <- mining.notify [job id, height (decimal), PoW hash, boundary]   a new job every block (~0.5 s), hash < boundary
//   -> mining.submit [login, job id, "0x" nonce, "0x" PoW hash]       <- result true
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo/octopus/octopus_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

// 2^256 / boundary (hashes per share) for a 256-bit big-endian boundary.
double cfx_difficulty(const octopus::Hash256& boundary);

class CfxPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  CfxPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~CfxPool();
  void start();
  void stop();

  bool job(octopus::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const octopus::Share& s, int tag);
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
  octopus::Job job_;
  bool have_job_ = false, authorized_ = false;
  std::atomic<uint64_t> version_{0};
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // the latest jobs; the pool accepts shares for recent ones
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
