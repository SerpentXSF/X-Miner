// EthereumStratum/1.0.0 ("NiceHash") pool client for Etchash, as Kryptex etc-*.kryptex.network:7033 speaks it
// (probed 2026-10-28):
//   -> mining.subscribe ["X-Miner/v", "EthereumStratum/1.0.0"]
//   <- result [["mining.notify", id, "EthereumStratum/1.0.0"], EXTRANONCE (hex, the top nonce bytes)]
//   -> mining.authorize [login, pass]        <- result true
//   <- mining.set_difficulty [d]             share target 2^224 / d (difficulty 1 = 2^32 hashes)
//   <- mining.notify [job id, seed hash, header hash, clean], "height": H (Kryptex adds the height)
//   -> mining.submit [login, job id, nonce hex without the extranonce digits]   <- result true
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

#include "algo/etchash/etchash_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class EthStratumPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  EthStratumPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~EthStratumPool();
  void start();
  void stop();

  bool job(etchash::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const etchash::Share& s, int tag);
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
  etchash::Job job_;
  bool have_job_ = false, authorized_ = false;
  std::atomic<uint64_t> version_{0};
  std::string extranonce_;
  double difficulty_ = 1;
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
