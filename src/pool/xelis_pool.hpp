// Xelis stratum (github.com/vipor-net/xelis-stratum-protocol; Kryptex xel-*.kryptex.network:7019, probed 2026-10-28):
//   -> mining.subscribe ["X-Miner/v", ["xel/v3"]]   <- [session id, extra nonce hex, extra nonce bytes, public key hex]
//   -> mining.authorize [login, pass]               <- true
//   <- mining.set_difficulty [d]                    share: hash <= (2^256 - 1) / d (big-endian)
//   <- mining.notify [job id, timestamp ms (hex), header work hash, "xel/v3", clean]
//   -> mining.submit [login, job id, nonce (16 hex digits)]   <- true
// The 112-byte work is header work hash | timestamp BE | nonce BE | extra nonce (32) | public key (32).
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

#include "algo/xelis/xelis_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class XelisPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  XelisPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~XelisPool();
  void start();
  void stop();

  bool job(xelis::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const xelis::Share& s, int tag);
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
  xelis::Job job_;
  bool have_job_ = false, authorized_ = false, subscribed_ = false;
  std::atomic<uint64_t> version_{0};
  uint8_t extra_nonce_[32] = {}, public_key_[32] = {};
  uint64_t difficulty_ = 0;
  uint64_t last_difficulty_ = 0;
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
