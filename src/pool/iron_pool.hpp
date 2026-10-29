// Iron Fish native stratum (ironfish src/mining/stratum, protocol version 3), as Kryptex iron-*.kryptex.network:7017
// speaks it (probed 2026-10-28). Newline-delimited {"id", "method", "body"} objects, not JSON-RPC params:
//   -> mining.subscribe {version: 3, publicAddress: login, agent}
//   <- mining.subscribed {clientId, xn}         xn: the leading nonce bytes (header bytes 172..)
//   <- mining.set_target {target}               32 bytes big-endian: a share is hash <= target
//   <- mining.notify {miningRequestId, header}  the 180-byte FishHash header
//   <- mining.wait_for_work                     pause
//   -> mining.submit {miningRequestId, randomness: 16 hex digits, starting with xn}
//   <- mining.submitted {id: the submit's id, result, message?}
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

#include "algo/fishhash/fishhash_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class IronPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  IronPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~IronPool();
  void start();
  void stop();

  bool job(fishhash::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const fishhash::Share& s, int tag);
  void report_rate(double) {}

 private:
  void run();
  bool session(const PoolEndpoint& ep);
  void handle(const std::string& line);
  void publish_locked();

  std::vector<PoolEndpoint> pools_;
  std::string login_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false}, reconnect_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  fishhash::Job job_;
  bool have_header_ = false, have_target_ = false, subscribed_ = false, waiting_ = false;
  std::atomic<uint64_t> version_{0};
  std::string xn_;
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // in arrival order (the ids are random numbers)
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
