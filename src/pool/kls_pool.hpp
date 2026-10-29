// Karlsen (Kaspa-style) stratum pool client for KarlsenHashV2, as karlsen-stratum-bridge speaks it:
//   -> mining.subscribe ["X-Miner/v", "EthereumStratum/1.0.0"]   <- result [true, "EthereumStratum/1.0.0"]
//   -> mining.authorize [login, pass]          <- result true
//   <- set_extranonce [hex] / mining.set_extranonce   the top nonce bytes
//   <- mining.set_difficulty [d]               share target (2^224 - 1) / d, compared little-endian
//   <- mining.notify [job id, [4 x u64 = pre-PoW hash, little-endian words], timestamp ms]
//      (also accepted: [job id, "pre-PoW hash hex" + "timestamp as 8 LE bytes hex"] and [job id, "hash hex", ts])
//   -> mining.submit [login, job id, "0x" + 16 hex digits of the full nonce]   <- result true
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

// Parses mining.notify params (a JSON array) into the job's id, pre-PoW hash and timestamp.
bool kls_parse_notify(const std::string& params_json, fishhash::Job& j);

class KlsPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  KlsPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~KlsPool();
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

  std::vector<PoolEndpoint> pools_;
  std::string login_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false}, reconnect_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  fishhash::Job job_;
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
