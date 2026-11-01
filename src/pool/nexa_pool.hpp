// Nexa (NexaPow) pool client: the Miningcore Nexa stratum (Kryptex nexa*.kryptex.network:7026, probed with a login
// 2026-11-01; Miningcore NexaJob.cs, MIT).
//   -> mining.subscribe ["X-Miner/x.y.z"]   <- [[subscriptions], extranonce1 (4 bytes hex), 4]
//   -> mining.authorize ["login", "password"]   <- true
//   <- mining.set_difficulty [d]   share: powHash (LE number) <= 0xFFFF * 2^208 / d
//   <- mining.notify [job_id, header commitment (32 bytes hex, hashing order), height, nbits (hex)]
//   <- mining.set_extranonce [extranonce1, size], client.reconnect
//   -> mining.submit [login, job_id, extranonce1, ntime (ignored), nonce (8 bytes hex)]   <- true
// The pool hashes commitment | 0x0c | extranonce1 | nonce. There is no clean flag: a new height retires the jobs.
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

#include "algo/nexa/nexa_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class NexaPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  NexaPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~NexaPool();
  void start();
  void stop();

  bool job(nexa::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const nexa::Share& s, int tag);
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
  nexa::Job job_;
  bool have_job_ = false, authorized_ = false, subscribed_ = false;
  std::atomic<uint64_t> version_{0};
  std::vector<uint8_t> extranonce1_;
  uint64_t height_ = 0;
  double difficulty_ = 0, last_difficulty_ = 0;
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

// mining.notify params -> job (id, commitment, height, nbits); false on anything malformed.
bool nexa_parse_notify(const std::string& params_json, nexa::Job& out);

}  // namespace xm
