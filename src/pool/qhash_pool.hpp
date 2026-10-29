// Qubitcoin (QHash) pool client: Bitcoin stratum v1 (LuckyPool qtc-*.lproute.com:8610, probed 2026-10-29).
//   -> mining.subscribe ["X-Miner/x.y.z"]   <- [[subscriptions], extranonce1 (hex), extranonce2_size]
//   -> mining.authorize ["ADDRESS.worker", "password"]   <- true
//   <- mining.set_difficulty [d]   share: hash (LE number) <= 0xFFFF * 2^208 / d
//   <- mining.notify [job_id, prevhash (word-swapped), coinb1, coinb2, [branches], version, nbits, ntime, clean]
//   <- mining.set_extranonce [extranonce1, extranonce2_size], client.reconnect
//   -> mining.submit [login, job_id, extranonce2, ntime, nonce (8 hex, big-endian u32)]   <- true
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

#include "algo/qhash/qhash_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class QhashPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  QhashPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~QhashPool();
  void start();
  void stop();

  bool job(qhash::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const qhash::Share& s, int tag);
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
  qhash::Job job_;
  bool have_job_ = false, authorized_ = false, subscribed_ = false;
  std::atomic<uint64_t> version_{0};
  std::vector<uint8_t> extranonce1_;
  int extranonce2_size_ = 4;
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

// mining.notify params -> job (template fields, id, ntime hex); false on anything malformed. clean = params[8].
bool qhash_parse_notify(const std::string& params_json, qhash::Job& out, bool& clean);

}  // namespace xm
