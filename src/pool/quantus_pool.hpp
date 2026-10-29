// Quantus pool client: the XMRig-style JSON-RPC dialect Kryptex speaks on qtc-*.kryptex.network:7049 (TLS 8049).
//   -> login {"agent", "login": "WALLET.worker", "pass"}
//   <- result {"id": session, "status": "OK", "extensions": ["keepalive"], "job": JOB}
//   <- {"method": "job", "params": {"clean_jobs": bool, "job": JOB}}
//      JOB = {"job_id", "mining_hash" (32 B hex), "extranonce" (leading nonce bytes, hex), "difficulty" (integer),
//             "target" (128 hex: U512::MAX / difficulty, big-endian)}
//   -> submit {"id": session, "job_id", "nonce" (64 B big-endian hex), "result" (the 64-byte hash, hex)}
//   <- result {"status": "OK"} or error {"code": -1, "message": "Stale share" | "Invalid job id" | ...}
//   -> keepalived {"id": session} after 60 s without sending, when the pool offers the extension
// Probed against Kryptex on 2026-10-24; message shapes as documented in rhvian/kangq docs/PROTOCOL.md.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "algo/quantus/quantus_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class QuantusPool {
 public:
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  QuantusPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~QuantusPool();
  void start();
  void stop();

  // The current job; version increases on every new job. Returns false if there is none.
  bool job(qtc::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const qtc::Share& s, int tag);
  void report_rate(double) {}  // the dialect has no hashrate report

 private:
  void run();
  bool session(const PoolEndpoint& ep);
  void handle(const std::string& line);
  void take_job(const std::string& job_json, bool clean);  // under mu_

  std::vector<PoolEndpoint> pools_;
  std::string login_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false}, reconnect_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  qtc::Job job_;
  bool have_job_ = false;
  std::atomic<uint64_t> version_{0};
  std::string session_id_;
  bool keepalive_ = false;
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // in arrival order: jobs whose shares the pool still takes
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

// A JOB object (see above) -> job fields. False on anything that is not usable Quantus work.
bool qtc_parse_job(const std::string& job_json, qtc::Job& out);

}  // namespace xm
