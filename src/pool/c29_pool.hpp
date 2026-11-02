// Tari Cuckaroo29 pool client: the XMRig-style JSON-RPC dialect of Kryptex xtm-c29*.kryptex.network:7040 (probed
// with a login 2026-11-01).
//   -> login {"agent", "login": "WALLET/worker", "pass"}
//   <- result {"id": session, "status": "OK", "job": JOB}
//   <- {"method": "job", "params": JOB} (or {"job": JOB})
//      JOB = {"algo": "cuckaroo", "job_id", "blob" (the 32-byte mining hash, hex), "target" (8 bytes, hex,
//             little-endian: share difficulty = (2^64 - 1) / target), "height", optional "xn" (leading nonce bytes)}
//   -> submit {"id": session, "job_id", "nonce" (8 bytes big-endian hex), "pow" (the 42 edge indices),
//              "result" (blake2b-256 of the packed proof, hex)}
//   <- result {"status": "OK"} (or true) or an error object
// A new height retires the earlier jobs.
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

#include "algo/c29/c29_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class C29Pool {
 public:
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  C29Pool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~C29Pool();
  void start();
  void stop();

  // The current job; version increases on every new job. Returns false if there is none.
  bool job(c29::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const c29::Share& s, int tag);
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
  c29::Job job_;
  bool have_job_ = false;
  std::atomic<uint64_t> version_{0};
  std::string session_id_;
  bool keepalive_ = false;
  double last_difficulty_ = 0;
  uint64_t height_ = 0;
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

// A JOB object (see above) -> job fields. False on anything that is not usable Cuckaroo29 work.
bool c29_parse_job(const std::string& job_json, c29::Job& out);

}  // namespace xm
