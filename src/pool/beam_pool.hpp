// Beam pool client: Beam's own JSON stratum (BeamMW/beam pow/stratum.h; 2Miners beam.2miners.com:5252 over TLS
// answered a login probe 2026-11-16):
//   -> {"method": "login", "api_key": "WALLET.worker", "id": "login", "jsonrpc": "2.0"}
//   <- {"method": "result", "id": "login", "code": 0, "description", "nonceprefix": "000115c6"}  (code < 0: failed)
//   <- {"method": "job", "id": JOB, "input": 32-byte hex, "difficulty": Beam packed u32, "height"}
//   <- {"method": "cancel", "id": JOB}
//   -> {"method": "solution", "id": JOB, "nonce": 8-byte hex (prefix first), "output": 104-byte hex, "jsonrpc": "2.0"}
//   <- {"method": "result", "id": JOB, "code": 1 accepted / 2 rejected / 3 expired, "description"}
// Results carry the job id, not a request id: they are matched to submissions of that job in order.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo/beam/beam_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class BeamPool {
 public:
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  BeamPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~BeamPool();
  void start();
  void stop();

  bool job(beam::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const beam::Share& s, int tag);
  void report_rate(double) {}  // the protocol has no hashrate report

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
  beam::Job job_;
  bool have_job_ = false, logged_in_ = false;
  std::atomic<uint64_t> version_{0};
  std::vector<uint8_t> nonce_prefix_;
  uint32_t last_difficulty_ = 0;
  uint64_t height_ = 0;
  std::deque<std::string> live_jobs_;
  size_t pool_index_ = 0;
  struct Pending {
    std::string job_id;
    int tag;
    double work;
  };
  std::deque<Pending> pending_;
  std::vector<std::string> outbox_;
};

// A job message -> job fields (nonce prefix not set). False on anything that is not usable BeamHash III work.
bool beam_parse_job(const std::string& json_line, beam::Job& out);

}  // namespace xm
