// BLAKE2b (RDTS) pool client: the Sia-style stratum v1 dialect of the DATUM gateway (also AlphaPool, Rabid).
//   -> mining.subscribe ["X-Miner/x.y.z"]
//   <- [[subscriptions], extranonce1 (hex), extranonce2_size]       extranonce1 + extranonce2 = 12 bytes
//   -> mining.authorize ["ADDRESS.worker", "password"]
//   <- mining.set_difficulty [d]       Sia-style: target = (2^224 - 1) / d; applies from the next notify
//   <- mining.notify [job_id, prevhash, coinb1 (39 B), "", [], version, nbits, ntime (8 or 16 hex), clean]
//   <- mining.set_extranonce [extranonce1, extranonce2_size], client.reconnect
//   -> mining.submit [login, job_id, extranonce2, ntime (echoed verbatim), nonce (8 LE bytes as hex)]
// The prevhash arrives already transformed by the gateway and ntime must never be rolled. DATUM folds the
// difficulty into coinb1, so it always follows set_difficulty with a fresh notify.
// Sources: the original Python X-Miner (GitHub\Bitcoin Knots\miner\stratum.py), written against datum_stratum.c.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "algo/blake2b/blake2b_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class Blake2bPool {
 public:
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  Blake2bPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~Blake2bPool();
  void start();
  void stop();

  // The current job; version increases on every new job. Returns false if there is none.
  bool job(b2b::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const b2b::Share& s, int tag);
  void report_rate(double) {}  // the dialect has no hashrate report

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
  b2b::Job job_;
  bool have_job_ = false;
  std::atomic<uint64_t> version_{0};
  std::vector<uint8_t> extranonce1_;
  int extranonce2_size_ = 0;
  double difficulty_ = 1, last_difficulty_ = 0;
  std::set<std::string> live_jobs_;  // jobs whose shares the pool still takes
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

// mining.notify params -> job fields (not the extranonce or targets). False on anything that is not a BLAKE2b job.
bool b2b_parse_notify(const std::string& params_json, b2b::Job& out, uint32_t& nbits, bool& clean);
// An 8-byte ntime field as the gateway reads it: 16 hex = the bytes in order; 8 hex = a u32 stored little-endian in
// bytes 0..3 with bytes 4..7 zero.
bool b2b_parse_wire8(const std::string& hex, std::array<uint8_t, 8>& out);
// Submit form of a nonce: its 8 little-endian bytes as hex.
std::string b2b_nonce_hex(uint64_t nonce);

}  // namespace xm
