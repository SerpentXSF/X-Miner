// Parano1d pool client: "parano1d-stratum-v1" (rplant, InnovLab, Suprnova, AlphaPool).
//   -> mining.subscribe ["X-Miner/x.y.z"]
//   <- {result: {protocol: "parano1d-stratum-v1", session_namespace (16 hex), nonce_bits: 64}}
//   -> mining.authorize ["o1ADDRESS.worker", "password"]
//   <- mining.notify [{job_id, work_domain_id, pow_fields_hex (16 x 16-byte LE fields), share_target_hex (32 B LE),
//                      block_target_hex, nonce_prefix_hex, height, clean, ...}]
//   <- mining.pause (stop hashing), client.reconnect
//   -> mining.submit [job_id, nonce (16 LE bytes as hex), work_domain_id]
//   -> miner.stats {nonce_rate, worker} every 60 s
// Nonce layout: bytes 0..7 = searched counter (LE), bytes 8..15 = the raw nonce_prefix bytes.
// Sources: protocol notes (docs), NoidMiner and noid-mac-miner stratum clients (Apache-2.0), live capture from rplant.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo/noid/noid_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class NoidPool {
 public:
  // Called with (accepted, error text, submitted job id, user tag) when the pool answers a submit.
  // (accepted, error, job id, tag, work credited for the share)
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  NoidPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~NoidPool();
  void start();
  void stop();

  // The current job; version increases on every new job. Returns false if there is none (or the pool paused us).
  bool job(noid::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const noid::Share& s, int tag);
  void report_rate(double hashrate);  // total H/s, sent as miner.stats

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
  noid::Job job_;
  bool have_job_ = false;
  std::atomic<uint64_t> version_{0};
  std::map<std::string, std::string> work_domain_;  // job id -> work_domain_id (recent jobs)
  std::array<uint8_t, 32> last_target_{};          // last share target announced on the console
  double rate_ = 0;
  bool stats_supported_ = true;
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

// pow_fields_hex (512 hex, 16 little-endian u128 fields) -> tower-basis fields. False on bad input.
bool noid_parse_fields(const std::string& hex, noid::Fields& out);
// 64 hex -> 32 bytes as given (targets are little-endian byte strings). False on bad input.
bool noid_parse_target(const std::string& hex, std::array<uint8_t, 32>& out);
// nonce_prefix_hex (16 hex) -> nonce bits 64..127: the raw bytes read as a little-endian u64.
bool noid_parse_prefix(const std::string& hex, uint64_t& out);
// Submit form of a nonce: its 16 little-endian bytes as hex.
std::string noid_nonce_hex(const noid::U128& n);

}  // namespace xm
