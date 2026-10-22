// Pearl pool client: the "object" stratum dialect used by Kryptex, HeroMiners, k1pool, LuckyPool and NushyPool.
//   -> mining.subscribe ["X-Miner/x.y.z"]
//   -> mining.authorize ["wallet.worker", "password"]                 (v1 session: plain base64 proofs)
//   <- mining.notify {job_id, header(152 hex), target(64 hex, big-endian share target), height, cert_version}
//   <- mining.set_difficulty [D]   (some pools; share target = 0xFFFF * 2^208 / D)
//   -> mining.submit {worker, job_id, plain_proof}
// Sources: Pearl spec notes (docs), arabel1a/ascend_prl kryptex.c (MIT), nushypool protocol README.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "algo/pearl/pearl_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

class PearlPool {
 public:
  // Called with (accepted, error text, submitted job id, user tag) when the pool answers a submit.
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int)>;

  PearlPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~PearlPool();
  void start();
  void stop();

  // The current job; version increases on every new job. Returns false if there is none yet.
  bool job(pearl::Job& out, uint64_t& version) const;
  uint64_t job_version() const { return version_.load(); }
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  // Queues a share. tag is passed back to on_result (the device index).
  void submit(const pearl::Share& s, int tag, double hashrate);

 private:
  void run();
  bool session(const PoolEndpoint& ep);
  void handle(const std::string& line);
  void set_job_locked(const std::string& id, const std::string& header_hex, uint32_t nbits, uint32_t cert);

  std::vector<PoolEndpoint> pools_;
  std::string login_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  pearl::Job job_;
  bool have_job_ = false;
  std::atomic<uint64_t> version_{0};
  uint32_t diff_nbits_ = 0;  // from mining.set_difficulty, used when notify has no target
  size_t pool_index_ = 0;
  int next_id_ = 10;
  struct Pending {
    std::string job_id;
    int tag;
  };
  std::map<int, Pending> pending_;
  std::vector<std::string> outbox_;
};

// Compact share target from a pool target given as big-endian hex (rounded down, so local == pool acceptance).
uint32_t nbits_from_target_hex(const std::string& hex);
// Compact share target for a stratum difficulty D (target = 0xFFFF * 2^208 / D).
uint32_t nbits_from_difficulty(double d);

}  // namespace xm
