// Zano "ethproxy" stratum for ProgPowZ (zano src/stratum/stratum_server.cpp; HeroMiners *.zano.herominers.com:1110,
// probed 2026-10-28):
//   -> eth_submitLogin [address, pass], "worker": name     <- true
//   -> eth_getWork []                                      <- [header hash, seed hash, boundary, height]
//   <- {"result": [the same four]} without an id when the work changes
//      boundary: 256-bit big-endian, a share is final hash <= boundary; height: 8 bytes big-endian
//   -> eth_submitWork ["0x" nonce (16 hex digits), "0x" header hash, "0x" mix hash], "worker"   <- true
// The login is sent as the bare address with the worker separate: HeroMiners reads "address.x" as a payment id.
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

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/config.hpp"
#include "net/tcp.hpp"

namespace xm {

// Parses an eth_getWork result (a JSON array) into a ProgPowZ job; false without the height element.
bool zano_parse_work(const std::string& result_json, kawpow::Job& j);

class ZanoPool {
 public:
  using ResultFn = std::function<void(bool, const std::string&, const std::string&, int, double)>;

  ZanoPool(std::vector<PoolEndpoint> pools, std::string login, std::string password, ResultFn on_result);
  ~ZanoPool();
  void start();
  void stop();

  bool job(kawpow::Job& out, uint64_t& version) const;
  bool connected() const { return connected_.load(); }
  std::string endpoint() const;

  void submit(const kawpow::Share& s, int tag);
  void report_rate(double) {}

 private:
  void run();
  bool session(const PoolEndpoint& ep);
  void handle(const std::string& line);

  std::vector<PoolEndpoint> pools_;
  std::string address_, worker_, password_;
  ResultFn on_result_;
  std::atomic<bool> stop_{false}, connected_{false}, reconnect_{false};
  std::thread thread_;
  mutable std::mutex mu_;
  net::TcpLine conn_;
  kawpow::Job job_;
  bool have_job_ = false, authorized_ = false, warned_height_ = false;
  std::atomic<uint64_t> version_{0};
  double last_difficulty_ = 0;
  std::deque<std::string> live_jobs_;  // header hashes, in arrival order
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
