// Command-line configuration. Flag names follow the conventions miners already use with other GPU miners
// (--algorithm, --pool, --wallet, --password, --worker, --gpu-id, --api-port), so launchers port easily.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gpu/tuning.hpp"

namespace xm {

struct PoolEndpoint {
  std::string host;
  uint16_t port = 0;
  bool tls = false;  // stratum+ssl:// or stratum+tls://
};

struct Config {
  std::string algorithm;              // "pearlhash", "noid", "blake2b"
  std::vector<PoolEndpoint> pools;    // first = primary, rest = failover in order
  std::string wallet;                 // payout address or pool login
  std::string password = "x";
  std::string worker;                 // optional; appended as wallet.worker unless the wallet already has one
  std::vector<int> gpu_ids;           // empty = all supported devices
  bool api_enable = true;
  std::string api_host = "127.0.0.1";
  uint16_t api_port = 21550;
  bool color = true;
  std::string log_file;
  // Per-GPU settings: one value for every selected GPU, or a comma list in --gpu-id order ("-" = leave alone).
  std::vector<std::string> intensity;  // work size per launch; empty / 0 = auto (see usage())
  std::vector<std::string> cclock, mclock, coffset, moffset, plimit;  // MHz / MHz / MHz / MHz / W or N%
  bool list_devices = false;
  bool list_algorithms = false;
  bool benchmark = false;             // hash locally without a pool (no shares submitted)
  int benchmark_seconds = 60;
  bool selftest = false;              // run CPU-vs-GPU known-answer checks and exit
  std::string pearl_share_test;       // developer: CPU-mine one Pearl share into this folder for the oracle
  std::string share_dump;             // developer: write the first share (header/nbits/base64) here
  bool ui_demo = false;               // developer: simulated devices to exercise the console and dashboard
  bool help = false;
  bool version = false;

  std::string login() const;          // wallet[.worker]
  int intensity_for(size_t gpu) const;               // 0 = auto
  gpu::TuneRequest tune_for(size_t gpu) const;      // index into the selected GPUs
};

// Parses argv. Returns an error message on bad input (nullopt on success).
std::optional<std::string> parse_args(int argc, char** argv, Config& out);

// "host:port", "stratum+tcp://host:port", "stratum+ssl://host:port". Multiple pools may be separated by
// '!' or ',' in one flag, or given by repeating --pool.
std::optional<std::string> parse_pool_list(const std::string& text, std::vector<PoolEndpoint>& out);

std::string usage();

}  // namespace xm
