// Etchash mining engine for one GPU: the DAG generated on the GPU (the ProgPoW family's generator, 256 parents), a
// hashimoto search kernel compiled at run time once per GPU, every candidate re-checked on the CPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "algo/etchash/etchash.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::etchash {

struct Job {
  std::string id;
  Hash256 header{};       // header (seal) hash, bytes in order as the pool sends it
  Hash256 seed{};
  int height = 0;
  Hash256 target{};       // share target, 256-bit big-endian bytes
  uint64_t nonce_prefix = 0;  // pool extranonce in the top bits
  int prefix_bits = 0;
  double difficulty = 0;  // hashes per share
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Hash256 header{};
  Hash256 mix{};
  Hash256 final_hash{};
  double work = 0;
  bool is_block = false;
};

// --gpu-intensity N: 2^N nonces per launch (16-26); 0 = auto (22).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 22 : intensity < 16 ? 16 : intensity > 26 ? 26 : intensity);
}

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  double iterate(uint32_t batch, std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_dag_ = 0;
  int dag_epoch_ = -1;
  gpu::RtcOutput module_;
  std::shared_ptr<const Light> light_;
  Job job_;
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t tpb_ = 256;
  uint32_t lanes_ = 8;  // lanes per hash (XM_ETCHASH_LANES; RTX 3060: 8 = 41.0, 4 = 38.6 MH/s; RX 9070 XT the same)
  static constexpr uint32_t kMaxFound = 256;
};

}  // namespace xm::etchash
