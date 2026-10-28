// Conflux Octopus mining engine for one GPU: the DAG generated on the GPU (the Ethash generator, 256 parents), a
// search kernel compiled at run time once per GPU (polynomial evaluation, then the DAG mix), every candidate
// re-checked on the CPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "algo/octopus/octopus.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::octopus {

struct Job {
  std::string id;
  Hash256 header{};    // PoW hash, bytes in order as the pool sends it
  uint64_t height = 0;
  Hash256 boundary{};  // share target, 256-bit big-endian bytes: valid when hash < boundary
  double difficulty = 0;  // hashes per share
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Hash256 header{};
  Hash256 final_hash{};
  double work = 0;
  bool is_block = false;
};

// --gpu-intensity N: 2^N nonces per launch (16-26); 0 = auto (21: Conflux has a new job every ~0.5 s).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 21 : intensity < 16 ? 16 : intensity > 26 ? 26 : intensity);
}

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  double iterate(uint32_t batch, std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  uint64_t next_nonce() const { return counter_; }

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_dag_ = 0, d_pts_ = 0;
  int dag_stage_ = -1;
  gpu::RtcOutput module_;
  std::shared_ptr<const Light> light_;
  std::unique_ptr<Points> pts_;
  Job job_;
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t tpb_ = 128;
  bool mul24_ = false;  // AMD: 24-bit multiplies in the polynomial (XM_OCTOPUS_MUL24=0: the 32-bit path)
  uint32_t lanes_ = 8;  // lanes per hash in the DAG mix (XM_OCTOPUS_LANES: 4, 8 or 16)
  static constexpr uint32_t kMaxFound = 256;
};

}  // namespace xm::octopus
