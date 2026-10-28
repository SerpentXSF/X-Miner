// Ergo Autolykos v2 mining engine for one GPU: the N x 32-byte element table rebuilt on the GPU at every new height
// (7.6 GB at N = 238.6 M, late 2026), the nonce search on the GPU, every candidate re-checked on the CPU. Kernels are
// compiled at run time once per GPU (NVRTC / hipRTC).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "algo/autolykos/autolykos.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::autolykos {

struct Job {
  std::string id;
  uint32_t height = 0;
  Bytes32 msg{};
  Bytes32 b{};            // share target: valid when hit < b (big-endian)
  uint64_t nonce_prefix = 0;  // pool extranonce1 in the top bytes
  int prefix_bits = 0;
  double difficulty = 0;  // hashes per share
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  int prefix_bits = 0;  // the submit sends the nonce without the extranonce1 digits as extranonce2
  Bytes32 hit{};
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
  double table_seconds() const { return table_seconds_; }  // last table build
  uint64_t next_nonce() const { return counter_; }          // the first nonce of the next iterate()

 private:
  bool build_table(uint32_t height, std::string& err);

  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_table_ = 0;
  uint32_t table_height_ = 0, table_n_ = 0;
  bool have_table_ = false;
  gpu::RtcOutput module_;
  Job job_;
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  double table_seconds_ = 0;
  uint32_t tpb_ = 256;
  static constexpr uint32_t kMaxFound = 256;
};

}  // namespace xm::autolykos
