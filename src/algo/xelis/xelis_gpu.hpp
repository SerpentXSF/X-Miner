// XelisHash v3 mining engine for one GPU: one thread per hash, each with its own 544 KB scratchpad in GPU memory
// (stage 1 ChaCha8 fill, stage 3 sequential random walk, stage 4 BLAKE3 over the scratchpad), compiled at run time;
// every candidate re-checked on the CPU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "algo/xelis/xelis.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::xelis {

struct Job {
  std::string id;
  Work work;                // nonce bytes 40..47 are filled per hash
  uint64_t difficulty = 0;  // share: hash <= (2^256 - 1) / difficulty
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Hash hash{};
  double work = 0;
  bool is_block = false;
};

class Engine {
 public:
  // max_threads: 0 = as many hashes in flight as GPU memory allows (XM_XELIS_THREADS caps it).
  Engine(gpu::Device& dev, uint64_t seed, uint32_t max_threads = 0);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  // Hashes `threads()` nonces (the batch is fixed by the scratchpad memory).
  double iterate(std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  uint64_t next_nonce() const { return counter_; }
  uint32_t threads() const { return threads_; }

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_scratch_ = 0, d_state_ = 0;
  gpu::RtcOutput module_;
  Job job_;
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t max_threads_ = 0, threads_ = 0, tpb_ = 64;
  uint32_t slice_steps_ = 8496;  // stage 3 steps per launch (8 slices; XM_XELIS_SLICE)
  std::vector<uint64_t> cpu_scratch_;
  static constexpr uint32_t kMaxFound = 64;
};

}  // namespace xm::xelis
