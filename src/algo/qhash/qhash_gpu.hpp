// QHash mining engine for one GPU: one thread per nonce (SHA-256 of the header from a host midstate, the 16-site
// circuit contraction, Q15 conversion and zero-byte rule, SHA-256 of the result, top-64-bit target compare), compiled
// at run time; every candidate is re-checked with the CPU reference. The engine rolls extranonce2 when the 32-bit
// nonce space of a header is used up.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "algo/qhash/qhash.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::qhash {

struct Job {
  std::string id;
  Template tpl;
  std::string ntime_hex;  // echoed in the submit
  double difficulty = 1;  // pool share difficulty (Bitcoin units)
};

struct Share {
  std::string job_id, extranonce2_hex, ntime_hex;
  uint32_t nonce = 0;
  Hash hash{};
  double work = 0;  // hashes per share at the job's difficulty
  bool is_block = false;
};

class Engine {
 public:
  // batch: nonces per launch (0 = auto; XM_QHASH_BATCH overrides). precision: -1 = default (float unless
  // XM_QHASH_FP64=1), 0 = float, 1 = double.
  Engine(gpu::Device& dev, uint64_t seed, uint32_t batch = 0, int precision = -1);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  double iterate(std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  // Next (extranonce2 << 32 | nonce) to hash.
  uint64_t next_nonce() const { return counter_; }
  uint32_t batch() const { return batch_; }
  bool fp64() const { return fp64_; }

 private:
  bool load_header(std::string& err);

  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0;
  gpu::RtcOutput module_;
  Job job_;
  Target target_{}, block_target_{};
  uint64_t counter_ = 0;      // extranonce2 << 32 | nonce
  uint64_t header_en2_ = ~0ull;
  uint8_t header_[kHeaderSize] = {};
  uint32_t midstate_[8] = {}, tail_[3] = {};
  bool have_job_ = false, fp64_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t batch_ = 0, tpb_ = 256;
  int precision_ = -1;
  static constexpr uint32_t kMaxFound = 64;
};

}  // namespace xm::qhash
