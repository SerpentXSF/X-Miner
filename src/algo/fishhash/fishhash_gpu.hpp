// FishHash / KarlsenHashV2 mining engine for one GPU: the fixed 4.5 GB dataset generated once on the GPU (the Ethash
// generator, 512 parents), a search kernel compiled at run time per variant, every candidate re-checked on the CPU.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "algo/fishhash/fishhash.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::fishhash {

struct Job {
  std::string id;
  uint8_t header[180] = {};  // Iron Fish: the pool's header; the nonce goes to bytes 172..179 (big-endian)
  Bytes32 pre_pow{};         // Karlsen
  uint64_t timestamp = 0;    // Karlsen
  Bytes32 target{};          // Iron Fish: big-endian; Karlsen: little-endian
  uint64_t nonce_prefix = 0; // pool extranonce in the top bits
  int prefix_bits = 0;
  double difficulty = 0;     // hashes per share
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Bytes32 hash{};
  double work = 0;
  bool is_block = false;
};

// --gpu-intensity N: 2^N nonces per launch (16-26); 0 = auto (22).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 22 : intensity < 16 ? 16 : intensity > 26 ? 26 : intensity);
}

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed, Variant v);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  double iterate(uint32_t batch, std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  uint64_t next_nonce() const { return counter_; }

 private:
  Bytes32 cpu_hash(uint64_t nonce) const;

  gpu::Device& dev_;
  uint64_t rng_;
  Variant v_;
  gpu::DevPtr d_found_ = 0, d_dag_ = 0;
  gpu::RtcOutput module_;
  std::shared_ptr<const Light> light_;
  Job job_;
  uint32_t cv_[8] = {};    // BLAKE3 chaining value before the block holding the nonce
  uint32_t tail_[11] = {}; // Iron Fish: header bytes 128..171
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t tpb_ = 256;
  uint32_t lanes_ = 8;  // lanes per hash (XM_FISHHASH_LANES: 4 or 8)
  static constexpr uint32_t kMaxFound = 256;
};

}  // namespace xm::fishhash
