// Quantus (QPoW) mining engine for one GPU: per-job prestate on the CPU, nonce search on the GPU, every candidate
// re-checked with the CPU reference hash.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "algo/quantus/quantus.hpp"
#include "gpu/gpu.hpp"

namespace xm::qtc {

struct Job {
  std::string id;
  std::array<uint8_t, 32> mining_hash{};  // the pre-seal header hash
  std::vector<uint8_t> extranonce;        // pool-fixed leading nonce bytes (Kryptex: 4)
  Target share_target{};
  double difficulty = 0;  // hashes per share
};

struct Share {
  std::string job_id;
  Nonce nonce{};
  Hash hash{};
  bool is_block = false;  // the pool does not send the network target
  double work = 0;
};

// --gpu-intensity N: 2^N nonces per launch (16-28); 0 = auto (24: ~0.07 s at 230 MH/s).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 24 : intensity < 16 ? 16 : intensity > 28 ? 28 : intensity);
}

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  // Tests `batch` nonces (multiple of 256). Appends verified shares; returns the number of nonces tested (0 = error).
  double iterate(uint32_t batch, std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }  // GPU candidates the CPU found above target (expected)

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0;
  Job job_;
  Nonce nonce_{};  // bytes 0..55 fixed per job; the low 64 bits are counter_
  uint64_t pre_[12] = {};
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t tpb_ = 256;
  uint32_t npt_ = 1;  // nonces per thread, as compiled into the kernel (XM_QTC_NPT)
  static constexpr uint32_t kMaxFound = 1024;
};

}  // namespace xm::qtc
