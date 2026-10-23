// Parano1d mining engine for one GPU: per-job midstate on the CPU, nonce search on the GPU, every candidate
// re-checked with the CPU reference digest.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/noid/noid.hpp"
#include "gpu/gpu.hpp"

namespace xm::noid {

struct Job {
  std::string id;
  Fields fields{};                       // tower basis; field 10 (nonce) is ignored
  std::array<uint8_t, 32> share_target{};  // little-endian
  std::array<uint8_t, 32> block_target{};  // little-endian (all 0 = unknown)
  uint64_t nonce_prefix = 0;             // pool session namespace: nonce bits 64..127
};

struct Share {
  std::string job_id;
  U128 nonce;       // tower basis, as submitted
  Digest digest{};
  bool is_block = false;
  double work = 0;  // hashes a pool credits for this share: 2^256 / share target
};

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
  gpu::DevPtr d_rc_ = 0, d_cm_ = 0, d_f2t_ = 0, d_found_ = 0;
  Job job_;
  State mid_{};
  bool have_job_ = false;
  uint32_t hi32_ = 0, lo32_ = 0;
  uint64_t cpu_rejects_ = 0;
  double share_work_ = 0;
  uint32_t tpb_ = 256;
  static constexpr uint32_t kMaxFound = 1024;
  static constexpr uint32_t kLanes = 1;  // nonces per GPU thread (XM_NOID_LANES in kernels/noid/noid.cu)
};

}  // namespace xm::noid
