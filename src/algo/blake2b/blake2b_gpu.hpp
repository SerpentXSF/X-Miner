// BLAKE2b (RDTS) mining engine for one GPU: per-job work root and round-1 precompute on the CPU, nonce search on the
// GPU, every candidate re-checked with the CPU reference digest.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/blake2b/blake2b.hpp"
#include "gpu/gpu.hpp"

namespace xm::b2b {

struct Job {
  std::string id;
  std::array<uint8_t, 32> prevhash{};    // as sent in mining.notify (already transformed by the gateway)
  std::array<uint8_t, kCoinb1Size> coinb1{};
  std::array<uint8_t, 8> ntime{};        // header bytes, echoed verbatim
  std::string ntime_hex;                 // the notify's own spelling, for mining.submit
  std::array<uint8_t, kExtranonceSize> extranonce{};  // extranonce1 || extranonce2
  std::string extranonce2_hex;
  Target share_target{};
  Target block_target{};                 // from nbits (all 0 = unknown)
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  std::string ntime_hex, extranonce2_hex;
  Digest digest{};
  bool is_block = false;
  double work = 0;  // hashes a pool credits for this share: 2^256 / share target
};

// --gpu-intensity N: 2^N nonces per launch (20-31); 0 = auto (29: ~0.1 s on an RX 9070 XT, ~0.3 s on an RTX 3060).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 29 : intensity < 20 ? 20 : intensity > 31 ? 31 : intensity);
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
  Digest root_{};
  uint64_t m_[10] = {}, v_[16] = {};
  bool have_job_ = false, have_block_ = false;
  uint64_t nonce_ = 0;
  uint64_t cpu_rejects_ = 0;
  double share_work_ = 0;
  uint32_t tpb_ = 256;
  static constexpr uint32_t kMaxFound = 1024;
};

// The kernel's per-job precompute, exposed for tests: header words and the state after round-1 column steps 0, 1, 3.
void precompute(const Header& h, uint64_t m[10], uint64_t v[16]);

}  // namespace xm::b2b
