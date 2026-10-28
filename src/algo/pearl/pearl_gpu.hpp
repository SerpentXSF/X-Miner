// Pearl mining engine for one GPU.
//
// Per session: random signal matrices A (m x k) and B^T (n x k) live on the GPU and in host memory.
// Per job:     the job key changes, so both matrix commitments are recomputed (chunk hashes on the GPU, tree on
//              the CPU), B's noise is generated and B' = B + E_B is written in the packed GEMM layout.
// Per iteration: one 1 KB "nonce chunk" of A is replaced, A's commitment is updated incrementally, the noise
//              seeds change, A' = A + E_A is rebuilt, and the GEMM computes every 16x16 tile's transcript; a check
//              kernel hashes each transcript and flags tiles under the share bound.
// Every GPU hit is recomputed on the CPU reference before it becomes a share.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "algo/pearl/pearl.hpp"
#include "gpu/gpu.hpp"

namespace xm::pearl {

struct Job {
  std::string id;
  Header header{};
  uint32_t share_nbits = 0;   // compact share target (pool difficulty)
  bool salted = true;         // certificate V3
};

struct Share {
  std::string job_id;
  Found found;
  std::string proof_b64;        // base64(bincode PlainProof)
  std::vector<uint8_t> proof;   // bincode PlainProof (gzip sessions compress this)
  bool is_block = false;      // also meets the header's block target
  double work = 0;            // MAC a pool credits for this share: 2^256 / share target
};

// --gpu-intensity N: N x 1024 B columns per pass (8-64); 0 = auto (64 -> 16384 x 65536 x 8192 MAC per pass; +0.7-0.9%
// over 32 on RX 9070 XT and RTX 3060, 2026-10-27).
inline uint32_t pass_columns(int intensity) {
  return intensity <= 0 ? 65536u : 1024u * static_cast<uint32_t>(intensity < 8 ? 8 : intensity > 64 ? 64 : intensity);
}

class Engine {
 public:
  Engine(gpu::Device& dev, uint32_t m, uint32_t n, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  // One iteration over all tiles. Appends verified shares; returns the number of hashes (MACs) done.
  double iterate(std::vector<Share>& shares, std::string& err);
  const Config& config() const { return cfg_; }
  uint64_t cpu_rejects() const { return cpu_rejects_; }  // GPU hits that failed CPU re-verification (should be 0)

 private:
  bool launch_noise(bool side_b, const Hash& seed, gpu::DevPtr sig, gpu::DevPtr packed, uint32_t rows, std::string& err);
  bool chunk_tree(gpu::DevPtr data, size_t len, MerkleTree& out, std::string& err);
  void fill_random(int8_t* p, size_t n);

  gpu::Device& dev_;
  Config cfg_;
  uint32_t m_, n_;
  std::mt19937_64 rng_;
  std::vector<int8_t> a_, bt_;  // host copies of the signal matrices
  gpu::DevPtr d_a_ = 0, d_bt_ = 0, d_ap_ = 0, d_bp_ = 0, d_uni_ = 0, d_pairs_ = 0, d_cvs_ = 0, d_out_ = 0,
              d_found_ = 0, d_wave_ = 0;
  Job job_;
  bool have_job_ = false;
  Hash job_key_{};
  MerkleTree tree_a_, tree_b_;
  Seeds seeds_{};
  uint64_t nonce_chunk_ = 0;
  uint64_t cpu_rejects_ = 0;
  static constexpr uint32_t kMaxFound = 64;
  static constexpr uint32_t kRecWords = 18;  // t_rows, t_cols, 16 transcript words (see kernels/pearl)
};

}  // namespace xm::pearl
