// Tari Cuckaroo29 solver for one GPU: edge generation and trimming rounds on the GPU (kernels/c29/c29.cu, compiled
// at run time), cycle finding on the CPU over the few surviving edges, edge index recovery on the GPU, and the CPU
// verifier on every proof. One graph per nonce; a graph holds a 42-cycle about once in 42 graphs.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/c29/c29.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::c29 {

struct Job {
  std::string id;
  Hash mining_hash{};
  uint64_t nonce_prefix = 0, prefix_mask = 0;  // pool extranonce ("xn") in the top bytes of the nonce
  uint64_t height = 0;
  uint64_t difficulty = 1;  // share difficulty
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Proof proof{};
  Hash result{};
  double work = 0;  // graphs per share at the job's difficulty (42 x difficulty)
  bool is_block = false;  // the pool sends no network target, so blocks are not told apart
};

struct SolveStats {
  uint32_t remaining = 0;  // edges left after the last round
  uint32_t cycles = 0;     // 42-cycles found
  bool overflow = false;   // a bucket ran out of room (the graph is skipped)
  double gpu_ms = 0, cpu_ms = 0;  // trimming (incl. read-backs), cycle search
};

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  // Solves the graph of keys k: the proofs it holds (verified on the CPU).
  bool solve(const Keys& k, std::vector<Proof>& proofs, SolveStats& st, std::string& err);
  bool set_job(const Job& job, std::string& err);
  // One graph of the current job. Returns 1 (graphs done) or 0 on error.
  double iterate(std::vector<Share>& shares, std::string& err);
  int rounds() const { return rounds_; }
  uint64_t cpu_rejects() const { return cpu_rejects_; }

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::RtcOutput module_;
  gpu::DevPtr d_a_ = 0, d_b_ = 0, d_cnt_a_ = 0, d_cnt_b_ = 0, d_bm1_ = 0, d_bm2_ = 0, d_want_ = 0, d_found_ = 0;
  uint32_t capb_ = 0, chunks_ = 0;
  int rounds_ = 200;  // trimming rounds (developer: XM_C29_ROUNDS); 160-300 measure the same
  Job job_;
  bool have_job_ = false;
  uint64_t counter_ = 0, cpu_rejects_ = 0;
  static constexpr int kNbBits = 6, kTpb = 256, kChunk = 4096;
};

}  // namespace xm::c29
