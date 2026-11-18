// Beam BeamHash III solver for one GPU (kernels/beam/beam.cu, compiled at run time): seeding and five collision rounds
// on the GPU, the 32 indices of each candidate rebuilt from the stored parent positions, and every solution checked by
// the CPU verifier. One run per nonce; a run holds about two solutions.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/beam/beam.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::beam {

struct Job {
  std::string id;
  std::array<uint8_t, 32> input{};
  std::vector<uint8_t> nonce_prefix;  // the pool's leading nonce bytes (login "nonceprefix")
  uint32_t difficulty = 0;            // Beam packed share difficulty
  uint64_t height = 0;
};

struct Share {
  std::string job_id;
  std::array<uint8_t, 8> nonce{};
  Solution solution{};
  double work = 0;  // solutions per share at the job's difficulty
  bool is_block = false;  // the pool sends no network difficulty, so blocks are not told apart
};

struct SolveStats {
  uint32_t candidates = 0;  // pairs that survived round 5 (before the CPU check)
  uint32_t overflow = 0;    // round-5 input elements over bucket capacity (earlier rounds are not counted)
  double gpu_ms = 0;
};

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed);
  ~Engine();
  bool init(std::string& err);
  // Every verified solution of (input, nonce) with extra nonce 0.
  bool solve(const uint8_t input[32], const uint8_t nonce[8], std::vector<Solution>& sols, SolveStats& st,
             std::string& err);
  bool set_job(const Job& job, std::string& err);
  // One run of the current job. Returns the verified solutions found (the miner's Sol/s), 0 on error.
  double iterate(std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  // This engine's GPU index among `units` GPUs: with a short pool nonce space each GPU gets its own range.
  void set_unit(int unit, int units);

  static constexpr uint32_t kTpb = 256;   // beam_gen
  static constexpr uint32_t kMaxFound = 64;

 private:
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::RtcOutput module_;
  uint32_t rtpb_ = 512;     // round kernels (XM_BEAM_RTPB)
  uint32_t nb_bits_ = 16;   // 2^16 buckets (XM_BEAM_NBBITS)
  uint32_t nb_ = 0;
  uint32_t cap_ = 608;      // slots per bucket: mean 512, s.d. ~23 (XM_BEAM_CAP)
  bool lds_bucket_ = true;  // see init (XM_BEAM_LDS)
  gpu::DevPtr d_a_ = 0, d_b_ = 0, d_ca_ = 0, d_cb_ = 0, d_found_ = 0;
  Job job_;
  bool have_job_ = false;
  uint64_t counter_ = 0, cpu_rejects_ = 0;
  unsigned unit_tag_ = 0, tag_bits_ = 0;  // this GPU's index and how many nonce bits carry it (set_unit)
};

}  // namespace xm::beam
