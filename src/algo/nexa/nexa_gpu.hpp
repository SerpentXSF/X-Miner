// NexaPow mining engine for one GPU: one thread per nonce (kernels/nexa/nexa.cu, compiled at run time), a signed
// fixed-base comb table of G multiples in device memory, top-64-bit target compare; every candidate is re-checked
// with the CPU reference. Nonces are commitment | 0x0c | extranonce1 (4) | 8 miner bytes (Miningcore pools).
// On RTX 30 (sm_86) with the default settings the kernel is the prebuilt cubin from tools/ngen instead: three
// kernels per batch, the comb multiplications as generated SASS (XM_NEXA_GEN=0: the compiled kernel).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/nexa/nexa.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::nexa {

struct Job {
  std::string id;
  Hash commitment{};               // raw (hashing) order
  std::vector<uint8_t> extranonce1;  // 4 bytes
  uint64_t height = 0;
  uint32_t nbits = 0;
  double difficulty = 1;  // pool share difficulty (Bitcoin units)
};

struct Share {
  std::string job_id, extranonce1_hex, nonce_hex;  // nonce: the 8 miner bytes
  uint64_t nonce = 0;
  Hash hash{};
  double work = 0;  // hashes per share at the job's difficulty
  bool is_block = false;
};

// The 12 nonce bytes hashed for a job: extranonce1 | miner nonce (8 bytes, little-endian).
std::array<uint8_t, 12> nonce_bytes(const Job& job, uint64_t nonce);

class Engine {
 public:
  // batch: nonces per launch (0 = auto; XM_NEXA_BATCH overrides).
  Engine(gpu::Device& dev, uint64_t seed, uint32_t batch = 0);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  double iterate(std::vector<Share>& shares, std::string& err);
  // Debug / selftest: powHash, R.x and s (big-endian words) of nonces base .. base + n - 1 (n a multiple of the
  // block size); an invalid key is flagged by an inverted first word.
  bool dump(uint64_t base, uint32_t n, std::vector<uint32_t>& out, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  uint32_t batch() const { return batch_; }
  int window() const { return w_; }
  bool generated() const { return gen_; }

 private:
  bool launch(uint64_t base, uint32_t n, uint32_t dump, std::string& err);

  gpu::Device& dev_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_table_ = 0, d_dbg_ = 0, d_st_ = 0, d_pr_ = 0;
  size_t dbg_bytes_ = 0;
  uint32_t split_n_ = 0;  // nonces d_st_ / d_pr_ hold
  bool split_ = false;    // three kernels (xm_nexa_a / _b / _c) instead of xm_nexa
  bool gen_ = false;      // the prebuilt sm_86 cubin with generated comb code
  gpu::RtcOutput module_;
  Job job_;
  Target target_{}, block_target_{};
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  uint32_t batch_ = 0, tpb_ = 256;
  int w_ = 20;  // comb window (24 where memory allows; see init)
  static constexpr int kGenW = 24;  // the window the prebuilt sm_86 xm_nexa_b (tools/ngen) is generated for
  std::vector<uint32_t> params_;
  static constexpr uint32_t kMaxFound = 64;
};

}  // namespace xm::nexa
