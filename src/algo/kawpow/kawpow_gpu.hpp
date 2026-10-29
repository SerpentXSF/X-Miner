// KawPow mining engine for one GPU: the epoch's DAG generated on the GPU from the light cache, each period's random
// program compiled at run time (NVRTC / hipRTC), nonce search on the GPU, every candidate re-checked on the CPU.
#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "algo/kawpow/kawpow.hpp"
#include "gpu/gpu.hpp"
#include "gpu/rtc.hpp"

namespace xm::kawpow {

struct Job {
  std::string id;
  Hash256 header{};       // header hash (bytes in order as the pool sends it)
  Hash256 seed{};         // epoch seed hash
  int height = 0;
  Hash256 target{};       // share target (256-bit, bytes in order)
  Hash256 block_target{}; // network target from nbits (all zero if unknown)
  uint64_t nonce_prefix = 0;  // pool extranonce in the top bits
  int prefix_bits = 0;
  double difficulty = 0;  // hashes per share
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  Hash256 header{};  // the job's header hash, echoed in the submit
  Hash256 mix{};
  Hash256 final_hash{};
  double work = 0;
  bool is_block = false;
};

// --gpu-intensity N: 2^N nonces per launch (16-26); 0 = auto (22).
inline uint32_t batch_nonces(int intensity) {
  return 1u << (intensity <= 0 ? 22 : intensity < 16 ? 16 : intensity > 26 ? 26 : intensity);
}

// The light cache of an epoch, built once and shared by every GPU.
std::shared_ptr<const Light> shared_light(const Variant& v, int epoch);
// The CUDA / HIP source of one period's search kernel and the DAG generator.
std::string kernel_source(const Variant& v, uint64_t period, uint32_t tpb);
// The shared part of every generated module: types, Keccak, KISS99 and the xm_ethash_dag DAG generator.
std::string kernel_common(int parents, uint32_t tpb);
// Generates the epoch's DAG with xm_ethash_dag (a module containing it must be loaded); replaces d_dag.
bool generate_dag(gpu::Device& dev, const Light& light, gpu::DevPtr& d_dag, std::string& err);

class Engine {
 public:
  Engine(gpu::Device& dev, uint64_t seed, const Variant& v = kKawpow);
  ~Engine();
  bool init(std::string& err);
  bool set_job(const Job& job, std::string& err);
  // Tests `batch` nonces (multiple of the block size). Appends verified shares; returns nonces tested (0 = error).
  double iterate(uint32_t batch, std::vector<Share>& shares, std::string& err);
  uint64_t cpu_rejects() const { return cpu_rejects_; }
  // At most this many candidates per second are re-checked on the CPU and returned (0 = all). For pools that cannot
  // take more (Epic Cash's tiny share difficulty), so the CPU re-check does not hold the GPU back.
  void set_share_rate_limit(double per_second) { share_rate_ = per_second; }

 private:
  bool build_dag(int epoch, std::string& err);
  bool load_program(uint64_t period, std::string& err);

  gpu::Device& dev_;
  const Variant& v_;
  uint64_t rng_;
  gpu::DevPtr d_found_ = 0, d_dag_ = 0;
  int dag_epoch_ = -1;
  uint64_t period_ = ~0ull;
  gpu::RtcOutput module_;
  std::future<gpu::RtcOutput> next_;  // the next period's program, compiling in the background
  uint64_t next_period_ = ~0ull;
  std::shared_ptr<const Light> light_;
  Job job_;
  uint64_t counter_ = 0;
  bool have_job_ = false;
  uint64_t cpu_rejects_ = 0;
  double share_rate_ = 0, share_tokens_ = 0;
  std::chrono::steady_clock::time_point share_refill_ = std::chrono::steady_clock::now();
  uint32_t tpb_ = 256;
  static constexpr uint32_t kMaxFound = 256;
};

}  // namespace xm::kawpow
