// Tari Cuckaroo29 solver for one GPU: edge generation and trimming rounds on the GPU (kernels/c29/c29.cu, compiled
// at run time), cycle finding on the CPU over the few surviving edges, edge index recovery on the GPU, and the CPU
// verifier on every proof. One graph per nonce; a graph holds a 42-cycle about once in 42 graphs.
#pragma once

#include <array>
#include <cstdint>
#include <future>
#include <string>
#include <utility>
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
  using Edges = std::vector<std::pair<uint32_t, uint32_t>>;
  using Cycles = std::vector<std::array<uint32_t, kProofSize>>;
  // Solves the graph of keys k: the proofs it holds (verified on the CPU).
  bool solve(const Keys& k, std::vector<Proof>& proofs, SolveStats& st, std::string& err);
  // The same, pipelined: trims graph k on the GPU while the CPU searches the previous graph (passed earlier), then
  // returns that graph's proofs and tag (have_prev = false on the first call or after a skipped graph). st holds this
  // graph's trimming and the previous graph's search.
  bool solve_pipelined(const Keys& k, uint64_t tag, std::vector<Proof>& proofs, uint64_t& prev_tag, bool& have_prev,
                       SolveStats& st, std::string& err);
  bool set_job(const Job& job, std::string& err);
  // One graph of the current job (its shares come with the next call). Returns 1 (graphs done) or 0 on error.
  double iterate(std::vector<Share>& shares, std::string& err);
  int rounds() const { return rounds_; }
  uint64_t cpu_rejects() const { return cpu_rejects_; }

 private:
  // GPU trimming: the surviving edges (empty when the graph is skipped: st.overflow).
  bool trim(const Keys& k, Edges& edges, SolveStats& st, std::string& err);
  // Edge index recovery (GPU) and verification (CPU) of the cycles found.
  bool finish(const Keys& k, const Edges& edges, const Cycles& cycles, std::vector<Proof>& proofs, std::string& err);

  struct Pending {  // the graph whose cycle search is running
    bool active = false;
    Keys keys{};
    uint64_t tag = 0;
    Edges edges;
    std::future<std::pair<Cycles, double>> search;
  };
  gpu::Device& dev_;
  uint64_t rng_;
  gpu::RtcOutput module_;
  gpu::DevPtr d_a_ = 0, d_b_ = 0, d_cnt_a_ = 0, d_cnt_b_ = 0, d_off_ = 0, d_n_ = 0, d_bm1_ = 0, d_bm2_ = 0,
              d_want_ = 0, d_found_ = 0;
  uint32_t capb_ = 0;
  bool direct_ = false;          // direct (unstaged) edge writes in generation and split (AMD)
  bool nibble_ = false;          // 4-bit degree counters in the big rounds (NVIDIA)
  uint32_t round_shared_ = 0;    // their dynamic shared memory per block
  int stage_rounds_ = 0;         // rounds with staged survivor writes
  uint32_t gen_tpb_ = 256;  // generating threads per block (set in init; developer: XM_C29_GEN_TPB; divides kTpb)
  uint32_t split_tpb_ = kTpb;  // splitting threads per block (developer: XM_C29_SPLIT_TPB)
  // Trimming rounds. The late rounds are short and latency-bound on the GPU, while the CPU's cycle search over the
  // edges they would remove runs in parallel with the next graph: fewer rounds are faster as long as the search keeps
  // up. Adapted while mining so the search takes about a third to a half of the GPU time (RTX 3060: 200 rounds 3.36,
  // 60 3.98, 40 4.13 g/s with the CPU at 3, 55, 139 ms per graph). Developer: XM_C29_ROUNDS (fixed).
  int rounds_ = 60;
  bool adapt_rounds_ = true;
  static constexpr int kMinRounds = 40, kMaxRounds = 200;
  Job job_, pend_job_;
  bool have_job_ = false;
  uint64_t counter_ = 0, cpu_rejects_ = 0;
  Pending pend_;  // its future waits for a running search on destruction (host memory only)
  // 4096 fine buckets (2^17 nodes each: two 16 KB shared bitmaps) reached through 64 coarse ones (split in slices of
  // ~16K edges); 1024 threads, generation sized as 8 sipblocks per thread of kTpb; bucket rounds while at least 256K
  // edges are left, then one flat list.
  static constexpr int kNbBits = 12, kTpb = 1024, kGenSb = 8, kNc = 64;
  static constexpr uint64_t kSplitSlice = 16384;
  static constexpr uint32_t kMaxParts = 256, kMaxSegs = 256;  // kMaxSegs = MAX_SEGS in the kernel
  uint64_t flat_edges_ = 1ull << 18;  // developer: XM_C29_FLAT
  static constexpr uint32_t kRsTpb = 256;  // threads per block of the later bucket rounds (RS_TPB in the kernel)
  uint32_t rgrid_ = 0;  // blocks of the segmented rounds (0: one per bucket; set in init)
  uint64_t small_edges_ = 1ull << 24;  // smaller blocks below this many edges (developer: XM_C29_RSMALL)
  uint64_t seg_edges_ = 1ull << 26;   // segmented rounds from this many edges up (developer: XM_C29_SEGS)
};

}  // namespace xm::c29
