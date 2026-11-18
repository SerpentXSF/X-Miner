#include "algo/beam/beam_gpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "beam_cu.hpp"

namespace xm::beam {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct PrePowArg {
  uint64_t v[4];
};

// Element strides (32-bit words) and layout of the round-4 and round-5 inputs (see the kernel's G<R>).
constexpr uint32_t kStrideA = 16, kStride4 = 20, kStride5 = 8;
constexpr uint32_t kWk4 = 12, kRef5 = 3;

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed * 0x9e3779b97f4a7c15ull + 1) {
  static std::atomic<unsigned> engines{0};  // one engine per GPU: distinct tags for up to 16 GPUs
  unit_tag_ = engines.fetch_add(1) & 15;
}

Engine::~Engine() {
  for (gpu::DevPtr p : {d_a_, d_b_, d_ca_, d_cb_, d_found_})
    if (p) dev_.free(p);
}

bool Engine::init(std::string& err) {
  // NVIDIA: 2^16 buckets of ~512 elements, each processed in shared memory; AMD: 4096 buckets of ~8192 (its large
  // last-level cache serves the partner re-reads and merges the scattered stores; generation into 2^16 buckets took
  // 7.5 instead of 3.9 ms on a 9070 XT).
  lds_bucket_ = dev_.info().vendor != gpu::Vendor::Amd;
  if (const char* t = std::getenv("XM_BEAM_LDS")) lds_bucket_ = std::atoi(t) != 0;  // developer
  if (!lds_bucket_) nb_bits_ = 12, cap_ = 8960, rtpb_ = 256;
  if (const char* t = std::getenv("XM_BEAM_NBBITS")) nb_bits_ = static_cast<uint32_t>(std::atoi(t));  // developer
  if (const char* t = std::getenv("XM_BEAM_CAP")) cap_ = static_cast<uint32_t>(std::atoi(t));
  nb_ = 1u << nb_bits_;
  // A holds the inputs of rounds 1, 3 and 5 (16, 16 and 8 words per element), B those of rounds 2 and 4 (16, 20).
  const size_t a_bytes = size_t(nb_) * cap_ * kStrideA * 4, b_bytes = size_t(nb_) * cap_ * kStride4 * 4;
  d_a_ = dev_.alloc(a_bytes);
  d_b_ = dev_.alloc(b_bytes);
  d_ca_ = dev_.alloc(4 * nb_);
  d_cb_ = dev_.alloc(4 * nb_);
  d_found_ = dev_.alloc(4 * (1 + 2 * kMaxFound));
  if (!d_a_ || !d_b_ || !d_ca_ || !d_cb_ || !d_found_) {
    err = "not enough GPU memory for BeamHash III (needs ~" + std::to_string((a_bytes + b_bytes) >> 20) +
          " MB): " + dev_.last_error();
    return false;
  }
  std::string kernel(reinterpret_cast<const char*>(beam_cu_bytes), beam_cu_size);
  if (const char* f = std::getenv("XM_BEAM_KERNEL")) {  // developer: kernel source from a file
    std::ifstream in(f, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    if (!in) {
      err = std::string("XM_BEAM_KERNEL ") + f + " not readable";
      return false;
    }
    kernel = ss.str();
  }
  if (const char* t = std::getenv("XM_BEAM_RTPB")) rtpb_ = static_cast<uint32_t>(std::atoi(t));  // developer
  // Whole elements per warp store on NVIDIA (scattered 16-byte stores queue up there); AMD's caches merge them, and
  // the shuffles only cost time (9070 XT: generation 3.9 -> 7.5 ms with them).
  int coop = dev_.info().vendor == gpu::Vendor::Amd ? 0 : 1;
  if (const char* t = std::getenv("XM_BEAM_COOP")) coop = std::atoi(t);  // developer
  const std::string src = "#define NB_BITS " + std::to_string(nb_bits_) + "\n#define CAP " + std::to_string(cap_) +
                          "u\n#define TPB " + std::to_string(rtpb_) + "\n#define COOP_STORE " + std::to_string(coop) +
                          "\n#define LDS_BUCKET " + std::to_string(lds_bucket_ ? 1 : 0) + "\n" + kernel;
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  if (!gpu::rtc_compile(dev_.info(), src, "beam.cu", opts, module_)) {
    err = "BeamHash III kernel did not compile: " + module_.log.substr(0, 3000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "beam"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::solve(const uint8_t input[32], const uint8_t nonce[8], std::vector<Solution>& sols, SolveStats& st,
                   std::string& err) {
  st = SolveStats{};
  sols.clear();
  const auto t0 = std::chrono::steady_clock::now();
  auto fail = [&] {
    err = dev_.last_error();
    return false;
  };
  const uint8_t extra[4] = {0, 0, 0, 0};
  const PrePow pp = pre_pow(input, nonce, extra);
  PrePowArg pa{{pp.v[0], pp.v[1], pp.v[2], pp.v[3]}};
  // gen -> A; round 1 A -> B; 2 B -> A; 3 A -> B; 4 B -> A (round-5 input in A, round-4 input stays in B)
  {
    void* args[] = {&pa, &d_a_, &d_ca_};
    if (!dev_.memset(d_ca_, 0, 4 * nb_) || !dev_.launch("beam_gen", kElements / kTpb, 1, 1, kTpb, 1, 1, 0, args))
      return fail();
  }
  static const char* kRound[] = {"beam_round1", "beam_round2", "beam_round3", "beam_round4"};
  static const bool prof = std::getenv("XM_BEAM_PROF") != nullptr;  // developer: per-kernel times, element counts
  auto t_last = t0;
  gpu::DevPtr in = d_a_, cin = d_ca_, out = d_b_, cout = d_cb_;
  for (int r = 0; r < 4; ++r) {
    if (prof) {
      std::vector<uint32_t> c(nb_);
      dev_.wait();
      const double ms = ms_since(t_last);
      dev_.download(c.data(), cin, 4 * nb_);
      uint64_t tot = 0;
      uint32_t mx = 0;
      for (uint32_t x : c) tot += x, mx = std::max(mx, x);
      std::fprintf(stderr, "beam %s %.2f ms -> round %d input %llu elements, fullest bucket %u\n",
                   r ? kRound[r - 1] : "beam_gen", ms, r + 1, static_cast<unsigned long long>(tot), mx);
      t_last = std::chrono::steady_clock::now();
    }
    void* args[] = {&in, &cin, &out, &cout};
    if (!dev_.memset(cout, 0, 4 * nb_) || !dev_.launch(kRound[r], nb_, 1, 1, rtpb_, 1, 1, 0, args))
      return fail();
    std::swap(in, out);
    std::swap(cin, cout);
  }
  uint32_t max_found = kMaxFound;
  {
    void* args[] = {&in, &cin, &d_found_, &max_found};
    if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("beam_final", nb_, 1, 1, rtpb_, 1, 1, 0, args)) return fail();
  }
  uint32_t found[1 + 2 * kMaxFound];
  std::vector<uint32_t> counts(nb_);
  if (!dev_.wait() || !dev_.download(found, d_found_, sizeof found) || !dev_.download(counts.data(), cin, 4 * nb_))
    return fail();
  if (prof) std::fprintf(stderr, "beam round4 + final %.2f ms, total %.2f ms\n", ms_since(t_last), ms_since(t0));
  for (uint32_t c : counts) st.overflow += c > cap_ ? c - cap_ : 0;
  st.gpu_ms = ms_since(t0);
  st.candidates = found[0];
  const uint32_t n = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < n; ++k) {
    Indices idx{};
    bool ok = true;
    for (int side = 0; side < 2 && ok; ++side) {
      uint32_t e5[kStride5];
      const size_t lim = size_t(nb_) * cap_;  // GPU-supplied positions: skip corrupt ones (unstable overclock)
      if (found[1 + 2 * k + side] >= lim) {
        ok = false;
        break;
      }
      if (!dev_.download(e5, in + size_t(found[1 + 2 * k + side]) * kStride5 * 4, sizeof e5)) return fail();
      for (int half = 0; half < 2; ++half) {
        uint32_t e4[kStride4];
        if (e5[kRef5 + half] >= lim) {
          ok = false;
          break;
        }
        if (!dev_.download(e4, out + size_t(e5[kRef5 + half]) * kStride4 * 4, sizeof e4)) return fail();
        for (int i = 0; i < 8; ++i) idx[16 * side + 8 * half + i] = e4[kWk4 + i];
      }
    }
    if (!ok) continue;
    Solution s{};
    const auto packed = pack_indices(idx);
    std::copy(packed.begin(), packed.end(), s.begin());
    if (std::find(sols.begin(), sols.end(), s) != sols.end()) continue;
    if (verify(input, nonce, s)) {
      sols.push_back(s);
      continue;
    }
    // A pair whose subtrees share indices also lands here (it cannot be a solution): not a GPU error. 32 distinct
    // indices that still fail the CPU verifier are one.
    Indices sorted = idx;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()) ++cpu_rejects_;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  if (!have_job_ || job.input != job_.input || job.nonce_prefix != job_.nonce_prefix) {
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    counter_ = r ^ (r >> 29);
  }
  job_ = job;
  have_job_ = true;
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  std::array<uint8_t, 8> nonce{};
  const size_t np = std::min<size_t>(job_.nonce_prefix.size(), 8);
  std::copy(job_.nonce_prefix.begin(), job_.nonce_prefix.begin() + np, nonce.begin());
  uint64_t c = counter_++;
  if (np >= 4 && np < 8) {  // 4 free bytes or fewer: the top 4 bits are this GPU's, so GPUs never try the same nonce
    const unsigned bits = unsigned(8 * (8 - np));
    c = (c & ((uint64_t(1) << (bits - 4)) - 1)) | (uint64_t(unit_tag_) << (bits - 4));
  }
  for (size_t i = np; i < 8; ++i) nonce[i] = static_cast<uint8_t>(c >> (8 * (7 - i)));
  std::vector<Solution> sols;
  SolveStats st;
  if (!solve(job_.input.data(), nonce.data(), sols, st, err)) return 0;
  const double work = difficulty_value(job_.difficulty);
  for (const auto& s : sols) {
    if (!meets_difficulty(s, job_.difficulty)) continue;
    Share sh;
    sh.job_id = job_.id;
    sh.nonce = nonce;
    sh.solution = s;
    sh.work = work;
    shares.push_back(sh);
  }
  // Verified solutions found (the usual Sol/s). The coordinator reads 0 as an error, and a run with no solution is
  // normal, so such a run counts as a tiny amount of work.
  return sols.empty() ? 1e-9 : double(sols.size());
}

}  // namespace xm::beam
