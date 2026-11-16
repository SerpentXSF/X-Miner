#include "algo/pearl/pearl_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/console.hpp"
#include "gpu/kernels.hpp"

namespace xm::pearl {

namespace {

// Must match XM_PEARL_STAGES / XM_PEARL_BK in kernels/pearl/pearl.cu (dynamic shared memory size).
constexpr unsigned kNvWm = 64;  // = XM_PEARL_WM (warp tile height: hash-tile rows and block size)
constexpr unsigned kNvStages = 2, kNvBk = 128, kNvBm = 128, kNvBn = 384 - kNvBm;  // + XM_PEARL_BM

struct Key8 {
  uint32_t w[8];
};
struct U256w {
  uint32_t w[8];
};

Key8 key8(const Hash& h) {
  Key8 k;
  auto w = blake3::key_words(h.data());
  for (int i = 0; i < 8; ++i) k.w[i] = w[i];
  return k;
}

unsigned blocks(size_t threads, unsigned bs) { return static_cast<unsigned>((threads + bs - 1) / bs); }

}  // namespace

// NVIDIA GPUs before Ampere (no int8 m16n8k32 mma) get the SIMT kernel of kernels/pearl/pearl.cu. XM_PEARL_SIMT=1
// selects its launch on newer GPUs too (developer check, together with an XM_KERNEL_FILE built with -DXM_PEARL_SIMT).
bool Engine::nv_simt() const {
  static const bool force = std::getenv("XM_PEARL_SIMT") != nullptr;
  const std::string& a = dev_.info().arch;
  return force || (a.rfind("sm_", 0) == 0 && std::atoi(a.c_str() + 3) < 80);
}

Engine::Engine(gpu::Device& dev, uint32_t m, uint32_t n, uint64_t seed) : dev_(dev), m_(m), n_(n), rng_(seed) {
  // K 8192 halves the per-tile epilogue (fold, hash, store) per MAC: +1.1% on the RTX 3060, +1.2% on the RX 9070 XT,
  // twice the share-proof size (Kryptex accepts them). Consensus allows 1024 <= k <= 65536.
  cfg_.k = 8192;
  if (const char* e = std::getenv("XM_PEARL_K")) cfg_.k = static_cast<uint32_t>(std::atoi(e));  // developer A/B
  cfg_.rank = 128;
  // Hash tiles match each kernel's accumulator ownership, so each lane folds and hashes its own tile. NVIDIA: the 128 accumulators one lane owns in a 64x64 warp tile of m16n8 fragments, rows
  // {0, 8, ..., 56} x cols {0, 1, 8, 9, ..., 56, 57}, so the fold needs no cross-lane traffic (kernels/pearl/pearl.cu).
  if (dev_.info().vendor == gpu::Vendor::Nvidia) {
    std::vector<uint32_t> rows, cols;
    for (uint32_t r = 0; r < kNvWm; r += 8) rows.push_back(r);
    for (uint32_t j = 0; j < 8; ++j) cols.insert(cols.end(), {8 * j, 8 * j + 1});
    Pattern::from_list(rows, cfg_.rows);
    Pattern::from_list(cols, cfg_.cols);
  } else {
    // AMD (gfx12 WMMA, wave32): a lane owns rows 8*(lane/16) + 0..7 of column lane%16 in each 16x16 accumulator;
    // over a 64x64 wave tile that is rows {0..7, 16..23, 32..39, 48..55} x cols {0, 16, 32, 48}.
    std::vector<uint32_t> rows, cols = {0, 16, 32, 48};
    for (uint32_t b = 0; b < 64; b += 16)
      for (uint32_t r = 0; r < 8; ++r) rows.push_back(b + r);
    Pattern::from_list(rows, cfg_.rows);
    Pattern::from_list(cols, cfg_.cols);
  }
}

Engine::~Engine() {
  for (gpu::DevPtr p : {d_a_, d_bt_, d_ap_, d_bp_, d_uni_, d_pairs_, d_cvs_, d_out_, d_found_, d_wave_}) dev_.free(p);
}

void Engine::fill_random(int8_t* p, size_t n) {
  // Signal values 63 or 64 (one random bit each). Consensus allows [-64, 64] and the work is the same (the GEMM runs on
  // A + noise, noise in [-63, 63]), but with this signal every GEMM operand is in [0, 127]: no sign bits flip between
  // operands, and matrix-unit power depends on bit toggling, so the clock rises at a power limit. 2026-11-15, 60 s
  // benchmarks against the earlier signal in [-8, 7]: RX 9070 XT stock 121.6 -> 128.1 TH/s (339 W); RTX 3060 at
  // 100 W 36.3 -> 37.8. A signal of 0 gains nothing, -64/-63 a little less than this; a constant 64 outside the
  // per-nonce chunk is no faster. One bit per byte also makes share proofs compress ~8x for gzip pools.
  size_t i = 0;
  while (i < n) {
    uint64_t r = rng_();
    for (int j = 0; j < 64 && i < n; ++j, ++i) p[i] = static_cast<int8_t>(63 + int((r >> j) & 1));
  }
}

bool Engine::init(std::string& err) {
  if (m_ % 128 || n_ % 256) {
    err = "matrix rows must be a multiple of 128 and columns of 256";
    return false;
  }
  if (cfg_.k < 2048 || cfg_.k % 2048) {  // the kernels' chunk and transcript loops (XM_PEARL_K is a developer setting)
    err = "Pearl k must be a multiple of 2048";
    return false;
  }
  dev_.make_current();
  if (!gpu::load_kernels(dev_, "pearl", err)) return false;
  const uint32_t k = cfg_.k, r = cfg_.rank;
  size_t sa = size_t(m_) * k, sb = size_t(n_) * k;
  a_.resize(sa);
  bt_.resize(sb);
  fill_random(a_.data(), sa);
  fill_random(bt_.data(), sb);
  d_a_ = dev_.alloc(sa);
  d_bt_ = dev_.alloc(sb);
  d_ap_ = dev_.alloc(sa);
  d_bp_ = dev_.alloc(sb);
  d_uni_ = dev_.alloc(size_t(std::max(m_, n_)) * r);
  d_pairs_ = dev_.alloc(size_t(k) * 2 * 4);
  d_cvs_ = dev_.alloc(std::max(sa, sb) / 1024 * 32);
  d_out_ = dev_.alloc(256);  // unused by the fused path; kept non-null for allocation checks
  d_found_ = dev_.alloc((1 + kMaxFound * kRecWords) * 4);
  d_wave_ = dev_.alloc(4);  // NVIDIA: wave start counter of the persistent GEMM
  if (!d_a_ || !d_bt_ || !d_ap_ || !d_bp_ || !d_uni_ || !d_pairs_ || !d_cvs_ || !d_out_ || !d_found_ || !d_wave_) {
    err = "out of GPU memory (" + dev_.last_error() + ")";
    return false;
  }
  if (!dev_.upload(d_a_, a_.data(), sa) || !dev_.upload(d_bt_, bt_.data(), sb)) {
    err = dev_.last_error();
    return false;
  }
  return check_gemm(err);
}

// Launches `kernel` with the GEMM's geometry for this GPU (AMD: workgroups of 2x4 waves; NVIDIA before Ampere: one
// 128x256 tile per block of 8 warps).
bool Engine::launch_gemm(const char* kernel, void** args, int M, int N) {
  if (dev_.info().vendor == gpu::Vendor::Nvidia)
    return dev_.launch(kernel, N / 256, M / 128, 1, 256, 1, 1, 0, args);
  return dev_.launch(kernel, N / (64 * amd_wnw_), M / (64 * amd_wmw_), 1, 32 * amd_wmw_ * amd_wnw_, 1, 1, 0, args);
}

// NVIDIA Ampere and newer: the persistent GEMM, one block per SM (8 math warps + 1 copy warp), 96 KB of shared memory.
bool Engine::launch_nv(const char* kernel, void** args, int M, int N, unsigned max_grid) {
  const unsigned tiles = (unsigned(M) / kNvBm) * (unsigned(N) / kNvBn), sms = static_cast<unsigned>(dev_.info().compute_units);
  unsigned grid = std::min(tiles, sms > 0 ? sms : 1u);
  if (max_grid) grid = std::min(grid, max_grid);
  const unsigned threads = (kNvBm / kNvWm) * (kNvBn / 64) * 32 + 32;
  const unsigned shared = kNvStages * (kNvBm + kNvBn) * kNvBk;
  return dev_.memset(d_wave_, 0, 4) && dev_.launch(kernel, grid, 1, 1, threads, 1, 1, shared, args);
}

// NVIDIA Ampere and newer have two GEMMs (kernels/pearl/pearl.cu): 2 stages of 128 k, or 4 of 64 k that read further
// ahead and keep their speed with a locked low memory clock (RTX 3060 at 100 W: 37.9 / 35.9 TH/s with memory at
// stock, 31.1 / 35.9 locked at 5001 MHz). Both run on random data on 2 blocks (several tiles each: the cross-tile
// pipeline and wave sync are exercised) and must agree; then both are timed, interleaved, on a slice of the committed
// matrices with the clocks as set, and the 64-k one is used if it wins every round by 2%. XM_PEARL_NV_BK=64 / 128
// forces one.
bool Engine::choose_nv_stages(std::string& err) {
  const char* force = std::getenv("XM_PEARL_NV_BK");
  if (force && std::atoi(force) == 128) return true;
  const std::string who = dev_.info().name + " (" + dev_.info().arch + ")";
  int M = 256, N = 512, K = 2048;
  const size_t sa = size_t(M) * K, sb = size_t(N) * K, words = size_t(M) * N / 64 * 16;
  gpu::DevPtr da = dev_.alloc(sa), db = dev_.alloc(sb), o1 = dev_.alloc(words * 4), o2 = dev_.alloc(words * 4);
  auto release = [&] {
    for (gpu::DevPtr p : {da, db, o1, o2}) dev_.free(p);
  };
  std::vector<int8_t> h(std::max(sa, sb));
  auto upload_random = [&](gpu::DevPtr p, size_t n) {
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<int8_t>(rng_());
    return dev_.upload(p, h.data(), n);
  };
  if (!da || !db || !o1 || !o2 || !upload_random(da, sa) || !upload_random(db, sb) || !dev_.memset(o1, 0, words * 4) ||
      !dev_.memset(o2, 0, words * 4)) {
    err = "Pearl kernel check: " + dev_.last_error();
    release();
    return false;
  }
  Key8 seed{};
  U256w bw{};
  uint32_t maxf = 0;
  void* g1[] = {&da, &db, &o1, &M, &N, &K, &seed, &bw, &d_found_, &maxf, &d_wave_};
  void* g2[] = {&da, &db, &o2, &M, &N, &K, &seed, &bw, &d_found_, &maxf, &d_wave_};
  if (!launch_nv("xm_pearl_gemm", g1, M, N, 2) || !dev_.sync()) {
    err = "Pearl kernel check: " + dev_.last_error();
    release();
    return false;
  }
  if (!launch_nv("xm_pearl_gemm_bk64", g2, M, N, 2)) {
    release();  // a module without the 64-k kernel (an older build for this GPU): keep the 128-k one
    return true;
  }
  if (!dev_.sync()) {  // the 64-k kernel ran and failed: a real fault, not a missing kernel
    err = "Pearl kernel check (64-k GEMM): " + dev_.last_error();
    release();
    return false;
  }
  std::vector<uint32_t> w1(words), w2(words);
  const bool got = dev_.download(w1.data(), o1, words * 4) && dev_.download(w2.data(), o2, words * 4);
  release();
  if (!got) {
    err = "Pearl kernel check: " + dev_.last_error();
    return false;
  }
  size_t bad = 0;
  for (size_t i = 0; i < words; ++i) bad += w1[i] != w2[i];
  if (bad) {
    warn("PEARL", who + ": the 64-k GEMM disagrees with the 128-k one (" + std::to_string(bad) + " of " +
                      std::to_string(words) + " words); using the 128-k one. Please report this.");
    return true;
  }
  if (force && std::atoi(force) == 64) {
    nv_gemm_ = "xm_pearl_gemm_bk64";
    return true;
  }
  // Speed on a slice of the committed matrices (signal values only, so a little less power than mining on matrix +
  // noise; what decides here is memory latency, which both see the same). Warm-up, then 4 interleaved rounds of
  // ~150 ms per kernel; the 64-k kernel must win every round.
  int Mt = static_cast<int>(std::min<uint32_t>(m_, 4096)), Nt = static_cast<int>(std::min<uint32_t>(n_, 16384)),
      Kt = static_cast<int>(cfg_.k);
  gpu::DevPtr no_out = 0;
  void* t[] = {&d_a_, &d_bt_, &no_out, &Mt, &Nt, &Kt, &seed, &bw, &d_found_, &maxf, &d_wave_};
  auto run_ms = [&](const char* kernel, int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      if (!launch_nv(kernel, t, Mt, Nt)) return -1.0;
    if (!dev_.sync()) return -1.0;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  };
  const double warm = std::max(run_ms("xm_pearl_gemm", 1), run_ms("xm_pearl_gemm_bk64", 1));
  if (warm < 0) {
    err = "Pearl kernel check: " + dev_.last_error();
    return false;
  }
  const int reps = std::clamp(static_cast<int>(150.0 / std::max(warm, 0.01)), 1, 200);
  double t128 = 0, t64 = 0;
  bool wins = true;
  for (int round = 0; round < 4; ++round) {
    const double a = run_ms("xm_pearl_gemm", reps), b = run_ms("xm_pearl_gemm_bk64", reps);
    if (a < 0 || b < 0) {
      err = "Pearl kernel check: " + dev_.last_error();
      return false;
    }
    t128 += a, t64 += b;
    wins = wins && b < 0.98 * a;
  }
  t128 /= 4, t64 /= 4;
  if (wins) {
    nv_gemm_ = "xm_pearl_gemm_bk64";
    char buf[160];
    std::snprintf(buf, sizeof buf, ": the 64-k GEMM is faster with this memory clock (%.1f vs %.1f ms); using it", t64,
                  t128);
    info("PEARL", who + buf);
  }
  return true;
}

// GEMM kernels we could not run on their GPUs ourselves come with xm_pearl_gemm_simt (verified by forcing it on our
// cards): AMD RDNA3 WMMA and NVIDIA Turing m8n8k16 (which GTX 16 GPUs, without tensor cores, may run slowly). Here
// both run on random data; if they differ the miner uses the SIMT kernel, if they agree the faster one.
// XM_PEARL_CHECK=1 runs this on any GPU whose module has both kernels, 0 skips it. Only a launch or memory error fails.
bool Engine::check_gemm(std::string& err) {
  const bool nv = dev_.info().vendor == gpu::Vendor::Nvidia;
  if (const char* e = std::getenv("XM_PEARL_AMD_WAVES")) std::sscanf(e, "%ux%u", &amd_wmw_, &amd_wnw_);
  if (nv && !nv_simt()) return choose_nv_stages(err);
  const char* env = std::getenv("XM_PEARL_CHECK");
  const std::string& arch = dev_.info().arch;
  const bool untested = nv ? arch == "sm_75" : arch.rfind("gfx11", 0) == 0;
  if (env ? std::atoi(env) == 0 : !untested) return true;
  const std::string who = dev_.info().name + " (" + arch + ")";

  // 1. Correctness: two blocks each way (covers the rasterization), K 2048 = 16 folds, full-range random int8.
  int M = nv ? 256 : static_cast<int>(128 * amd_wmw_), N = nv ? 512 : static_cast<int>(128 * amd_wnw_), K = 2048;
  // Debug transcripts: AMD one per lane in launch order; NVIDIA at a sparse index over M N / 64 slots (pearl.cu).
  const size_t sa = size_t(M) * K, sb = size_t(N) * K, words = size_t(M) * N / (nv ? 64 : 128) * 16;
  gpu::DevPtr da = dev_.alloc(sa), db = dev_.alloc(sb), o1 = dev_.alloc(words * 4), o2 = dev_.alloc(words * 4);
  auto release = [&] {
    for (gpu::DevPtr p : {da, db, o1, o2}) dev_.free(p);
  };
  std::vector<int8_t> h(std::max(sa, sb));
  auto upload_random = [&](gpu::DevPtr p, size_t n) {
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<int8_t>(rng_());
    return dev_.upload(p, h.data(), n);
  };
  if (!da || !db || !o1 || !o2 || !upload_random(da, sa) || !upload_random(db, sb) || !dev_.memset(o1, 0, words * 4) ||
      !dev_.memset(o2, 0, words * 4)) {
    err = "Pearl kernel check: " + dev_.last_error();
    release();
    return false;
  }
  Key8 seed{};
  U256w bw{};
  uint32_t maxf = 0;
  void* g1[] = {&da, &db, &o1, &M, &N, &K, &seed, &bw, &d_found_, &maxf, &d_wave_};
  void* g2[] = {&da, &db, &o2, &M, &N, &K, &seed, &bw, &d_found_, &maxf, &d_wave_};
  if (!launch_gemm(gemm_, g1, M, N) || !dev_.sync()) {
    err = "Pearl kernel check: " + dev_.last_error();
    release();
    return false;
  }
  if (!launch_gemm("xm_pearl_gemm_simt", g2, M, N) || !dev_.sync()) {
    release();  // a module without the SIMT entry point: nothing to compare with
    return true;
  }
  std::vector<uint32_t> w1(words), w2(words);
  const bool got = dev_.download(w1.data(), o1, words * 4) && dev_.download(w2.data(), o2, words * 4);
  release();
  if (!got) {
    err = "Pearl kernel check: " + dev_.last_error();
    return false;
  }
  size_t bad = 0;
  for (size_t i = 0; i < words; ++i) bad += w1[i] != w2[i];
  const char* fast = nv ? "tensor-core" : "WMMA";
  if (bad) {
    warn("PEARL", who + ": the " + fast + " kernel disagrees with the SIMT kernel (" + std::to_string(bad) + " of " +
                      std::to_string(words) + " words); mining with the SIMT kernel. Please report this.");
    gemm_ = "xm_pearl_gemm_simt";
    return true;
  }

  // 2. Speed: a slice of the mining matrices (16 folds of 128 x 256 tiles), best of three runs each, no hits.
  int Mt = static_cast<int>(std::min<uint32_t>(m_, 4096)), Nt = static_cast<int>(std::min<uint32_t>(n_, 8192)),
      Kt = static_cast<int>(cfg_.k);
  gpu::DevPtr no_out = 0;
  void* t1[] = {&d_ap_, &d_bp_, &no_out, &Mt, &Nt, &Kt, &seed, &bw, &d_found_, &maxf, &d_wave_};
  auto best_ms = [&](const char* kernel) {
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      if (!launch_gemm(kernel, t1, Mt, Nt) || !dev_.sync()) return -1.0;
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
  };
  const double t_fast = best_ms(gemm_), t_simt = best_ms("xm_pearl_gemm_simt");
  if (t_fast < 0 || t_simt < 0) {
    err = "Pearl kernel check: " + dev_.last_error();
    return false;
  }
  char buf[160];
  if (t_simt < 0.9 * t_fast) {
    std::snprintf(buf, sizeof buf, ": the SIMT kernel is faster than the %s kernel (%.1f vs %.1f ms); using it", fast,
                  t_simt, t_fast);
    info("PEARL", who + buf);
    gemm_ = "xm_pearl_gemm_simt";
  } else {
    std::snprintf(buf, sizeof buf, ": %s kernel matches the SIMT kernel and is faster (%.1f vs %.1f ms)", fast,
                  t_fast, t_simt);
    info("PEARL", who + buf);
  }
  return true;
}
bool Engine::chunk_tree(gpu::DevPtr data, size_t len, MerkleTree& out, std::string& err) {
  uint32_t chunks = static_cast<uint32_t>(len / 1024);
  Key8 key = key8(job_key_);
  void* args[] = {&data, &chunks, &key, &d_cvs_};
  if (!dev_.launch("xm_b3_chunk_cvs", blocks(chunks, 128), 1, 1, 128, 1, 1, 0, args) || !dev_.sync()) {
    err = dev_.last_error();
    return false;
  }
  std::vector<uint32_t> w(size_t(chunks) * 8);
  if (!dev_.download(w.data(), d_cvs_, w.size() * 4)) {
    err = dev_.last_error();
    return false;
  }
  std::vector<Hash> cvs(chunks);
  for (uint32_t c = 0; c < chunks; ++c) {
    blake3::Cv cv;
    for (int i = 0; i < 8; ++i) cv[i] = w[size_t(c) * 8 + i];
    cvs[c] = blake3::cv_bytes(cv);
  }
  out = tree_from_chunk_cvs(job_key_, std::move(cvs));
  return true;
}

bool Engine::launch_noise(bool side_b, const Hash& seed, gpu::DevPtr sig, gpu::DevPtr packed, uint32_t rows,
                          std::string& err) {
  Key8 key = key8(seed);
  uint32_t sb = side_b ? 1 : 0, k = cfg_.k, r = cfg_.rank;
  uint32_t nhash = rows * r / 32;
  void* a1[] = {&key, &sb, &nhash, &d_uni_};
  void* a2[] = {&key, &sb, &k, &r, &d_pairs_};
  void* a3[] = {&sig, &d_uni_, &d_pairs_, &packed, &rows, &k, &r};
  bool ok = dev_.launch("xm_noise_uniform", blocks(nhash, 256), 1, 1, 256, 1, 1, 0, a1) &&
            dev_.launch("xm_noise_perm", blocks(k / 8, 128), 1, 1, 128, 1, 1, 0, a2) &&
            dev_.launch("xm_apply_noise", blocks(size_t(rows) * (k / 8), 256), 1, 1, 256, 1, 1, 0, a3);
  if (!ok) err = dev_.last_error();
  return ok;
}

bool Engine::set_job(const Job& job, std::string& err) {
  job_ = job;
  job_key_ = job_key(job.header, cfg_);
  const size_t sa = size_t(m_) * cfg_.k, sb = size_t(n_) * cfg_.k;
  if (!chunk_tree(d_a_, sa, tree_a_, err) || !chunk_tree(d_bt_, sb, tree_b_, err)) return false;
  // B's seed depends only on the job key and B's commitment, so B' is built once per job.
  seeds_ = commitment_seeds(job_key_, tree_a_.root(), tree_b_.root(), m_, n_, job.salted);
  if (!launch_noise(true, seeds_.b_noise_seed, d_bt_, d_bp_, n_, err) || !dev_.sync()) {
    if (err.empty()) err = dev_.last_error();
    return false;
  }
  have_job_ = true;
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  const uint32_t k = cfg_.k;
  // New nonce chunk: changes A's commitment and therefore both noise seeds' A side.
  nonce_chunk_ = (nonce_chunk_ + 1) % tree_a_.leaves();
  int8_t* chunk = a_.data() + nonce_chunk_ * 1024;
  fill_random(chunk, 1024);
  if (!dev_.upload(d_a_ + nonce_chunk_ * 1024, chunk, 1024)) {
    err = dev_.last_error();
    return 0;
  }
  tree_update_chunk(tree_a_, nonce_chunk_, reinterpret_cast<const uint8_t*>(chunk));
  seeds_ = commitment_seeds(job_key_, tree_a_.root(), tree_b_.root(), m_, n_, job_.salted);

  static const bool prof = std::getenv("XM_PROFILE") != nullptr;
  auto tp = [&](const char* what, std::chrono::steady_clock::time_point& t) {
    if (!prof) return;
    dev_.sync();
    auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "%s %.3f ms  ", what, std::chrono::duration<double, std::milli>(now - t).count());
    t = now;
  };
  auto t_prof = std::chrono::steady_clock::now();
  tp("cpu", t_prof);
  if (!launch_noise(false, seeds_.a_noise_seed, d_a_, d_ap_, m_, err)) return 0;
  tp("noiseA", t_prof);
  int M = static_cast<int>(m_), N = static_cast<int>(n_), K = static_cast<int>(k);
  uint32_t maxf = kMaxFound;
  Key8 seed = key8(seeds_.a_noise_seed);
  U256 bound = difficulty_bound(job_.share_nbits, cfg_);
  U256w bw;
  for (int i = 0; i < 8; ++i) bw.w[i] = bound.w[i];
  gpu::DevPtr no_out = 0;  // fused jackpot check in the GEMM epilogue
  void* g[] = {&d_ap_, &d_bp_, &no_out, &M, &N, &K, &seed, &bw, &d_found_, &maxf, &d_wave_};
  std::vector<uint32_t> found(1 + kMaxFound * kRecWords);
  // AMD: 128x256 blocks of 8 waves. NVIDIA: persistent, one block per SM (8 math warps + 1 copy warp) walking
  // 128x256 tiles, 96 KB of dynamic shared memory.
  const bool nv = dev_.info().vendor == gpu::Vendor::Nvidia;
  bool ok = dev_.memset(d_found_, 0, 4);
  if (!nv || nv_simt()) {
    // AMD, and NVIDIA Pascal / Volta / Turing (dp4a or m8n8k16, one 128x256 tile per block of 8 warps).
    ok = ok && launch_gemm(gemm_, g, M, N);
  } else {
    ok = ok && launch_nv(nv_gemm_, g, M, N);
  }
  if (!ok) {
    err = dev_.last_error();
    return 0;
  }
  // Blocking wait first: the download alone would let the runtime spin a CPU core for the whole pass.
  if (!dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  tp("gemm+check", t_prof);
  if (prof) std::fputc('\n', stderr);
  uint32_t hits = std::min(found[0], kMaxFound);
  if (hits) {
    U256 block_bound = difficulty_bound(header_nbits(job_.header), cfg_);
    auto perm_a = permutation(Side::A, seeds_.a_noise_seed, k, cfg_.rank);
    auto perm_b = permutation(Side::B, seeds_.b_noise_seed, k, cfg_.rank);
    const auto rl = cfg_.rows.to_list(), cl = cfg_.cols.to_list();
    for (uint32_t h = 0; h < hits; ++h) {
      const uint32_t* rec = &found[1 + h * kRecWords];
      Found f;
      f.t_rows = rec[0];
      f.t_cols = rec[1];
      for (int w = 0; w < 16; ++w) f.transcript[w] = rec[2 + w];
      if (!cfg_.rows.offset_is_valid(f.t_rows) || !cfg_.cols.offset_is_valid(f.t_cols) ||
          f.t_rows + cfg_.rows.max() >= m_ || f.t_cols + cfg_.cols.max() >= n_) {
        ++cpu_rejects_;
        continue;
      }
      // Re-verify on the CPU reference: the transcript of this tile from the host matrices and noise.
      std::vector<std::vector<int32_t>> ar, bc;
      for (uint32_t d : rl) {
        auto ea = noise_line(uniform_row(Side::A, seeds_.a_noise_seed, f.t_rows + d, cfg_.rank), perm_a);
        std::vector<int32_t> ra(k);
        for (uint32_t l = 0; l < k; ++l) ra[l] = a_[size_t(f.t_rows + d) * k + l] + ea[l];
        ar.push_back(std::move(ra));
      }
      for (uint32_t d : cl) {
        auto eb = noise_line(uniform_row(Side::B, seeds_.b_noise_seed, f.t_cols + d, cfg_.rank), perm_b);
        std::vector<int32_t> rb(k);
        for (uint32_t l = 0; l < k; ++l) rb[l] = bt_[size_t(f.t_cols + d) * k + l] + eb[l];
        bc.push_back(std::move(rb));
      }
      auto cpu_t = jackpot(ar, bc, k, cfg_.rank);
      f.hash = jackpot_hash(cpu_t, seeds_.a_noise_seed);
      if (cpu_t != f.transcript || !(U256::from_le_bytes(f.hash.data()) <= bound)) {
        ++cpu_rejects_;
        continue;
      }
      PlainProof p;
      p.m = m_;
      p.n = n_;
      p.k = k;
      p.noise_rank = cfg_.rank;
      for (uint32_t d : rl) p.a_rows.push_back(f.t_rows + d);
      for (uint32_t d : cl) p.b_cols.push_back(f.t_cols + d);
      p.a = tree_proof(tree_a_, a_.data(), a_.size(), k, p.a_rows);
      p.bt = tree_proof(tree_b_, bt_.data(), bt_.size(), k, p.b_cols);
      Share s;
      s.job_id = job_.id;
      s.found = f;
      s.proof = serialize(p);
      s.proof_b64 = base64(s.proof);
      s.is_block = U256::from_le_bytes(f.hash.data()) <= block_bound;
      {
        const double mant = job_.share_nbits & 0x7fffff;
        const int e = static_cast<int>(job_.share_nbits >> 24);
        s.work = mant > 0 ? std::ldexp(1.0, 256 - 8 * (e - 3)) / mant : 0.0;
      }
      shares.push_back(std::move(s));
    }
  }
  return double(m_) * double(n_) * double(k);
}

}  // namespace xm::pearl
