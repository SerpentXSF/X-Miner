#include "algo/c29/c29_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <sstream>

#include "c29_cu.hpp"

namespace xm::c29 {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct KeysArg {
  uint64_t k0, k1, k2, k3;
};

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() {
  for (gpu::DevPtr p : {d_a_, d_b_, d_cnt_a_, d_cnt_b_, d_off_, d_n_, d_bm1_, d_bm2_, d_want_, d_found_}) dev_.free(p);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_C29_ROUNDS")) rounds_ = std::clamp(std::atoi(e), 2, 1000);  // developer
  if (const char* e = std::getenv("XM_C29_FLAT")) flat_edges_ = std::strtoull(e, nullptr, 0);  // developer
  if (const char* e = std::getenv("XM_C29_SEGS")) seg_edges_ = std::strtoull(e, nullptr, 0);   // developer
  // Edge writes: NVIDIA stages them in shared memory and writes contiguous runs (RTX 3060 2.47 -> 3.00 g/s); AMD
  // writes them straight out, where the Infinity Cache merges scattered stores (RX 9070 XT: staged 6.30 vs direct
  // 6.67 g/s). Developer: XM_C29_DIRECT=0/1, XM_C29_GEN_TPB.
  direct_ = dev_.info().vendor == gpu::Vendor::Amd;
  if (const char* e = std::getenv("XM_C29_DIRECT")) direct_ = std::atoi(e) != 0;
  if (direct_) gen_tpb_ = kTpb;
  if (const char* e = std::getenv("XM_C29_GEN_TPB")) gen_tpb_ = uint32_t(std::atoi(e));        // developer
  const uint64_t per_bucket = kEdges >> kNbBits;
  capb_ = uint32_t(per_bucket + per_bucket / 25 + 512) & ~1u;  // +4% (the spread is ~0.3% = 1 sd); even: the
                                                                // kernels read buckets with 16-byte loads
  const size_t edge_bytes = size_t(capb_) << kNbBits << 3;
  d_a_ = dev_.alloc(edge_bytes);
  d_b_ = dev_.alloc(edge_bytes);
  d_cnt_a_ = dev_.alloc(4 << kNbBits);
  d_cnt_b_ = dev_.alloc(size_t(4) * kNc << kNbBits);  // coarse segment counts
  d_off_ = dev_.alloc(4 << kNbBits);
  d_n_ = dev_.alloc(8);
  d_bm1_ = dev_.alloc(kEdges / 8);
  d_bm2_ = dev_.alloc(kEdges / 8);
  d_want_ = dev_.alloc(kProofSize * 8);
  d_found_ = dev_.alloc(kProofSize * 4);
  if (!d_a_ || !d_b_ || !d_cnt_a_ || !d_cnt_b_ || !d_off_ || !d_n_ || !d_bm1_ || !d_bm2_ || !d_want_ || !d_found_) {
    err = "not enough GPU memory for Cuckaroo29 (needs ~" + std::to_string((2 * edge_bytes + kEdges / 4) >> 20) +
          " MB): " + dev_.last_error();
    return false;
  }
  std::string kernel(reinterpret_cast<const char*>(c29_cu_bytes), c29_cu_size);
  if (const char* f = std::getenv("XM_C29_KERNEL")) {  // developer: kernel source from a file
    std::ifstream in(f, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    if (!in) {
      err = std::string("XM_C29_KERNEL ") + f + " not readable";
      return false;
    }
    kernel = ss.str();
  }
  const std::string src = "#define NB_BITS " + std::to_string(kNbBits) + "\n#define CAPB " + std::to_string(capb_) +
                          "u\n#define TPB " + std::to_string(kTpb) + "\n#define GEN_SB " + std::to_string(kGenSb) +
                          "\n#define GEN_TPB " + std::to_string(gen_tpb_) + "\n" +
                          (direct_ ? "#define GEN_DIRECT 1\n#define SPLIT_DIRECT 1\n" : "") + kernel;
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  if (!gpu::rtc_compile(dev_.info(), src, "c29.cu", opts, module_)) {
    err = "Cuckaroo29 kernel did not compile: " + module_.log.substr(0, 3000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "c29"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::trim(const Keys& k, Edges& edges, SolveStats& st, std::string& err) {
  st = SolveStats{};
  edges.clear();
  const auto t_start = std::chrono::steady_clock::now();
  KeysArg ka{k[0], k[1], k[2], k[3]};
  constexpr uint32_t nb = 1u << kNbBits;
  auto fail = [&] {
    err = dev_.last_error();
    return false;
  };
  static const bool prof = std::getenv("XM_C29_PROF") != nullptr;  // developer: per-round GPU times
  auto mark = [&](const char* what, int r) {
    if (!prof) return;
    dev_.wait();
    std::fprintf(stderr, "c29 %s %d: %.2f ms\n", what, r, ms_since(t_start));
  };
  // generation into the coarse U buckets (d_b_, segment counts in d_cnt_b_), then split into the fine ones (d_a_)
  gpu::DevPtr in = d_a_, out = d_b_;
  gpu::DevPtr cin = d_cnt_a_;
  const uint32_t gen_blocks = uint32_t((kEdges / 64) / (uint64_t(kTpb) * kGenSb));
  const uint32_t ccap = capb_ * (nb / kNc);
  // split blocks: slices of ~kSplitSlice edges (the second read of a slice then hits the cache), <= kMaxSegs segments
  auto split = [&](uint32_t side, uint64_t edges, uint32_t nseg) {
    uint32_t seg_cap = ccap / nseg;
    uint32_t parts = 1;  // a power of two, as nseg is (more parts than segments: slices of one segment)
    while (parts < kMaxParts && (parts * 2 * kNc * kSplitSlice <= edges || nseg / parts > kMaxSegs))
      parts *= 2;
    void* args[] = {&side, &d_b_, &d_cnt_b_, &nseg, &seg_cap, &d_a_, &d_cnt_a_, &parts};
    return dev_.memset(d_cnt_a_, 0, 4 * nb) && dev_.launch("c29_split", kNc * parts, 1, 1, kTpb, 1, 1, 0, args);
  };
  {
    void* args[] = {&ka, &d_b_, &d_cnt_b_};
    if (!dev_.launch("c29_gen", gen_blocks, 1, 1, gen_tpb_, 1, 1, 0, args)) return fail();
    mark("  gen kernel", 0);
    if (!split(0, kEdges, gen_blocks)) return fail();
  }
  mark("gen", 0);
  // bucket rounds (shared-memory degree counts) while edges are many
  std::vector<uint32_t> counts(nb);
  uint64_t total = kEdges;
  int r = 0;
  for (; r < rounds_ && total >= flat_edges_; ++r) {
    uint32_t side = uint32_t(r & 1);
    void* args[] = {&side, &d_a_, &d_cnt_a_, &d_b_, &d_cnt_b_};
    const bool segs = total >= seg_edges_;  // per-block segments while edges are many, else whole coarse buckets
    if ((!segs && !dev_.memset(d_cnt_b_, 0, 4 * kNc)) ||
        !dev_.launch(segs ? "c29_round" : "c29_round_r", nb, 1, 1, kTpb, 1, 1, 0, args))
      return fail();
    if (r < 4) mark("  round kernel", r);  // (profile mode: the first rounds' kernel and split separately)
    if (!split(side ^ 1, total, segs ? nb : 1)) return fail();
    if (r < 4) mark("  split after round", r);
    // counts come back every 4th round (a host round trip each); in between `total` is an upper bound
    if ((r & 3) != 3 && r + 1 < rounds_) continue;
    if (!dev_.download(counts.data(), cin, 4 * nb)) return fail();
    total = 0;
    for (uint32_t c : counts) {
      if (c > capb_) st.overflow = true;
      total += std::min(c, capb_);
    }
    mark("round", r);
    if (prof) std::fprintf(stderr, "c29   edges after round %d: %llu\n", r, static_cast<unsigned long long>(total));
  }
  if (st.overflow) return true;
  // one flat list for the late rounds (device-side count; the global bitmaps are cleared through the edges)
  {
    std::vector<uint32_t> off(nb);
    uint32_t o = 0;
    for (uint32_t b = 0; b < nb; ++b) {
      off[b] = o;
      o += counts[b];
    }
    const uint32_t n2[2] = {o, 0};
    void* args[] = {&in, &cin, &d_off_, &out};
    if (!dev_.upload(d_off_, off.data(), 4 * nb) || !dev_.upload(d_n_, n2, 8) ||
        !dev_.launch("c29_flatten", nb, 1, 1, kTpb, 1, 1, 0, args) || !dev_.memset(d_bm1_, 0, kEdges / 8) ||
        !dev_.memset(d_bm2_, 0, kEdges / 8))
      return fail();
    std::swap(in, out);
  }
  gpu::DevPtr nin = d_n_, nout = d_n_ + 4;
  constexpr unsigned fgrid = 256;
  for (; r < rounds_; ++r) {
    uint32_t side = uint32_t(r & 1);
    void* ca[] = {&side, &in, &nin, &d_bm1_, &d_bm2_};
    void* fa[] = {&side, &in, &nin, &d_bm2_, &out, &nout};
    if (!dev_.memset(nout, 0, 4) || !dev_.launch("c29_fcount", fgrid, 1, 1, kTpb, 1, 1, 0, ca) ||
        !dev_.launch("c29_ffilter", fgrid, 1, 1, kTpb, 1, 1, 0, fa) ||
        !dev_.launch("c29_fclear", fgrid, 1, 1, kTpb, 1, 1, 0, ca))
      return fail();
    std::swap(in, out);
    std::swap(nin, nout);
    if (r % 50 == 0) mark("flat round", r);
  }
  // survivors to the host
  uint32_t n = 0;
  if (!dev_.wait() || !dev_.download(&n, nin, 4)) return fail();
  st.gpu_ms = ms_since(t_start);
  st.remaining = n;
  if (n > (1u << 22)) {  // trimming far from done: not worth a CPU search
    st.overflow = true;
    return true;
  }
  std::vector<uint32_t> buf(size_t(n) * 2);
  if (n && !dev_.download(buf.data(), in, buf.size() * 4)) return fail();
  edges.resize(n);
  for (uint32_t i = 0; i < n; ++i) edges[i] = {buf[2 * i], buf[2 * i + 1]};
  return true;
}

bool Engine::finish(const Keys& k, const Edges& edges, const Cycles& cycles, std::vector<Proof>& proofs,
                    std::string& err) {
  KeysArg ka{k[0], k[1], k[2], k[3]};
  for (const auto& c : cycles) {
    uint32_t want[2 * kProofSize];
    for (int i = 0; i < kProofSize; ++i) {
      want[2 * i] = edges[c[i]].first;
      want[2 * i + 1] = edges[c[i]].second;
    }
    uint32_t nwant = kProofSize;
    void* ra[] = {&ka, &d_want_, &nwant, &d_found_};
    uint32_t idx[kProofSize];
    if (!dev_.upload(d_want_, want, sizeof want) || !dev_.memset(d_found_, 0xff, kProofSize * 4) ||
        !dev_.launch("c29_recover", unsigned((kEdges / 64) / kTpb), 1, 1, kTpb, 1, 1, 0, ra) || !dev_.wait() ||
        !dev_.download(idx, d_found_, sizeof idx)) {
      err = dev_.last_error();
      return false;
    }
    Proof p{};
    std::copy(idx, idx + kProofSize, p.begin());
    std::sort(p.begin(), p.end());
    if (verify(k, p)) proofs.push_back(p);
    else ++cpu_rejects_;
  }
  return true;
}

bool Engine::solve(const Keys& k, std::vector<Proof>& proofs, SolveStats& st, std::string& err) {
  Edges edges;
  if (!trim(k, edges, st, err)) return false;
  const auto t_cpu = std::chrono::steady_clock::now();
  const Cycles cycles = find_cycles(edges);
  st.cpu_ms = ms_since(t_cpu);
  st.cycles = uint32_t(cycles.size());
  return finish(k, edges, cycles, proofs, err);
}

bool Engine::solve_pipelined(const Keys& k, uint64_t tag, std::vector<Proof>& proofs, uint64_t& prev_tag,
                             bool& have_prev, SolveStats& st, std::string& err) {
  have_prev = false;
  Edges edges;
  if (!trim(k, edges, st, err)) return false;  // the previous graph's cycle search runs meanwhile
  if (pend_.active) {
    pend_.active = false;
    const auto [cycles, cpu_ms] = pend_.search.get();
    st.cpu_ms = cpu_ms;
    st.cycles = uint32_t(cycles.size());
    if (!finish(pend_.keys, pend_.edges, cycles, proofs, err)) return false;
    have_prev = true;
    prev_tag = pend_.tag;
  }
  if (edges.empty()) return true;
  pend_.keys = k;
  pend_.tag = tag;
  pend_.edges = std::move(edges);
  pend_.active = true;
  pend_.search = std::async(std::launch::async, [e = &pend_.edges] {
    const auto t = std::chrono::steady_clock::now();
    Cycles c = find_cycles(*e);
    return std::pair<Cycles, double>(std::move(c), ms_since(t));
  });
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  if (!have_job_ || job.mining_hash != job_.mining_hash || job.nonce_prefix != job_.nonce_prefix) {
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
  // graphs are pipelined: this call trims a new graph and returns the shares of the one before
  const uint64_t nonce = job_.nonce_prefix | (counter_++ & ~job_.prefix_mask);
  const Job prev_job = pend_job_;
  pend_job_ = job_;
  std::vector<Proof> proofs;
  SolveStats st;
  uint64_t prev_nonce = 0;
  bool have_prev = false;
  if (!solve_pipelined(sip_keys(job_.mining_hash, nonce), nonce, proofs, prev_nonce, have_prev, st, err)) return 0;
  if (!have_prev || prev_job.mining_hash != job_.mining_hash) return 1;  // a graph of an older block: stale
  for (const auto& p : proofs) {
    Share s;
    s.job_id = prev_job.id;
    s.nonce = prev_nonce;
    s.proof = p;
    s.result = result(p);
    if (share_difficulty(s.result) < prev_job.difficulty) continue;
    s.work = 42.0 * double(prev_job.difficulty);
    shares.push_back(s);
  }
  return 1;
}

}  // namespace xm::c29
