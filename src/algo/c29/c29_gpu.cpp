#include "algo/c29/c29_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
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
  for (gpu::DevPtr p : {d_a_, d_b_, d_cnt_a_, d_cnt_b_, d_bm1_, d_bm2_, d_want_, d_found_}) dev_.free(p);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_C29_ROUNDS")) rounds_ = std::clamp(std::atoi(e), 2, 1000);  // developer
  const uint64_t per_bucket = kEdges >> kNbBits;
  capb_ = uint32_t(per_bucket + per_bucket / 50 + 4096);  // +2% (the spread is ~0.04%)
  chunks_ = (capb_ + kChunk - 1) / kChunk;
  const size_t edge_bytes = size_t(capb_) << kNbBits << 3;
  d_a_ = dev_.alloc(edge_bytes);
  d_b_ = dev_.alloc(edge_bytes);
  d_cnt_a_ = dev_.alloc(4 << kNbBits);
  d_cnt_b_ = dev_.alloc(4 << kNbBits);
  d_bm1_ = dev_.alloc(kEdges / 8);
  d_bm2_ = dev_.alloc(kEdges / 8);
  d_want_ = dev_.alloc(kProofSize * 8);
  d_found_ = dev_.alloc(kProofSize * 4);
  if (!d_a_ || !d_b_ || !d_cnt_a_ || !d_cnt_b_ || !d_bm1_ || !d_bm2_ || !d_want_ || !d_found_) {
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
                          "u\n#define TPB " + std::to_string(kTpb) + "\n#define CHUNK " + std::to_string(kChunk) +
                          "u\n#define CHUNKS " + std::to_string(chunks_) + "u\n" + kernel;
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

bool Engine::solve(const Keys& k, std::vector<Proof>& proofs, SolveStats& st, std::string& err) {
  st = SolveStats{};
  const auto t_start = std::chrono::steady_clock::now();
  KeysArg ka{k[0], k[1], k[2], k[3]};
  const unsigned gen_blocks = unsigned((kEdges / 64) / kTpb);
  auto fail = [&] {
    err = dev_.last_error();
    return false;
  };
  if (!dev_.memset(d_cnt_a_, 0, 4 << kNbBits) || !dev_.memset(d_bm1_, 0, kEdges / 8) ||
      !dev_.memset(d_bm2_, 0, kEdges / 8))
    return fail();
  {
    void* args[] = {&ka, &d_a_, &d_cnt_a_};
    if (!dev_.launch("c29_gen", gen_blocks, 1, 1, kTpb, 1, 1, 0, args)) return fail();
  }
  static const bool prof = std::getenv("XM_C29_PROF") != nullptr;  // developer: per-round GPU times
  auto mark = [&](const char* what, int r) {
    if (!prof) return;
    dev_.wait();
    std::fprintf(stderr, "c29 %s %d: %.2f ms\n", what, r, ms_since(t_start));
  };
  mark("gen", 0);
  gpu::DevPtr in = d_a_, out = d_b_, cin = d_cnt_a_, cout = d_cnt_b_;
  // A round counts the degrees of side r & 1 (bucket by bucket, so the bitmap range in use stays in the cache), then
  // keeps the edges whose node was seen twice. The bitmaps are cleared whole while edges are many, word by word
  // through the edges once few are left. The grid follows the largest bucket: counts are read back every 4th round
  // while edges are many, then not at all (counts only shrink, so an older grid still covers every bucket).
  uint32_t counts[1 << kNbBits];
  uint32_t chunks = chunks_, total = uint32_t(kEdges);
  for (int r = 0; r < rounds_; ++r) {
    uint32_t side = uint32_t(r & 1);
    const bool small = total < (1u << 22);
    if (!dev_.memset(cout, 0, 4 << kNbBits)) return fail();
    const unsigned blocks = chunks << kNbBits;
    void* ca[] = {&side, &in, &cin, &d_bm1_, &d_bm2_, &chunks};
    void* fa[] = {&side, &in, &cin, &d_bm2_, &out, &cout, &chunks};
    if (!dev_.launch("c29_count", blocks, 1, 1, kTpb, 1, 1, 0, ca) ||
        !dev_.launch("c29_filter", blocks, 1, 1, kTpb, 1, 1, 0, fa))
      return fail();
    if (small) {
      if (!dev_.launch("c29_clear", blocks, 1, 1, kTpb, 1, 1, 0, ca)) return fail();
    } else if (!dev_.memset(d_bm1_, 0, kEdges / 8) || !dev_.memset(d_bm2_, 0, kEdges / 8)) {
      return fail();
    }
    std::swap(in, out);
    std::swap(cin, cout);
    if (r < 24 || r % 50 == 0) mark("round", r);
    if (small || (r & 3) != 3) continue;
    if (!dev_.download(counts, cin, sizeof counts)) return fail();
    uint32_t mx = 0;
    total = 0;
    for (uint32_t c : counts) {
      mx = std::max(mx, c);
      total += c;
    }
    chunks = std::max(1u, std::min(chunks_, (mx + kChunk - 1) / kChunk));
  }
  st.gpu_ms = ms_since(t_start);
  // survivors to the host
  if (!dev_.wait() || !dev_.download(counts, cin, sizeof counts)) return fail();
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (int b = 0; b < (1 << kNbBits); ++b) {
    if (counts[b] > capb_) {
      st.overflow = true;
      return true;
    }
    st.remaining += counts[b];
  }
  if (st.remaining > (1u << 22)) {  // trimming far from done: not worth a CPU search
    st.overflow = true;
    return true;
  }
  std::vector<uint32_t> buf;
  for (int b = 0; b < (1 << kNbBits); ++b) {
    if (!counts[b]) continue;
    buf.resize(size_t(counts[b]) * 2);
    if (!dev_.download(buf.data(), in + gpu::DevPtr(size_t(b) * capb_ * 8), buf.size() * 4)) return fail();
    for (uint32_t i = 0; i < counts[b]; ++i) edges.push_back({buf[2 * i], buf[2 * i + 1]});
  }
  const auto t_cpu = std::chrono::steady_clock::now();
  const auto cycles = find_cycles(edges);
  st.cpu_ms = ms_since(t_cpu);
  for (const auto& c : cycles) {
    ++st.cycles;
    uint32_t want[2 * kProofSize];
    for (int i = 0; i < kProofSize; ++i) {
      want[2 * i] = edges[c[i]].first;
      want[2 * i + 1] = edges[c[i]].second;
    }
    uint32_t nwant = kProofSize;
    if (!dev_.upload(d_want_, want, sizeof want) || !dev_.memset(d_found_, 0xff, kProofSize * 4)) return fail();
    void* ra[] = {&ka, &d_want_, &nwant, &d_found_};
    uint32_t idx[kProofSize];
    if (!dev_.launch("c29_recover", gen_blocks, 1, 1, kTpb, 1, 1, 0, ra) || !dev_.wait() ||
        !dev_.download(idx, d_found_, sizeof idx))
      return fail();
    Proof p{};
    std::copy(idx, idx + kProofSize, p.begin());
    std::sort(p.begin(), p.end());
    if (verify(k, p)) proofs.push_back(p);
    else ++cpu_rejects_;
  }
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
  const uint64_t nonce = job_.nonce_prefix | (counter_++ & ~job_.prefix_mask);
  std::vector<Proof> proofs;
  SolveStats st;
  if (!solve(sip_keys(job_.mining_hash, nonce), proofs, st, err)) return 0;
  for (const auto& p : proofs) {
    Share s;
    s.job_id = job_.id;
    s.nonce = nonce;
    s.proof = p;
    s.result = result(p);
    if (share_difficulty(s.result) < job_.difficulty) continue;
    s.work = 42.0 * double(job_.difficulty);
    shares.push_back(s);
  }
  return 1;
}

}  // namespace xm::c29
