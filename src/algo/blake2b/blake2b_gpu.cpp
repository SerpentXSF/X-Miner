#include "algo/blake2b/blake2b_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "gpu/kernels.hpp"

namespace xm::b2b {

namespace {

// Must match struct B2bJob in kernels/blake2b/blake2b.cu.
struct B2bJobArgs {
  uint64_t v[16];
  uint64_t m[10];
  uint64_t target_hi;
};

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

inline void g(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t x, uint64_t y) {
  a = a + b + x;
  d = rotr(d ^ a, 32);
  c = c + d;
  b = rotr(b ^ c, 24);
  a = a + b + y;
  d = rotr(d ^ a, 16);
  c = c + d;
  b = rotr(b ^ c, 63);
}

}  // namespace

void precompute(const Header& h, uint64_t m[10], uint64_t v[16]) {
  static const uint64_t iv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                                 0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                                 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  for (int i = 0; i < 10; ++i) {
    m[i] = 0;
    for (int b = 0; b < 8; ++b) m[i] |= uint64_t(h[8 * i + b]) << (8 * b);
  }
  for (int i = 0; i < 8; ++i) v[i] = v[8 + i] = iv[i];
  v[0] ^= 0x01010020ull;
  v[12] ^= 80;  // 80 bytes, final block
  v[14] = ~v[14];
  g(v[0], v[4], v[8], v[12], m[0], m[1]);
  g(v[1], v[5], v[9], v[13], m[2], m[3]);
  g(v[3], v[7], v[11], v[15], m[6], m[7]);
}

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() { dev_.free(d_found_); }

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_B2B_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (!gpu::load_kernels(dev_, "blake2b", err)) return false;
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  job_ = job;
  root_ = work_root(job.coinb1.data(), job.extranonce.data());
  precompute(work_header(job.prevhash.data(), 0, job.ntime.data(), root_), m_, v_);
  nonce_ = splitmix(rng_);  // a fresh random start: GPUs sharing a session never overlap in practice
  have_job_ = true;
  have_block_ = std::any_of(job.block_target.begin(), job.block_target.end(), [](uint8_t b) { return b != 0; });
  share_work_ = target_work(job.share_target);
  return true;
}

double Engine::iterate(uint32_t batch, std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  B2bJobArgs a{};
  std::memcpy(a.v, v_, sizeof a.v);
  std::memcpy(a.m, m_, sizeof a.m);
  for (int i = 0; i < 8; ++i) a.target_hi |= uint64_t(job_.share_target[24 + i]) << (8 * i);
  uint64_t base = nonce_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&a, &base, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_blake2b_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args)) {
    err = dev_.last_error();
    return 0;
  }
  // Blocking wait first: the download alone would let the runtime spin a CPU core for the whole batch.
  if (!dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  nonce_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    const uint64_t nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    const Header h = work_header(job_.prevhash.data(), nonce, job_.ntime.data(), root_);
    const Digest d = blake2b256(h.data(), h.size());
    if (!meets_target(d, job_.share_target)) {
      ++cpu_rejects_;  // the GPU filter only compares the top 64 bits
      continue;
    }
    Share s;
    s.job_id = job_.id;
    s.nonce = nonce;
    s.ntime_hex = job_.ntime_hex;
    s.extranonce2_hex = job_.extranonce2_hex;
    s.digest = d;
    s.is_block = have_block_ && meets_target(d, job_.block_target);
    s.work = share_work_;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::b2b
