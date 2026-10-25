#include "algo/quantus/quantus_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "gpu/kernels.hpp"

namespace xm::qtc {

namespace {

// Must match struct QtcJob in kernels/quantus/quantus.cu.
struct QtcJobArgs {
  uint64_t pre[12];
  uint64_t target_hi;
};

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

void set_low64(Nonce& n, uint64_t v) {
  for (int i = 0; i < 8; ++i) n[63 - i] = static_cast<uint8_t>(v >> (8 * i));
}

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() { dev_.free(d_found_); }

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_QTC_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (!gpu::load_kernels(dev_, "quantus", err)) return false;
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string& err) {
  if (job.extranonce.size() > 48) {
    err = "extranonce too long";
    return false;
  }
  job_ = job;
  // Pool prefix, then random bytes up to byte 55 (GPUs sharing a session never overlap in practice), then a
  // counter in the low 64 bits that starts below 2^63 so a batch never carries into byte 55.
  size_t i = 0;
  for (; i < job.extranonce.size(); ++i) nonce_[i] = job.extranonce[i];
  for (; i < 56; i += 8) {
    const uint64_t r = splitmix(rng_);
    for (size_t b = 0; b < 8 && i + b < 56; ++b) nonce_[i + b] = static_cast<uint8_t>(r >> (8 * b));
  }
  counter_ = splitmix(rng_) >> 1;
  set_low64(nonce_, 0);
  prestate_low64(job.mining_hash.data(), nonce_, pre_);
  have_job_ = true;
  return true;
}

double Engine::iterate(uint32_t batch, std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  QtcJobArgs a{};
  std::memcpy(a.pre, pre_, sizeof a.pre);
  for (int i = 0; i < 8; ++i) a.target_hi = a.target_hi << 8 | job_.share_target[i];
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&a, &base, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_quantus_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args)) {
    err = dev_.last_error();
    return 0;
  }
  // Blocking wait first: the download alone would let the runtime spin a CPU core for the whole batch.
  if (!dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = nonce_;
    set_low64(s.nonce, uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k]);
    s.hash = nonce_hash(job_.mining_hash.data(), s.nonce);
    if (!meets_target(s.hash, job_.share_target)) {
      ++cpu_rejects_;  // the GPU filter only compares the top 64 bits
      continue;
    }
    s.job_id = job_.id;
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::qtc
