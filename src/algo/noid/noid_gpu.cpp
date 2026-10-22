#include "algo/noid/noid_gpu.hpp"

#include <cstdlib>
#include <cstring>

#include "algo/noid/noid_consts.hpp"
#include "algo/noid/noid_gpu_consts.hpp"
#include "gpu/kernels.hpp"

namespace xm::noid {

namespace {

// Must match struct NoidJob in kernels/noid/noid.cu.
struct NoidJobArgs {
  W4 mid[4];
  W4 f[5];
  W4 nonce_base;
  W4 t2f_lo[32];
  uint64_t target_hi;
};

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() {
  for (gpu::DevPtr p : {d_rc_, d_cm_, d_f2t_, d_found_}) dev_.free(p);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  // Threads per block: every block holds its own 48 KB copy of the constant tables, so larger blocks fit more
  // threads per compute unit (RX 9070 XT: 128 -> 15.3, 256 -> 17.1, 512 -> 17.7 MH/s). NVIDIA is flat from 256 to
  // 768 (RTX 3060 ~8.65 MH/s) and 1024 exceeds its register budget. XM_NOID_TPB overrides (developer tuning).
  tpb_ = dev_.info().vendor == gpu::Vendor::Amd ? 512 : 256;
  if (const char* e = std::getenv("XM_NOID_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));
  if (!gpu::load_kernels(dev_, "noid", err)) return false;
  std::vector<W4> rc(4 * 66);
  for (int l = 0; l < 4; ++l)
    for (int r = 0; r < 66; ++r) rc[l * 66 + r] = to_w4({kRoundConstFlat[l][r].lo, kRoundConstFlat[l][r].hi});
  std::vector<uint32_t> tbl(NOID_TBL_WORDS);
  constant_tables(tbl.data());
  std::vector<uint64_t> f2t(128);
  for (int i = 0; i < 128; ++i) f2t[i] = kFlatToTower[i].hi;
  d_rc_ = dev_.alloc(rc.size() * sizeof(W4));
  d_cm_ = dev_.alloc(tbl.size() * 4);
  d_f2t_ = dev_.alloc(f2t.size() * 8);
  d_found_ = dev_.alloc((1 + kMaxFound) * 4);
  if (!d_rc_ || !d_cm_ || !d_f2t_ || !d_found_ || !dev_.upload(d_rc_, rc.data(), rc.size() * sizeof(W4)) ||
      !dev_.upload(d_cm_, tbl.data(), tbl.size() * 4) || !dev_.upload(d_f2t_, f2t.data(), f2t.size() * 8)) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  job_ = job;
  mid_ = midstate(job.fields);
  hi32_ = static_cast<uint32_t>(splitmix(rng_));
  lo32_ = 0;
  have_job_ = true;
  return true;
}

double Engine::iterate(uint32_t batch, std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  if (uint64_t(lo32_) + batch > 0x100000000ull) {  // counter space exhausted: next 2^32 block
    hi32_ = static_cast<uint32_t>(splitmix(rng_));
    lo32_ = 0;
  }
  NoidJobArgs a{};
  for (int i = 0; i < 4; ++i) a.mid[i] = to_w4(mid_[i]);
  for (int i = 0; i < 5; ++i) a.f[i] = to_w4(tower_to_flat(job_.fields[11 + i]));
  U128 base{uint64_t(hi32_) << 32, job_.nonce_prefix};
  a.nonce_base = to_w4(tower_to_flat(base));
  for (int b = 0; b < 32; ++b) a.t2f_lo[b] = to_w4({kTowerToFlat[b].lo, kTowerToFlat[b].hi});
  for (int i = 0; i < 8; ++i) a.target_hi |= uint64_t(job_.share_target[24 + i]) << (8 * i);
  uint32_t start = lo32_, maxf = kMaxFound;
  void* args[] = {&a, &d_rc_, &d_cm_, &d_f2t_, &start, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_noid_search", batch / (tpb_ * kLanes), 1, 1, tpb_, 1, 1, NOID_TBL_WORDS * 4, args) ||
      !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  lo32_ += batch;
  uint32_t hits = std::min(found[0], kMaxFound);
  bool have_block = false;
  for (uint8_t b : job_.block_target) have_block |= b != 0;
  for (uint32_t h = 0; h < hits; ++h) {
    U128 nonce{uint64_t(hi32_) << 32 | found[1 + h], job_.nonce_prefix};
    Digest d = digest_from_midstate(mid_, job_.fields, nonce);
    if (!less_le256(d, job_.share_target.data())) {
      ++cpu_rejects_;  // the GPU filter only compares the top 64 bits
      continue;
    }
    Share s;
    s.job_id = job_.id;
    s.nonce = nonce;
    s.digest = d;
    s.is_block = have_block && less_le256(d, job_.block_target.data());
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::noid
