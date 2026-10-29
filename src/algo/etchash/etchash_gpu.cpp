#include "algo/etchash/etchash_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/console.hpp"

namespace xm::etchash {

namespace {

// hashimoto with ETH_LANES (1, 2, 4 or 8) lanes per hash: lane k holds mix words W*k .. W*k+W-1 and loads its share of
// each 128-byte DAG row, so a row is one coalesced read. Thread t of a group owns nonce t (its Keccak-512 seed and
// its final Keccak-256); the group works its hashes in turn. Fewer lanes per hash keep more independent DAG read
// chains in flight (hashimoto is a chain of 64 dependent reads).
const char* kSearch = R"KRN(
#if defined(__HIPCC_RTC__) || defined(__HIP__)
#define SHFLL(v, src) __shfl((v), (src), ETH_LANES)
#else
#define SHFLL(v, src) __shfl_sync(0xffffffffu, (v), (src), ETH_LANES)
#endif
#define W (32 / ETH_LANES)  // mix words per lane
__device__ __forceinline__ u64 bswap64(u64 x) { return ((u64)bswap32((u32)x) << 32) | bswap32((u32)(x >> 32)); }
// x mod n with magic = floor(2^32 / n): the quotient estimate is at most one low, so one correction.
__device__ __forceinline__ u32 fast_mod(u32 x, u32 n, u32 magic) {
  u32 r = x - __umulhi(x, magic) * n;
  return r >= n ? r - n : r;
}

extern "C" __global__ __launch_bounds__(TPB) void xm_etchash_search(const uint4* __restrict__ dag, u32 rows, u32 magic,
                                                                     Hdr hdr, u64 start_nonce, u64 target_hi,
                                                                     u32* __restrict__ found, u32 max_found) {
  const u32 sub = threadIdx.x % ETH_LANES;
  const u64 nonce = start_nonce + (u64)blockIdx.x * TPB + threadIdx.x;
  u64 seed[8];
  {
    u64 st[25];
    for (int i = 0; i < 4; ++i) st[i] = (u64)hdr.w[2 * i] | ((u64)hdr.w[2 * i + 1] << 32);
    st[4] = nonce;
    st[5] = 0x01ull;  // Keccak-512 padding of the 40-byte message (rate 72)
    for (int i = 6; i < 25; ++i) st[i] = 0;
    st[8] = 0x8000000000000000ull;
    keccakf1600(st);
    for (int i = 0; i < 8; ++i) seed[i] = st[i];
  }
  u32 digest[8];
#pragma unroll 1
  for (u32 h = 0; h < ETH_LANES; ++h) {
    u32 s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#if ETH_LANES > 1
      s[2 * i] = SHFLL((u32)seed[i], h);
      s[2 * i + 1] = SHFLL((u32)(seed[i] >> 32), h);
#else
      s[2 * i] = (u32)seed[i];
      s[2 * i + 1] = (u32)(seed[i] >> 32);
#endif
    }
    u32 mix[W];
#pragma unroll
    for (int k = 0; k < W; ++k) {  // word W*sub + k of seed || seed, with constant register indices
      const u32 q = (sub * W / 4) & 3u;  // which quarter of the 16 seed words starts this lane (W < 16)
      mix[k] = W >= 16 ? s[k & 15] : q == 0 ? s[k & 15] : q == 1 ? s[(4 + k) & 15] : q == 2 ? s[(8 + k) & 15] : s[(12 + k) & 15];
    }
    const u32 s0 = s[0];
#pragma unroll 1
    for (u32 i0 = 0; i0 < 64; i0 += W) {
#if ETH_LANES > 1
      const u32 owner = (i0 & 31u) / W;  // the lane holding words i0 % 32 .. + W - 1
#endif
#pragma unroll
      for (u32 k = 0; k < W; ++k) {
#if ETH_LANES > 1
        const u32 word = SHFLL(mix[k], owner);
#else
        const u32 word = mix[k];
#endif
        const u32 p = fast_mod(fnv1((i0 + k) ^ s0, word), rows, magic);
        const uint4* q = dag + (size_t)p * 8 + sub * (W / 4);
#pragma unroll
        for (int v = 0; v < W / 4; ++v) {
          const uint4 a = q[v];
          mix[4 * v] = fnv1(mix[4 * v], a.x);
          mix[4 * v + 1] = fnv1(mix[4 * v + 1], a.y);
          mix[4 * v + 2] = fnv1(mix[4 * v + 2], a.z);
          mix[4 * v + 3] = fnv1(mix[4 * v + 3], a.w);
        }
      }
    }
    u32 d[W / 4];
#pragma unroll
    for (int j = 0; j < W / 4; ++j) d[j] = fnv1(fnv1(fnv1(mix[4 * j], mix[4 * j + 1]), mix[4 * j + 2]), mix[4 * j + 3]);
#pragma unroll
    for (u32 l = 0; l < ETH_LANES; ++l) {
#pragma unroll
      for (int j = 0; j < W / 4; ++j) {
#if ETH_LANES > 1
        const u32 x = SHFLL(d[j], l);
        if (sub == h) digest[l * (W / 4) + j] = x;
#else
        digest[j] = d[j];
#endif
      }
    }
  }
  u64 st[25];
  for (int i = 0; i < 8; ++i) st[i] = seed[i];
  for (int i = 0; i < 4; ++i) st[8 + i] = (u64)digest[2 * i] | ((u64)digest[2 * i + 1] << 32);
  st[12] = 0x01ull;  // Keccak-256 padding of the 96-byte message (rate 136)
  for (int i = 13; i < 25; ++i) st[i] = 0;
  st[16] = 0x8000000000000000ull;
  keccakf1600(st);
  if (bswap64(st[0]) <= target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)nonce;
      found[2 + 2 * slot] = (u32)(nonce >> 32);
    }
  }
}
)KRN";

std::shared_ptr<const Light> shared_light(int height) {
  static std::mutex mu;
  static std::map<int, std::shared_ptr<const Light>> cache;  // by epoch
  const int epoch = epoch_for_height(height);
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(epoch);
  if (it != cache.end()) return it->second;
  for (auto i = cache.begin(); i != cache.end();) i = i->first < epoch - 1 ? cache.erase(i) : std::next(i);
  info("DAG", "building the Etchash epoch " + std::to_string(epoch) + " light cache");
  return cache[epoch] = make_light(height);
}

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_dag_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_ETCHASH_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (const char* e = std::getenv("XM_ETCHASH_LANES")) lanes_ = static_cast<uint32_t>(std::atoi(e));
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  const std::string src = "#define ETH_LANES " + std::to_string(lanes_) + "\n" + kawpow::kernel_common(kDatasetParents, tpb_) + kSearch;
  if (!gpu::rtc_compile(dev_.info(), src, "etchash.cu", opts, module_)) {
    err = "Etchash kernel did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "etchash"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string& err) {
  const int epoch = epoch_for_height(job.height);
  if (epoch != dag_epoch_) {
    dag_epoch_ = -1;  // generate_dag frees the old DAG first: set again only on success
    if (epoch < 0 || epoch > 4096) {
      err = "implausible block height " + std::to_string(job.height);
      return false;
    }
    light_ = shared_light(job.height);
    if (!kawpow::generate_dag(dev_, *light_, d_dag_, err)) return false;
    dag_epoch_ = epoch;
  }
  if (job.header != job_.header || job.nonce_prefix != job_.nonce_prefix || !have_job_) {
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    const uint64_t low_mask = job.prefix_bits >= 64 ? 0 : (~0ull >> job.prefix_bits);
    counter_ = job.nonce_prefix | ((r & low_mask) >> 1);
  }
  job_ = job;
  have_job_ = true;
  return true;
}

double Engine::iterate(uint32_t batch, std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  uint32_t rows = uint32_t(light_->dataset_items);
  uint32_t magic = uint32_t(0x100000000ull / rows);
  struct { uint32_t w[8]; } hdr;
  std::memcpy(hdr.w, job_.header.data(), 32);
  uint64_t target_hi = 0;
  const uint8_t* t = reinterpret_cast<const uint8_t*>(job_.target.data());
  for (int i = 0; i < 8; ++i) target_hi = target_hi << 8 | t[i];
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&d_dag_, &rows, &magic, &hdr, &base, &target_hi, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_etchash_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args) ||
      !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    const Result r = hash(*light_, job_.header, s.nonce);
    if (!kawpow::less_or_equal(r.final_hash, job_.target)) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.header = job_.header;
    s.mix = r.mix_hash;
    s.final_hash = r.final_hash;
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::etchash
