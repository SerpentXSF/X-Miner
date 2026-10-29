#include "algo/octopus/octopus_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/console.hpp"

namespace xm::octopus {

namespace {

// One thread per nonce; each aligned group of 32 threads is one Octopus nonce group.
// 1. Thread i of the group runs SipHash on its nonce and writes coefficients j * 32 + i (j < 32) to shared memory.
// 2. Every thread evaluates the group's polynomial (Horner, 1024 steps) at its 32 points together, with Shoup's
//    multiply: q = hi(acc * xs), acc * x - q * p is in [0, 2p) for any 32-bit acc, so the accumulators stay below 3p
//    without a reduction inside the loop (one high and two low multiply-adds per step).
// 3. The Ethash-style mix of 32 dependent 256-byte DAG pages, OCT_LANES lanes per hash (lane k holds mix words
//    W*k .. W*k+W-1 and loads its slice of each page), the hashes of an OCT_LANES group in turn.
const char* kSearch = R"KRN(
#if defined(__HIPCC_RTC__) || defined(__HIP__)
#define SHFLL(v, src) __shfl((v), (src), OCT_LANES)
#else
#define SHFLL(v, src) __shfl_sync(0xffffffffu, (v), (src), OCT_LANES)
#endif
#define OCT_P 1032193u
#define W (64 / OCT_LANES)  // mix words per lane
__device__ __forceinline__ u64 bswap64(u64 x) { return ((u64)bswap32((u32)x) << 32) | bswap32((u32)(x >> 32)); }
__device__ __forceinline__ u32 fast_mod(u32 x, u32 n, u32 magic) {
  u32 r = x - __umulhi(x, magic) * n;
  return r >= n ? r - n : r;
}
#define SIPROUND                                                     \
  do {                                                               \
    v0 += v1; v2 += v3; v1 = rotl64(v1, 13); v3 = rotl64(v3, 16);    \
    v1 ^= v0; v3 ^= v2; v0 = rotl64(v0, 32);                         \
    v2 += v1; v0 += v3; v1 = rotl64(v1, 17); v3 = rotl64(v3, 21);    \
    v1 ^= v2; v3 ^= v0; v2 = rotl64(v2, 32);                         \
  } while (0)

extern "C" __global__ __launch_bounds__(TPB) void xm_octopus_search(const uint4* __restrict__ dag, u32 pages, u32 magic,
                                                                     const uint2* __restrict__ pts, Hdr hdr,
                                                                     u64 start_nonce, u64 target_hi,
                                                                     u32* __restrict__ found, u32 max_found) {
  __shared__ u32 coef[TPB / 32][1024];
  const u32 lane = threadIdx.x & 31u, grp = threadIdx.x >> 5;
  const u64 nonce = start_nonce + (u64)blockIdx.x * TPB + threadIdx.x;  // start_nonce is a multiple of 32
  const u64 k0 = (u64)hdr.w[0] | ((u64)hdr.w[1] << 32), k1 = (u64)hdr.w[2] | ((u64)hdr.w[3] << 32);
  const u64 k2 = (u64)hdr.w[4] | ((u64)hdr.w[5] << 32), k3 = (u64)hdr.w[6] | ((u64)hdr.w[7] << 32);
  {
    u64 v0 = k0, v1 = k1, v2 = k2, v3 = k3 ^ nonce;
    SIPROUND;
    SIPROUND;
    v0 ^= nonce;
    v2 ^= 0xffull;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
#pragma unroll 4
    for (u32 j = 0; j < 32; ++j) {
      SIPROUND;
      coef[grp][j * 32 + lane] = fast_mod((u32)(v0 ^ v1 ^ v2 ^ v3), OCT_P, 4161u);  // 4161 = floor(2^32 / p)
    }
  }
  __syncthreads();
  u32 res[32];
  u64 result = 0;
  {
    u32 x[32], xs[32];
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      const uint2 q = pts[lane + 32 * t];
      x[t] = q.x;
      xs[t] = q.y;
      res[t] = 0;
#if OCT_MUL24
      asm volatile("" : "+v"(x[t]), "+v"(xs[t]));  // keeps them in registers (the compiler re-loads them per step)
#endif
    }
    const u32* c = coef[grp];
#pragma unroll 2
    for (int j = 1023; j >= 0; --j) {
      const u32 dj = c[j];
#pragma unroll
      for (int t = 0; t < 32; ++t) {
#if OCT_MUL24
        // AMD: 32-bit multiplies are quarter rate, 24-bit ones full rate. acc < 3p < 2^22 and xs = floor(x 2^24 / p)
        // < 2^24: q = (acc xs) >> 24 from the 48-bit product's two halves.
        // Written in asm: the compiler otherwise re-masks the operands and turns q * p into a quarter-rate mul_lo.
        u32 lo, hi, r, qp;
        asm("v_mul_u32_u24 %0, %1, %2" : "=v"(lo) : "v"(res[t]), "v"(xs[t]));
        asm("v_mul_hi_u32_u24 %0, %1, %2" : "=v"(hi) : "v"(res[t]), "v"(xs[t]));
        asm("v_mad_u32_u24 %0, %1, %2, %3" : "=v"(r) : "v"(res[t]), "v"(x[t]), "v"(dj));
        const u32 q = __builtin_amdgcn_alignbit(hi, lo, 24);
        asm("v_mul_u32_u24 %0, 0xfc001, %1" : "=v"(qp) : "v"(q));  // p = 0xfc001
        res[t] = r - qp;
#else
        const u32 q = __umulhi(res[t], xs[t]);
        res[t] = res[t] * x[t] + dj - q * OCT_P;
#endif
      }
    }
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      u32 v = res[t];
      v = v >= 2 * OCT_P ? v - 2 * OCT_P : v;
      v = v >= OCT_P ? v - OCT_P : v;
      res[t] = v;
      result = result * 0x01000193ull ^ v;
    }
  }
  u64 seed[8];
  {
    u64 st[25];
    for (int i = 0; i < 4; ++i) st[i] = (u64)hdr.w[2 * i] | ((u64)hdr.w[2 * i + 1] << 32);
    st[4] = result;
    st[5] = 0x01ull;  // Keccak-512 padding of the 40-byte message (rate 72)
    for (int i = 6; i < 25; ++i) st[i] = 0;
    st[8] = 0x8000000000000000ull;
    keccakf1600(st);
    for (int i = 0; i < 8; ++i) seed[i] = st[i];
  }
  const u32 sub = threadIdx.x % OCT_LANES;
  u32 red[16];  // this thread's hash: the 16 four-word FNV reductions of its final mix
#pragma unroll 1
  for (u32 h = 0; h < OCT_LANES; ++h) {
    u32 s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s[2 * i] = SHFLL((u32)seed[i], h);
      s[2 * i + 1] = SHFLL((u32)(seed[i] >> 32), h);
    }
    u32 mix[W];
#pragma unroll
    for (int k = 0; k < W; ++k) {  // word W*sub + k of the seed repeated, with constant register indices
      const u32 q = (sub * W / 4) & 3u;
      mix[k] = W >= 16 ? s[k & 15] : q == 0 ? s[k & 15] : q == 1 ? s[(4 + k) & 15] : q == 2 ? s[(8 + k) & 15] : s[(12 + k) & 15];
    }
    const u32 s0 = s[0];
#pragma unroll
    for (u32 i = 0; i < 32; ++i) {
      const u32 r = SHFLL(res[i], h);
      const u32 word = SHFLL(mix[i % W], i / W);
      const u32 p = fast_mod(fnv1(s0 ^ i ^ r, word), pages, magic);
      const uint4* g = dag + (size_t)p * 16 + sub * (W / 4);
#pragma unroll
      for (int v = 0; v < W / 4; ++v) {
        const uint4 a = g[v];
        mix[4 * v] = fnv1(mix[4 * v], a.x);
        mix[4 * v + 1] = fnv1(mix[4 * v + 1], a.y);
        mix[4 * v + 2] = fnv1(mix[4 * v + 2], a.z);
        mix[4 * v + 3] = fnv1(mix[4 * v + 3], a.w);
      }
    }
    u32 d[W / 4];
#pragma unroll
    for (int j = 0; j < W / 4; ++j) d[j] = fnv1(fnv1(fnv1(mix[4 * j], mix[4 * j + 1]), mix[4 * j + 2]), mix[4 * j + 3]);
#pragma unroll
    for (u32 l = 0; l < OCT_LANES; ++l) {
#pragma unroll
      for (int j = 0; j < W / 4; ++j) {
        const u32 x = SHFLL(d[j], l);
        if (sub == h) red[l * (W / 4) + j] = x;
      }
    }
  }
  u64 st[25];
  for (int i = 0; i < 8; ++i) st[i] = seed[i];
  for (int i = 0; i < 4; ++i)
    st[8 + i] = (u64)fnv1(red[2 * i], red[8 + 2 * i]) | ((u64)fnv1(red[2 * i + 1], red[9 + 2 * i]) << 32);
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

std::shared_ptr<const Light> shared_light(int stage) {
  static std::mutex mu;
  static std::map<int, std::shared_ptr<const Light>> cache;  // by stage
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(stage);
  if (it != cache.end()) return it->second;
  for (auto i = cache.begin(); i != cache.end();) i = i->first < stage - 1 ? cache.erase(i) : std::next(i);
  info("DAG", "building the Octopus stage " + std::to_string(stage) + " light cache");
  return cache[stage] = make_light(stage);
}

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed), pts_(std::make_unique<Points>()) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_dag_);
  dev_.free(d_pts_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_OCTOPUS_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (const char* e = std::getenv("XM_OCTOPUS_LANES")) lanes_ = static_cast<uint32_t>(std::atoi(e));
  mul24_ = dev_.info().vendor == gpu::Vendor::Amd;
  if (const char* e = std::getenv("XM_OCTOPUS_MUL24")) mul24_ = mul24_ && std::atoi(e) != 0;
  if (tpb_ % 32 || lanes_ < 4 || lanes_ > 16 || (lanes_ & (lanes_ - 1))) {
    err = "XM_OCTOPUS_TPB must be a multiple of 32 and XM_OCTOPUS_LANES 4, 8 or 16";
    return false;
  }
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  d_pts_ = dev_.alloc(sizeof(uint32_t) * 2 * kN);
  if (!d_found_ || !d_pts_) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  const std::string src =
      "#define OCT_LANES " + std::to_string(lanes_) + "\n#define OCT_MUL24 " + (mul24_ ? "1" : "0") + "\n" +
      kawpow::kernel_common(kDatasetParents, tpb_) + kSearch;
  if (!gpu::rtc_compile(dev_.info(), src, "octopus.cu", opts, module_)) {
    err = "Octopus kernel did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "octopus"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string& err) {
  const int stage = stage_for_height(job.height);
  if (stage != dag_stage_) {
    dag_stage_ = -1;  // generate_dag frees the old DAG first: set again only on success
    if (stage < 0 || stage > 1024) {  // 1024 stages = a 20 GB DAG
      err = "implausible block height " + std::to_string(job.height);
      return false;
    }
    light_ = shared_light(stage);
    if (!kawpow::generate_dag(dev_, *light_, d_dag_, err)) return false;
    dag_stage_ = stage;
  }
  if (job.header != job_.header || !have_job_) {
    make_points(job.header, *pts_);
    std::vector<uint32_t> packed(2 * kN);
    for (int k = 0; k < kN; ++k) {
      packed[2 * k] = pts_->x[k];
      packed[2 * k + 1] = mul24_ ? uint32_t((uint64_t(pts_->x[k]) << 24) / kMod) : pts_->xs[k];
    }
    if (!dev_.upload(d_pts_, packed.data(), packed.size() * 4)) {
      err = dev_.last_error();
      return false;
    }
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    counter_ = (r >> 1) & ~uint64_t(kGroup - 1);  // nonce groups start at multiples of 32
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
  batch = std::max(batch / tpb_, 1u) * tpb_;
  uint32_t pages = uint32_t(light_->dataset_items / 2);
  uint32_t magic = uint32_t(0x100000000ull / pages);
  struct { uint32_t w[8]; } hdr;
  std::memcpy(hdr.w, job_.header.data(), 32);
  uint64_t target_hi = 0;
  const uint8_t* t = reinterpret_cast<const uint8_t*>(job_.boundary.data());
  for (int i = 0; i < 8; ++i) target_hi = target_hi << 8 | t[i];
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&d_dag_, &pages, &magic, &d_pts_, &hdr, &base, &target_hi, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_octopus_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args) ||
      !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    s.final_hash = hash(*light_, *pts_, job_.header, s.nonce);
    if (!below(s.final_hash, job_.boundary)) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.header = job_.header;
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::octopus
