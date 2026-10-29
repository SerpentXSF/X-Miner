#include "algo/fishhash/fishhash_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/console.hpp"
#include "crypto/blake3.hpp"

namespace xm::fishhash {

namespace {

// Thread t owns nonce t: BLAKE3 of the block holding the nonce gives its 64-byte seed. FH_LANES lanes then work each
// hash of the group in turn: lane k holds mix words W*k .. W*k+W-1 and loads its slice of the three 128-byte items of
// a round. Finally the owner hashes the 32-byte digest with BLAKE3 and compares the top 64 bits with the target.
const char* kSearch = R"KRN(
#if defined(__HIPCC_RTC__) || defined(__HIP__)
#define SHFLL(v, src) __shfl((v), (src), FH_LANES)
#else
#define SHFLL(v, src) __shfl_sync(0xffffffffu, (v), (src), FH_LANES)
#endif
#define W (32 / FH_LANES)  // mix words per lane
#define FH_ITEMS 37748717u
#define B3_CHUNK_START 1u
#define B3_CHUNK_END 2u
#define B3_ROOT 8u
__device__ __forceinline__ u64 bswap64(u64 x) { return ((u64)bswap32((u32)x) << 32) | bswap32((u32)(x >> 32)); }
#define B3G(a, b, c, d, x, y)                                         \
  a = a + b + (x); d = rotr32(d ^ a, 16); c = c + d; b = rotr32(b ^ c, 12); \
  a = a + b + (y); d = rotr32(d ^ a, 8); c = c + d; b = rotr32(b ^ c, 7);
// BLAKE3 compression with block counter 0: out[0..7] is the chaining value / hash, out[0..15] the 64-byte XOF block.
__device__ __forceinline__ void b3_compress(const u32 cv[8], const u32 msg[16], u32 len, u32 flags, u32 out[16]) {
  u32 v[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
               0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0u, 0u, len, flags};
  u32 m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg[i];
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    B3G(v[0], v[4], v[8], v[12], m[0], m[1]);
    B3G(v[1], v[5], v[9], v[13], m[2], m[3]);
    B3G(v[2], v[6], v[10], v[14], m[4], m[5]);
    B3G(v[3], v[7], v[11], v[15], m[6], m[7]);
    B3G(v[0], v[5], v[10], v[15], m[8], m[9]);
    B3G(v[1], v[6], v[11], v[12], m[10], m[11]);
    B3G(v[2], v[7], v[8], v[13], m[12], m[13]);
    B3G(v[3], v[4], v[9], v[14], m[14], m[15]);
    const u32 t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = v[i] ^ v[i + 8];
    out[i + 8] = v[i + 8] ^ cv[i];
  }
}
struct FhParams { u32 cv[8]; u32 tail[11]; };

extern "C" __global__ __launch_bounds__(TPB) void xm_fishhash_search(const uint4* __restrict__ dag, FhParams prm,
                                                                      u64 start_nonce, u64 target_hi,
                                                                      u32* __restrict__ found, u32 max_found) {
  const u32 sub = threadIdx.x % FH_LANES;
  const u64 nonce = start_nonce + (u64)blockIdx.x * TPB + threadIdx.x;
  u32 seed[16];
  {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0;
#if FH_KARLSEN
    m[2] = (u32)nonce;  // block 1 of pre_pow || time || 32 zero bytes || nonce (LE): 8 zero bytes, then the nonce
    m[3] = (u32)(nonce >> 32);
    u32 h1[16];
    b3_compress(prm.cv, m, 16u, B3_CHUNK_END | B3_ROOT, h1);
#pragma unroll
    for (int i = 0; i < 8; ++i) seed[i] = h1[i], seed[8 + i] = 0;
#else
#pragma unroll
    for (int i = 0; i < 11; ++i) m[i] = prm.tail[i];
    m[11] = bswap32((u32)(nonce >> 32));  // header bytes 172..179: the nonce big-endian
    m[12] = bswap32((u32)nonce);
    b3_compress(prm.cv, m, 52u, B3_CHUNK_END | B3_ROOT, seed);
#endif
  }
  u32 digest[8];
#pragma unroll 1
  for (u32 h = 0; h < FH_LANES; ++h) {
    u32 s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = SHFLL(seed[i], h);
    u32 mix[W];
#pragma unroll
    for (int k = 0; k < W; ++k) {  // word W*sub + k of seed || seed, with constant register indices
      const u32 q = (sub * W / 4) & 3u;
      mix[k] = q == 0 ? s[k & 15] : q == 1 ? s[(4 + k) & 15] : q == 2 ? s[(8 + k) & 15] : s[(12 + k) & 15];
    }
#pragma unroll 1
    for (u32 r = 0; r < 32; ++r) {
#if FH_KARLSEN
      u32 gl[W / 4];
#pragma unroll
      for (int g = 0; g < W / 4; ++g) gl[g] = mix[4 * g] ^ mix[4 * g + 1] ^ mix[4 * g + 2] ^ mix[4 * g + 3];
      u32 gg[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) gg[c] = SHFLL(gl[c % (W / 4)], c / (W / 4));
      const u32 p0 = (gg[0] ^ gg[3] ^ gg[6]) % FH_ITEMS, p1 = (gg[1] ^ gg[4] ^ gg[7]) % FH_ITEMS;
      const u32 p2 = (gg[2] ^ gg[5] ^ r) % FH_ITEMS;
#else
      const u32 p0 = SHFLL(mix[0], 0) % FH_ITEMS, p1 = SHFLL(mix[4 % W], 4 / W) % FH_ITEMS;
      const u32 p2 = SHFLL(mix[8 % W], 8 / W) % FH_ITEMS;
#endif
      uint4 a[W / 4], b[W / 4], c[W / 4];
#pragma unroll
      for (int v = 0; v < W / 4; ++v) {
        a[v] = dag[(size_t)p0 * 8 + sub * (W / 4) + v];
        b[v] = dag[(size_t)p1 * 8 + sub * (W / 4) + v];
        c[v] = dag[(size_t)p2 * 8 + sub * (W / 4) + v];
      }
#pragma unroll
      for (int v = 0; v < W / 4; ++v) {
        const u32 f0[4] = {a[v].x, a[v].y, a[v].z, a[v].w};
        const u32 f1[4] = {fnv1(mix[4 * v], b[v].x), fnv1(mix[4 * v + 1], b[v].y), fnv1(mix[4 * v + 2], b[v].z),
                           fnv1(mix[4 * v + 3], b[v].w)};
        const u32 f2[4] = {mix[4 * v] ^ c[v].x, mix[4 * v + 1] ^ c[v].y, mix[4 * v + 2] ^ c[v].z, mix[4 * v + 3] ^ c[v].w};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const u64 x = ((u64)f0[2 * j] | ((u64)f0[2 * j + 1] << 32)) * ((u64)f1[2 * j] | ((u64)f1[2 * j + 1] << 32)) +
                        ((u64)f2[2 * j] | ((u64)f2[2 * j + 1] << 32));
          mix[4 * v + 2 * j] = (u32)x;
          mix[4 * v + 2 * j + 1] = (u32)(x >> 32);
        }
      }
    }
    u32 d[W / 4];
#pragma unroll
    for (int j = 0; j < W / 4; ++j) d[j] = fnv1(fnv1(fnv1(mix[4 * j], mix[4 * j + 1]), mix[4 * j + 2]), mix[4 * j + 3]);
#pragma unroll
    for (u32 l = 0; l < FH_LANES; ++l) {
#pragma unroll
      for (int j = 0; j < W / 4; ++j) {
        const u32 x = SHFLL(d[j], l);
        if (sub == h) digest[l * (W / 4) + j] = x;
      }
    }
  }
  const u32 iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                     0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
  u32 m[16], out[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = i < 8 ? digest[i] : 0u;
#if FH_KARLSEN
  b3_compress(iv, m, 32u, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT, out);
  const u64 top = ((u64)out[7] << 32) | out[6];  // the hash read little-endian: its last 8 bytes are the top
#else
  u32 cv[16];
  b3_compress(iv, seed, 64u, B3_CHUNK_START, cv);
  b3_compress(cv, m, 32u, B3_CHUNK_END | B3_ROOT, out);
  const u64 top = bswap64(((u64)out[1] << 32) | out[0]);  // the hash read big-endian
#endif
  if (top <= target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)nonce;
      found[2 + 2 * slot] = (u32)(nonce >> 32);
    }
  }
}
)KRN";

std::shared_ptr<const Light> shared_light() {
  static std::mutex mu;
  static std::shared_ptr<const Light> light;
  std::lock_guard<std::mutex> lk(mu);
  if (!light) {
    info("DAG", "building the FishHash light cache (72 MB)");
    light = make_light();
  }
  return light;
}

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed, Variant v) : dev_(dev), rng_(seed), v_(v) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_dag_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_FISHHASH_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (const char* e = std::getenv("XM_FISHHASH_LANES")) lanes_ = static_cast<uint32_t>(std::atoi(e));
  if (lanes_ != 4 && lanes_ != 8) {
    err = "XM_FISHHASH_LANES must be 4 or 8";
    return false;
  }
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  const std::string src = "#define FH_LANES " + std::to_string(lanes_) + "\n#define FH_KARLSEN " +
                          (v_ == Variant::Karlsen ? "1" : "0") + "\n" + kawpow::kernel_common(kParents, tpb_) + kSearch;
  if (!gpu::rtc_compile(dev_.info(), src, "fishhash.cu", opts, module_)) {
    err = "FishHash kernel did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "fishhash"))) {
    err = dev_.last_error();
    return false;
  }
  light_ = shared_light();
  return kawpow::generate_dag(dev_, *light_, d_dag_, err);
}

bool Engine::set_job(const Job& job, std::string& err) {
  (void)err;
  const bool same = have_job_ && job.nonce_prefix == job_.nonce_prefix &&
                    (v_ == Variant::Iron ? std::memcmp(job.header, job_.header, 172) == 0
                                         : job.pre_pow == job_.pre_pow && job.timestamp == job_.timestamp);
  job_ = job;
  blake3::Cv cv = blake3::kIV;
  if (v_ == Variant::Iron) {
    for (int b = 0; b < 2; ++b) {
      const auto o = blake3::compress(cv, job.header + 64 * b, 0, 64, b == 0 ? blake3::CHUNK_START : 0);
      std::copy(o.begin(), o.begin() + 8, cv.begin());
    }
    std::memcpy(tail_, job.header + 128, 44);
  } else {
    uint8_t block[64] = {};
    std::memcpy(block, job.pre_pow.data(), 32);
    std::memcpy(block + 32, &job.timestamp, 8);
    const auto o = blake3::compress(cv, block, 0, 64, blake3::CHUNK_START);
    std::copy(o.begin(), o.begin() + 8, cv.begin());
  }
  std::copy(cv.begin(), cv.end(), cv_);
  if (!same) {
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    const uint64_t low_mask = job.prefix_bits >= 64 ? 0 : (~0ull >> job.prefix_bits);
    counter_ = job.nonce_prefix | ((r & low_mask) >> 1);
  }
  have_job_ = true;
  return true;
}

Bytes32 Engine::cpu_hash(uint64_t nonce) const {
  if (v_ == Variant::Karlsen) return karlsen_hash(*light_, job_.pre_pow, job_.timestamp, nonce);
  uint8_t header[180];
  std::memcpy(header, job_.header, 172);
  for (int b = 0; b < 8; ++b) header[172 + b] = static_cast<uint8_t>(nonce >> (56 - 8 * b));
  return iron_hash(*light_, header, 180);
}

double Engine::iterate(uint32_t batch, std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  batch = std::max(batch / tpb_, 1u) * tpb_;
  struct { uint32_t cv[8]; uint32_t tail[11]; } prm;
  std::memcpy(prm.cv, cv_, 32);
  std::memcpy(prm.tail, tail_, 44);
  uint64_t target_hi = 0;
  for (int i = 0; i < 8; ++i)  // the top 64 bits of the target
    target_hi = target_hi << 8 | job_.target[v_ == Variant::Iron ? i : 31 - i];
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&d_dag_, &prm, &base, &target_hi, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_fishhash_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args) ||
      !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    s.hash = cpu_hash(s.nonce);
    const bool ok = v_ == Variant::Iron ? be_less_equal(s.hash, job_.target) : le_less_equal(s.hash, job_.target);
    if (!ok) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::fishhash
