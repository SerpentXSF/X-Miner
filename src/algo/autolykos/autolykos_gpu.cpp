#include "algo/autolykos/autolykos_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "core/console.hpp"

namespace xm::autolykos {

namespace {

const char* kSource = R"KRN(
typedef unsigned int u32;
typedef unsigned long long u64;

__constant__ u64 kIV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                           0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
__device__ __forceinline__ u64 bswap64(u64 x) {
  x = ((x & 0x00ff00ff00ff00ffull) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffull);
  x = ((x & 0x0000ffff0000ffffull) << 16) | ((x >> 16) & 0x0000ffff0000ffffull);
  return (x << 32) | (x >> 32);
}
__device__ __forceinline__ u32 bswap32(u32 x) {
  return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

#define G(a, b, c, d, x, y)                                 \
  v[a] = v[a] + v[b] + (x); v[d] = rotr64(v[d] ^ v[a], 32);  \
  v[c] = v[c] + v[d];       v[b] = rotr64(v[b] ^ v[c], 24);  \
  v[a] = v[a] + v[b] + (y); v[d] = rotr64(v[d] ^ v[a], 16);  \
  v[c] = v[c] + v[d];       v[b] = rotr64(v[b] ^ v[c], 63);
#define ROUND(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15) \
  G(0, 4, 8, 12, m[s0], m[s1]) G(1, 5, 9, 13, m[s2], m[s3]) G(2, 6, 10, 14, m[s4], m[s5]) G(3, 7, 11, 15, m[s6], m[s7]) \
  G(0, 5, 10, 15, m[s8], m[s9]) G(1, 6, 11, 12, m[s10], m[s11]) G(2, 7, 8, 13, m[s12], m[s13]) G(3, 4, 9, 14, m[s14], m[s15])

// One BLAKE2b compression of the 16 message words m into h (t: bytes so far including this block).
__device__ __forceinline__ void compress(u64 h[8], const u64 m[16], u64 t, bool last) {
  u64 v[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { v[i] = h[i]; v[i + 8] = kIV[i]; }
  v[12] ^= t;
  if (last) v[14] = ~v[14];
  ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
  ROUND(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4)
  ROUND(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8)
  ROUND(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13)
  ROUND(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9)
  ROUND(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11)
  ROUND(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10)
  ROUND(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5)
  ROUND(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0)
  ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  ROUND(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

__device__ __forceinline__ void init_h(u64 h[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = kIV[i];
  h[0] ^= 0x01010020ull;  // 32-byte digest, no key
}

// Table elements [start, start + count): Blake2b256(j BE || height BE || M), M = the 64-bit BE values 0..1023, so
// message word q >= 1 is bswap64(q - 1). Stored as the digest with its first byte cleared (drop(1)).
extern "C" __global__ void xm_al_table(u64* __restrict__ table, u32 height, u32 start, u32 count) {
  const u32 g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) return;
  const u32 j = start + g;
  u64 h[8];
  init_h(h);
  u64 m[16];
#pragma unroll 1
  for (u32 blk = 0; blk < 64; ++blk) {
#pragma unroll
    for (int k = 0; k < 16; ++k) m[k] = bswap64((u64)(blk * 16 + k) - 1);
    if (blk == 0) m[0] = (u64)bswap32(j) | ((u64)bswap32(height) << 32);
    compress(h, m, (u64)(blk + 1) * 128, false);
  }
  m[0] = bswap64(1023ull);
#pragma unroll
  for (int k = 1; k < 16; ++k) m[k] = 0;
  compress(h, m, 8200, true);
  u64* o = table + (size_t)g * 0 + (size_t)j * 4;
  o[0] = h[0] & ~0xffull;
  o[1] = h[1];
  o[2] = h[2];
  o[3] = h[3];
}

struct Msg { u64 w[4]; };

__device__ __forceinline__ u32 byte_at(const u64 H[4], u32 p) { return (u32)(H[(p >> 3) & 3] >> (8 * (p & 7))) & 0xffu; }

extern "C" __global__ __launch_bounds__(TPB) void xm_al_search(const u64* __restrict__ table, u32 n, Msg msg,
                                                                u64 nonce_base, u64 b_top, u32* __restrict__ found,
                                                                u32 max_found) {
  const u64 nonce = nonce_base + (u64)blockIdx.x * TPB + threadIdx.x;
  const u64 nb = bswap64(nonce);  // the nonce's 8 big-endian bytes as a little-endian word
  u64 h[8], m[16];
  // i = (last 8 bytes of Blake2b256(msg || nonce)) mod N.
  init_h(h);
  m[0] = msg.w[0]; m[1] = msg.w[1]; m[2] = msg.w[2]; m[3] = msg.w[3]; m[4] = nb;
#pragma unroll
  for (int k = 5; k < 16; ++k) m[k] = 0;
  compress(h, m, 40, true);
  const u32 i = (u32)(bswap64(h[3]) % (u64)n);
  // seed = elem(i)[1..31] || msg || nonce (71 bytes) -> Blake2b256 -> 32 indexes.
  const u64* e = table + (size_t)i * 4;
  const u64 e0 = e[0], e1 = e[1], e2 = e[2], e3 = e[3];
  init_h(h);
  m[0] = (e0 >> 8) | (e1 << 56);
  m[1] = (e1 >> 8) | (e2 << 56);
  m[2] = (e2 >> 8) | (e3 << 56);
  m[3] = (e3 >> 8) | (msg.w[0] << 56);
  m[4] = (msg.w[0] >> 8) | (msg.w[1] << 56);
  m[5] = (msg.w[1] >> 8) | (msg.w[2] << 56);
  m[6] = (msg.w[2] >> 8) | (msg.w[3] << 56);
  m[7] = (msg.w[3] >> 8) | (nb << 56);
  m[8] = nb >> 8;
#pragma unroll
  for (int k = 9; k < 16; ++k) m[k] = 0;
  compress(h, m, 71, true);
  const u64 H[4] = {h[0], h[1], h[2], h[3]};
  // Sum of the 32 elements as a 256-bit number (limb 0 least significant; element limb k = bswap64(word 3 - k)).
  u64 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  const ulonglong2* t2 = (const ulonglong2*)table;  // an element is two 16-byte loads
#pragma unroll AL_UNROLL
  for (u32 k = 0; k < 32; ++k) {
    const u32 idx = ((byte_at(H, k) << 24) | (byte_at(H, k + 1) << 16) | (byte_at(H, k + 2) << 8) | byte_at(H, k + 3)) % n;
    const ulonglong2 xa = t2[(size_t)idx * 2], xb = t2[(size_t)idx * 2 + 1];
    const u64 l0 = bswap64(xb.y), l1 = bswap64(xb.x), l2 = bswap64(xa.y), l3 = bswap64(xa.x);
    u64 t = s0 + l0;
    u64 c = t < s0;
    s0 = t;
    t = s1 + l1 + c;
    c = (t < s1) | ((t == s1) & c);
    s1 = t;
    t = s2 + l2 + c;
    c = (t < s2) | ((t == s2) & c);
    s2 = t;
    s3 = s3 + l3 + c;
  }
  // hit = Blake2b256(the sum as 32 big-endian bytes).
  init_h(h);
  m[0] = bswap64(s3); m[1] = bswap64(s2); m[2] = bswap64(s1); m[3] = bswap64(s0);
#pragma unroll
  for (int k = 4; k < 16; ++k) m[k] = 0;
  compress(h, m, 32, true);
  if (bswap64(h[0]) <= b_top) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)nonce;
      found[2 + 2 * slot] = (u32)(nonce >> 32);
    }
  }
}
)KRN";

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed) : dev_(dev), rng_(seed) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_table_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_AL_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  // How many of the 32 element reads the compiler may batch: NVIDIA does best with the fewest registers (occupancy
  // hides the latency: RTX 3060 1 = 96.3, 4 = 85.6, 8 = 64.6 MH/s), AMD with 8 (RX 9070 XT 64.5 vs 62.8).
  int unroll = dev_.info().vendor == gpu::Vendor::Nvidia ? 1 : 8;
  if (const char* e = std::getenv("XM_AL_UNROLL")) unroll = std::atoi(e);  // developer tuning
  const std::string src =
      "#define TPB " + std::to_string(tpb_) + "\n#define AL_UNROLL " + std::to_string(unroll) + "\n" + kSource;
  if (!gpu::rtc_compile(dev_.info(), src, "autolykos.cu", opts, module_)) {
    err = "Autolykos kernels did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "autolykos"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::build_table(uint32_t height, std::string& err) {
  const uint32_t n = table_size(height);
  const size_t bytes = size_t(n) * 32;
  if (n != table_n_ || !d_table_) {
    dev_.free(d_table_);
    d_table_ = 0;
    if (bytes + (size_t(256) << 20) > dev_.info().memory) {
      err = "the Autolykos table (" + std::to_string(bytes >> 20) + " MB at height " + std::to_string(height) +
            ") does not fit in GPU memory";
      return false;
    }
    d_table_ = dev_.alloc(bytes);
    if (!d_table_) {
      err = dev_.last_error();
      return false;
    }
    table_n_ = n;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const uint32_t chunk = 1u << 22;  // short launches for the Windows display watchdog
  for (uint32_t start = 0; start < n; start += chunk) {
    uint32_t count = std::min(chunk, n - start);
    void* args[] = {&d_table_, &height, &start, &count};
    if (!dev_.launch("xm_al_table", (count + 255) / 256, 1, 1, 256, 1, 1, 0, args) || !dev_.wait()) {
      err = dev_.last_error();
      return false;
    }
  }
  table_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  table_height_ = height;
  have_table_ = true;
  return true;
}

bool Engine::set_job(const Job& job, std::string& err) {
  if ((!have_table_ || job.height != table_height_) && !build_table(job.height, err)) return false;
  if (job.msg != job_.msg || job.nonce_prefix != job_.nonce_prefix || !have_job_) {
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
  struct { uint64_t w[4]; } msg;
  std::memcpy(msg.w, job_.msg.data(), 32);
  uint64_t b_top = 0;
  for (int i = 0; i < 8; ++i) b_top = b_top << 8 | job_.b[i];
  uint64_t base = counter_;
  uint32_t n = table_n_, maxf = kMaxFound;
  void* args[] = {&d_table_, &n, &msg, &base, &b_top, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_al_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args) ||
      !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    s.hit = hit(job_.msg, s.nonce, job_.height);
    if (!less_than(s.hit, job_.b)) {
      ++cpu_rejects_;  // the GPU compares the top 64 bits only
      continue;
    }
    s.job_id = job_.id;
    s.prefix_bits = job_.prefix_bits;
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::autolykos
