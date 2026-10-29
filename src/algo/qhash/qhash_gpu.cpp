#include "algo/qhash/qhash_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/console.hpp"
#include "crypto/sha256.hpp"

namespace xm::qhash {

namespace {

// One thread per nonce. The header's first 64 bytes are a host midstate; the second block carries header bytes 64..75
// (tail) and the nonce. The circuit contraction runs in QH_REAL (float or double): double reproduces the CPU
// reference exactly; float differs in the last Q15 bit on ~0.7% of hashes (those are missed shares, never bad ones:
// the CPU re-checks every candidate). The cos / sin tables live in shared memory (per-thread nibble indices would
// serialize constant-cache reads).
const char* kKernel = R"KRN(
typedef unsigned int u32;
typedef unsigned long long u64;
__device__ __forceinline__ u32 bswap32(u32 x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
__device__ __forceinline__ u32 ror(u32 x, u32 n) { return (x >> n) | (x << (32u - n)); }

__constant__ u32 K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

#define S0(x) (ror(x, 2) ^ ror(x, 13) ^ ror(x, 22))
#define S1(x) (ror(x, 6) ^ ror(x, 11) ^ ror(x, 25))
#define s0(x) (ror(x, 7) ^ ror(x, 18) ^ ((x) >> 3))
#define s1(x) (ror(x, 17) ^ ror(x, 19) ^ ((x) >> 10))
#define CH(e, f, g) ((g) ^ ((e) & ((f) ^ (g))))
#define MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) | (b))))

// Rounds START..63 from the working state v (the state after rounds 0..START-1), feed-forward into s.
template <int START>
__device__ __forceinline__ void sha256_rounds(u32 s[8], const u32 v[8], u32 w[16]) {
  u32 a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5], g = v[6], h = v[7];
#pragma unroll
  for (int i = START; i < 64; ++i) {
    if (i >= 16) w[i & 15] += s1(w[(i + 14) & 15]) + w[(i + 9) & 15] + s0(w[(i + 1) & 15]);
    const u32 t1 = h + S1(e) + CH(e, f, g) + K256[i] + w[i & 15];
    const u32 t2 = S0(a) + MAJ(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}
__device__ __forceinline__ void sha256_block(u32 s[8], u32 w[16]) { sha256_rounds<0>(s, s, w); }

// The constant padding block of a 64-byte message (0x80, zeros, length 512): K[i] + W[i] folded at compile time.
__constant__ u32 PADWK[64] = {QH_PADWK};
__device__ __forceinline__ void sha256_pad64(u32 s[8]) {
  u32 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const u32 t1 = h + S1(e) + CH(e, f, g) + PADWK[i];
    const u32 t2 = S0(a) + MAJ(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

struct Params {
  u32 mid[8];
  u32 r3[8];  // working state after rounds 0..2 of the header's second block (they use only tail words)
  u32 tail[3];
  u32 base;
  u32 odd;
  u32 zlimit;
  u64 target_hi;
};

__device__ __forceinline__ u32 to_fixed(QH_REAL e) {
  const QH_REAL v = e >= (QH_REAL)0 ? e * (QH_REAL)32768 + (QH_REAL)0.5 : e * (QH_REAL)32768 - (QH_REAL)0.5;
  return ((u32)(int)v) & 0xffffu;
}
__device__ __forceinline__ u32 zero_bytes(u32 f) { return (u32)((f & 0xffu) == 0u) + (u32)((f >> 8) == 0u); }

struct __align__(4 * sizeof(QH_REAL)) Site1 { QH_REAL A, B, C, pad; };  // by (n1, n2)
struct __align__(2 * sizeof(QH_REAL)) Site3 { QH_REAL C2, S2; };         // by n3

extern "C" __global__ __launch_bounds__(TPB) void xm_qhash(Params p, u32* __restrict__ found, u32 maxf) {
  // Per-site coefficients as two vector loads: (A, B, C) by the layer-1 RY / RZ nibble pair, (C2, S2) by the layer-2
  // RY nibble. Products formed exactly as the reference (sin * cos in QH_REAL).
  __shared__ Site1 t1[256];
  __shared__ Site3 t3[16];
  for (u32 i = threadIdx.x; i < 256u; i += TPB) {
    const u32 n1 = i >> 4, n2 = i & 15u;
    const QH_REAL s1v = p.odd ? QH_SIN1[n1] : QH_SIN0[n1], c1v = p.odd ? QH_COS1[n1] : QH_COS0[n1];
    const QH_REAL s2v = p.odd ? QH_SIN1[n2] : QH_SIN0[n2], c2v = p.odd ? QH_COS1[n2] : QH_COS0[n2];
    Site1 v;
    v.A = s1v * c2v; v.B = s1v * s2v; v.C = c1v; v.pad = (QH_REAL)0;
    t1[i] = v;
  }
  if (threadIdx.x < 16) {
    Site3 v;
    v.C2 = p.odd ? QH_COS1[threadIdx.x] : QH_COS0[threadIdx.x];
    v.S2 = -(p.odd ? QH_SIN1[threadIdx.x] : QH_SIN0[threadIdx.x]);
    t3[threadIdx.x] = v;
  }
  __syncthreads();
  const u32 nonce = p.base + blockIdx.x * TPB + threadIdx.x;

  // inner = SHA256(header): the second block from the midstate, rounds 0..2 done on the host.
  u32 s[8], w[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = p.mid[i];
  w[0] = p.tail[0]; w[1] = p.tail[1]; w[2] = p.tail[2]; w[3] = bswap32(nonce);
  w[4] = 0x80000000u;
#pragma unroll
  for (int i = 5; i < 15; ++i) w[i] = 0u;
  w[15] = 640u;
  sha256_rounds<3>(s, p.r3, w);

  // Nibble k of the hash: word k / 8, most significant nibble first. Site k uses nibbles k, 16 + k, 32 + k.
#define NIB(k) ((s[(k) >> 3] >> (28 - 4 * ((k) & 7))) & 15u)
  QH_REAL t = (QH_REAL)1, z = (QH_REAL)1, x = (QH_REAL)0, y = (QH_REAL)0;
  Site1 cur = t1[NIB(0) << 4 | NIB(16)];
  u32 fx[16];
  u32 zeros = 0;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const Site3 q = t3[NIB(32 + k)];
    const QH_REAL A = cur.A, B = cur.B, C = cur.C;
    const QH_REAL t2 = q.C2 * (C * z - B * y), z2 = q.C2 * (t + A * x), x2 = q.S2 * (x + A * t), y2 = q.S2 * (C * y + B * z);
    t = t2; z = z2; x = x2; y = y2;
    QH_REAL next = (QH_REAL)1;
    if (k < 15) {
      cur = t1[NIB(k + 1) << 4 | NIB(17 + k)];
      next = cur.A;
    }
    fx[k] = to_fixed(t + x * next);
    zeros += zero_bytes(fx[k]);
  }
#undef NIB
  if (zeros >= p.zlimit) return;

  // hash = SHA256(inner | 32 fixed-point bytes): one data block and the constant padding block.
  u32 h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = s[i];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const u32 f0 = fx[2 * i], f1 = fx[2 * i + 1];  // bytes lo0 hi0 lo1 hi1
    w[8 + i] = (f0 & 0xffu) << 24 | (f0 >> 8) << 16 | (f1 & 0xffu) << 8 | (f1 >> 8);
  }
  sha256_block(h, w);
  sha256_pad64(h);
  // The hash as a little-endian number: its top 64 bits are bytes 31..24 = bswap(h7), bswap(h6).
  const u64 top = (u64)bswap32(h[7]) << 32 | bswap32(h[6]);
  if (top <= p.target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < maxf) found[1 + slot] = nonce;
  }
}
)KRN";

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// K[i] + W[i] of the padding block that follows a 64-byte message.
std::string pad_wk() {
  uint32_t w[64] = {0x80000000u};
  w[15] = 512;
  for (int i = 16; i < 64; ++i)
    w[i] = w[i - 16] + (ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
           (ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10));
  std::string s = "#define QH_PADWK ";
  for (int i = 0; i < 64; ++i) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%08xu%s", w[i] + crypto::kSha256K[i], i < 63 ? ", " : "\n");
    s += b;
  }
  return s;
}

// The working state after SHA-256 rounds 0..2 (message words w0..w2) from state st.
void rounds3(const uint32_t st[8], const uint32_t w[3], uint32_t out[8]) {
  uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
  for (int i = 0; i < 3; ++i) {
    const uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + crypto::kSha256K[i] + w[i];
    const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = e; out[5] = f; out[6] = g; out[7] = h;
}

std::string table(const char* name, bool odd, bool f32) {
  std::string s = std::string("__constant__ QH_REAL ") + name + "[16] = {";
  for (int n = 0; n < 16; ++n) {
    const double a = (2.0 * n + (odd ? 1 : 0)) * 3.14159265358979323846 / 16;
    const double v = name[3] == 'C' ? std::cos(a) : std::sin(a);
    char buf[64];
    std::snprintf(buf, sizeof buf, f32 ? "%.9ef%s" : "%.17e%s", v, n < 15 ? ", " : "};\n");
    s += buf;
  }
  return s;
}

struct Params {
  uint32_t mid[8];
  uint32_t r3[8];
  uint32_t tail[3];
  uint32_t base;
  uint32_t odd;
  uint32_t zlimit;
  uint64_t target_hi;
};
static_assert(sizeof(Params) == 96, "kernel parameter layout");

}  // namespace

Engine::Engine(gpu::Device& dev, uint64_t seed, uint32_t batch, int precision)
    : dev_(dev), rng_(seed), batch_(batch), precision_(precision) {}

Engine::~Engine() { dev_.free(d_found_); }

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_QHASH_TPB")) tpb_ = std::max(32u, uint32_t(std::atoi(e)));  // developer tuning
  // Float on both vendors by default: consumer GPUs run doubles at 1/16..1/64 rate. XM_QHASH_FP64=1 for exact.
  fp64_ = false;
  if (const char* e = std::getenv("XM_QHASH_FP64")) fp64_ = std::atoi(e) != 0;
  if (precision_ >= 0) fp64_ = precision_ == 1;
  if (batch_ == 0) {
    batch_ = 1u << 24;
    if (const char* e = std::getenv("XM_QHASH_BATCH")) batch_ = uint32_t(std::strtoul(e, nullptr, 10));
  }
  batch_ = std::max(tpb_, batch_ / tpb_ * tpb_);
  d_found_ = dev_.alloc((1 + kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  std::string src = "#define TPB " + std::to_string(tpb_) + "\n#define QH_REAL " + (fp64_ ? "double" : "float") + "\n";
  src += table("QH_COS0", false, !fp64_) + table("QH_COS1", true, !fp64_) + table("QH_SIN0", false, !fp64_) +
         table("QH_SIN1", true, !fp64_);
  src += pad_wk();
  src += kKernel;
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  if (!gpu::rtc_compile(dev_.info(), src, "qhash.cu", opts, module_)) {
    err = "QHash kernel did not compile: " + module_.log.substr(0, 2000);
    return false;
  }
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "qhash"))) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  const bool same_work = have_job_ && job.tpl.coinb1 == job_.tpl.coinb1 && job.tpl.coinb2 == job_.tpl.coinb2 &&
                         job.tpl.extranonce1 == job_.tpl.extranonce1 && job.tpl.prevhash == job_.tpl.prevhash &&
                         job.tpl.branches == job_.tpl.branches && job.tpl.ntime == job_.tpl.ntime;
  if (!same_work) {
    // A random extranonce2 start per GPU and job keeps several GPUs (and restarts) off each other's work.
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    r ^= r >> 29;
    const int bits = std::min(job.tpl.extranonce2_size, 4) * 8;
    const uint64_t en2 = bits >= 32 ? (r & 0xffffffffull) : (r & ((1ull << bits) - 1));
    counter_ = en2 << 32;
    header_en2_ = ~0ull;
  }
  job_ = job;
  target_ = target_from_difficulty(job.difficulty);
  block_target_ = target_from_nbits(job.tpl.nbits);
  have_job_ = true;
  return true;
}

bool Engine::load_header(std::string&) {
  const uint64_t en2 = counter_ >> 32;
  if (en2 == header_en2_) return true;
  header(job_.tpl, en2, header_);
  uint32_t st[8];
  std::memcpy(st, crypto::kSha256Init, sizeof st);
  crypto::sha256_compress(st, header_);
  std::memcpy(midstate_, st, sizeof st);
  for (int i = 0; i < 3; ++i)
    tail_[i] = uint32_t(header_[64 + 4 * i]) << 24 | uint32_t(header_[65 + 4 * i]) << 16 |
               uint32_t(header_[66 + 4 * i]) << 8 | header_[67 + 4 * i];
  header_en2_ = en2;
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  // A launch never crosses an extranonce2 boundary (the 32-bit nonce space of one header); a sliver smaller than a
  // block at the end of the nonce space is skipped.
  uint64_t room = (1ull << 32) - (counter_ & 0xffffffffull);
  if (room < tpb_) {
    counter_ = ((counter_ >> 32) + 1) << 32;
    room = 1ull << 32;
  }
  const uint32_t n = uint32_t(std::min<uint64_t>(batch_, room / tpb_ * tpb_));
  load_header(err);
  Params p{};
  std::memcpy(p.mid, midstate_, sizeof p.mid);
  std::memcpy(p.tail, tail_, sizeof p.tail);
  rounds3(midstate_, tail_, p.r3);
  p.base = uint32_t(counter_);
  p.odd = odd_angles(job_.tpl.ntime) ? 1 : 0;
  p.zlimit = zero_limit(job_.tpl.ntime);
  p.target_hi = target_top64(target_);
  uint32_t maxf = kMaxFound;
  void* args[] = {&p, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_qhash", n / tpb_, 1, 1, tpb_, 1, 1, 0, args) || !dev_.wait() ||
      !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  const uint64_t en2 = counter_ >> 32;
  counter_ += n;  // carries into the next extranonce2 at the end of the nonce space
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = found[1 + k];
    uint8_t hdr[kHeaderSize];
    std::memcpy(hdr, header_, kHeaderSize);
    for (int i = 0; i < 4; ++i) hdr[76 + i] = uint8_t(s.nonce >> (8 * i));
    s.hash = hash(hdr);
    if (!meets(s.hash, target_)) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.extranonce2_hex = extranonce2_hex(job_.tpl, en2);
    s.ntime_hex = job_.ntime_hex;
    s.work = job_.difficulty * 4294967296.0;
    s.is_block = meets(s.hash, block_target_);
    shares.push_back(s);
  }
  return n;
}

}  // namespace xm::qhash
