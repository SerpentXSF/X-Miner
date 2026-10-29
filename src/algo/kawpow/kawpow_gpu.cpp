#include "algo/kawpow/kawpow_gpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <sstream>

#include "core/console.hpp"

namespace xm::kawpow {

namespace {

// Shared by every generated module: types, KISS99, Keccak-f800 / f1600, the Ethash DAG generator.
// One hash is worked by 16 consecutive threads ("lanes"), each holding 32 mix registers; thread k of a 16-thread
// group also owns nonce k (its Keccak seed and final hash), so the Keccak work is not repeated per lane.
const char* kHead = R"KRN(
typedef unsigned int u32;
typedef unsigned long long u64;
#if defined(__HIPCC_RTC__) || defined(__HIP__)
#define SHFL(v, src) __shfl((v), (src), 16)
#else
#define SHFL(v, src) __shfl_sync(0xffffffffu, (v), (src), 16)
#endif
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET 0x811c9dc5u
__device__ __forceinline__ u32 fnv1(u32 u, u32 v) { return (u * FNV_PRIME) ^ v; }
__device__ __forceinline__ u32 fnv1a(u32 u, u32 v) { return (u ^ v) * FNV_PRIME; }
__device__ __forceinline__ u32 rotl32(u32 x, u32 n) { n &= 31u; return (x << n) | (x >> ((32u - n) & 31u)); }
__device__ __forceinline__ u32 rotr32(u32 x, u32 n) { n &= 31u; return (x >> n) | (x << ((32u - n) & 31u)); }
__device__ __forceinline__ u32 bswap32(u32 x) {
  return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

__constant__ u64 kRC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull,
    0x000000000000808Bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008Aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000Aull,
    0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};

#define THETA(j) st[j] ^= d0; st[j + 1] ^= d1; st[j + 2] ^= d2; st[j + 3] ^= d3; st[j + 4] ^= d4;
#define CHI(W, j)                                                                                   \
  {                                                                                                 \
    W a0 = st[j], a1 = st[j + 1], a2 = st[j + 2], a3 = st[j + 3], a4 = st[j + 4];                     \
    st[j] = a0 ^ (~a1 & a2); st[j + 1] = a1 ^ (~a2 & a3); st[j + 2] = a2 ^ (~a3 & a4);                \
    st[j + 3] = a3 ^ (~a4 & a0); st[j + 4] = a4 ^ (~a0 & a1);                                       \
  }
#define KECCAK_ROUND(W, BITS, RC)                                                                   \
  {                                                                                                 \
    W bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20], bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];   \
    W bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22], bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];   \
    W bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];                                                  \
    W d0 = bc4 ^ ROL(bc1, 1), d1 = bc0 ^ ROL(bc2, 1), d2 = bc1 ^ ROL(bc3, 1), d3 = bc2 ^ ROL(bc4, 1),   \
      d4 = bc3 ^ ROL(bc0, 1);                                                                       \
    THETA(0) THETA(5) THETA(10) THETA(15) THETA(20)                                                 \
    W t = st[1], b;                                                                                 \
    b = st[10]; st[10] = ROL(t, 1); t = b;   b = st[7];  st[7] = ROL(t, 3);   t = b;                  \
    b = st[11]; st[11] = ROL(t, 6); t = b;   b = st[17]; st[17] = ROL(t, 10); t = b;                  \
    b = st[18]; st[18] = ROL(t, 15); t = b;  b = st[3];  st[3] = ROL(t, 21);  t = b;                  \
    b = st[5];  st[5] = ROL(t, 28); t = b;   b = st[16]; st[16] = ROL(t, 36); t = b;                  \
    b = st[8];  st[8] = ROL(t, 45); t = b;   b = st[21]; st[21] = ROL(t, 55); t = b;                  \
    b = st[24]; st[24] = ROL(t, 2); t = b;   b = st[4];  st[4] = ROL(t, 14);  t = b;                  \
    b = st[15]; st[15] = ROL(t, 27); t = b;  b = st[23]; st[23] = ROL(t, 41); t = b;                  \
    b = st[19]; st[19] = ROL(t, 56); t = b;  b = st[13]; st[13] = ROL(t, 8);  t = b;                  \
    b = st[12]; st[12] = ROL(t, 25); t = b;  b = st[2];  st[2] = ROL(t, 43);  t = b;                  \
    b = st[20]; st[20] = ROL(t, 62); t = b;  b = st[14]; st[14] = ROL(t, 18); t = b;                  \
    b = st[22]; st[22] = ROL(t, 39); t = b;  b = st[9];  st[9] = ROL(t, 61);  t = b;                  \
    b = st[6];  st[6] = ROL(t, 20); t = b;   b = st[1];  st[1] = ROL(t, 44);  t = b;                  \
    CHI(W, 0) CHI(W, 5) CHI(W, 10) CHI(W, 15) CHI(W, 20)                                             \
    st[0] ^= (W)(RC);                                                                               \
  }

#define ROL(x, s) rotl32((x), (s) % 32)
__device__ __forceinline__ void keccakf800(u32 st[25]) {
  for (int r = 0; r < 22; ++r) KECCAK_ROUND(u32, 32, kRC[r])
}
#undef ROL
__device__ __forceinline__ u64 rotl64(u64 x, int s) { return s ? (x << s) | (x >> (64 - s)) : x; }
#define ROL(x, s) rotl64((x), (s))
__device__ __forceinline__ void keccakf1600(u64 st[25]) {
  for (int r = 0; r < 24; ++r) KECCAK_ROUND(u64, 64, kRC[r])
}
#undef ROL

__device__ __forceinline__ u32 kiss(u32& z, u32& w, u32& jsr, u32& jcong) {
  z = 36969u * (z & 0xffffu) + (z >> 16);
  w = 18000u * (w & 0xffffu) + (w >> 16);
  jcong = 69069u * jcong + 1234567u;
  jsr ^= jsr << 17;
  jsr ^= jsr >> 13;
  jsr ^= jsr << 5;
  return (((z << 16) + w) ^ jcong) + jsr;
}

// keccak512 of a 64-byte message (16 words) in place, original Keccak padding.
__device__ __forceinline__ void keccak512_64(u32 w[16]) {
  u64 st[25];
  for (int i = 0; i < 8; ++i) st[i] = (u64)w[2 * i] | ((u64)w[2 * i + 1] << 32);
  st[8] = 0x8000000000000001ull;
  for (int i = 9; i < 25; ++i) st[i] = 0;
  keccakf1600(st);
  for (int i = 0; i < 8; ++i) { w[2 * i] = (u32)st[i]; w[2 * i + 1] = (u32)(st[i] >> 32); }
}

// Ethash dataset items [start, start + count) of 512 bits from the light cache (KawPow: 512 parents).
extern "C" __global__ void xm_ethash_dag(uint4* __restrict__ dag, const uint4* __restrict__ light, u32 light_items,
                                         u32 start, u32 count) {
  const u32 g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) return;
  const u32 idx = start + g;
  u32 mix[16];
  const uint4* c = light + (size_t)(idx % light_items) * 4;
  for (int q = 0; q < 4; ++q) {
    uint4 v = c[q];
    mix[4 * q] = v.x; mix[4 * q + 1] = v.y; mix[4 * q + 2] = v.z; mix[4 * q + 3] = v.w;
  }
  mix[0] ^= idx;
  keccak512_64(mix);
  for (u32 j0 = 0; j0 < DAG_PARENTS; j0 += 16) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const u32 parent = fnv1(idx ^ (j0 + k), mix[k]) % light_items;
      const uint4* p = light + (size_t)parent * 4;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint4 v = p[q];
        mix[4 * q] = fnv1(mix[4 * q], v.x); mix[4 * q + 1] = fnv1(mix[4 * q + 1], v.y);
        mix[4 * q + 2] = fnv1(mix[4 * q + 2], v.z); mix[4 * q + 3] = fnv1(mix[4 * q + 3], v.w);
      }
    }
  }
  keccak512_64(mix);
  uint4* o = dag + (size_t)idx * 4;
  for (int q = 0; q < 4; ++q) o[q] = make_uint4(mix[4 * q], mix[4 * q + 1], mix[4 * q + 2], mix[4 * q + 3]);
}

struct Hdr { u32 w[8]; };
)KRN";

// The search kernel around the generated program (PROGRAM, DAG_MERGE).
const char* kSearch = R"KRN(
extern "C" __global__ __launch_bounds__(TPB) void xm_kawpow_search(const uint4* __restrict__ dag, u32 items2048, Hdr hdr,
                                                                    u64 start_nonce, u64 target_hi,
                                                                    u32* __restrict__ found, u32 max_found) {
  __shared__ u32 c_dag[4096];
  for (u32 i = threadIdx.x; i < 1024; i += TPB) {
    uint4 v = dag[i];
    c_dag[4 * i] = v.x; c_dag[4 * i + 1] = v.y; c_dag[4 * i + 2] = v.z; c_dag[4 * i + 3] = v.w;
  }
  __syncthreads();
  const u32 lane = threadIdx.x & 15u;
  const u64 nonce = start_nonce + (u64)blockIdx.x * TPB + threadIdx.x;
  u32 state2[8];
  {
    u32 st[25];
    for (int i = 0; i < 8; ++i) st[i] = hdr.w[i];
    st[8] = (u32)nonce;
    st[9] = (u32)(nonce >> 32);
INIT_PAD
    keccakf800(st);
    for (int i = 0; i < 8; ++i) state2[i] = st[i];
  }
  u32 digest[8];
#pragma unroll 1
  for (u32 h = 0; h < 16; ++h) {
    u32 mix[32];
    {
      const u32 s0 = SHFL(SEED_LO, h), s1 = SHFL(SEED_HI, h);
      u32 z = fnv1a(FNV_OFFSET, s0), w = fnv1a(z, s1), jsr = fnv1a(w, lane), jcong = fnv1a(jsr, lane);
#pragma unroll
      for (int i = 0; i < 32; ++i) mix[i] = kiss(z, w, jsr, jcong);
    }
#pragma unroll 1
    for (u32 r = 0; r < 64; ++r) {
      const u32 item = SHFL(mix[0], r & 15u) % items2048;
      const uint4 d = dag[(size_t)item * 16 + ((lane ^ r) & 15u)];
      u32 data;
PROGRAM
DAG_MERGE
    }
    u32 lh = FNV_OFFSET;
#pragma unroll
    for (int i = 0; i < 32; ++i) lh = fnv1a(lh, mix[i]);
    u32 dg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) dg[i] = FNV_OFFSET;
#pragma unroll
    for (int l = 0; l < 16; ++l) dg[l % 8] = fnv1a(dg[l % 8], SHFL(lh, l));
    if (h == lane) {
#pragma unroll
      for (int i = 0; i < 8; ++i) digest[i] = dg[i];
    }
  }
  u32 st[25];
FINAL_STATE
  keccakf800(st);
  const u64 top = ((u64)bswap32(st[0]) << 32) | bswap32(st[1]);
  if (top <= target_hi) {
    const u32 slot = atomicAdd(found, 1u);
    if (slot < max_found) {
      found[1 + 2 * slot] = (u32)nonce;
      found[2 + 2 * slot] = (u32)(nonce >> 32);
    }
  }
}
)KRN";

std::string math_expr(const std::string& a, const std::string& b, int op) {
  switch (op) {
    default:
    case 0: return a + " + " + b;
    case 1: return a + " * " + b;
    case 2: return "__umulhi(" + a + ", " + b + ")";
    case 3: return "min(" + a + ", " + b + ")";
    case 4: return "rotl32(" + a + ", " + b + ")";
    case 5: return "rotr32(" + a + ", " + b + ")";
    case 6: return a + " & " + b;
    case 7: return a + " | " + b;
    case 8: return a + " ^ " + b;
    case 9: return "__clz(" + a + ") + __clz(" + b + ")";
    case 10: return "__popc(" + a + ") + __popc(" + b + ")";
  }
}

std::string merge_stmt(const std::string& a, const std::string& b, uint32_t sel) {
  const uint32_t x = (sel >> 16) % 31 + 1;
  switch (sel % 4) {
    default:
    case 0: return a + " = (" + a + " * 33) + " + b + ";";
    case 1: return a + " = (" + a + " ^ " + b + ") * 33;";
    case 2: return a + " = rotl32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";";
    case 3: return a + " = rotr32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";";
  }
}

std::string reg(int i) { return "mix[" + std::to_string(i) + "]"; }

}  // namespace

std::string kernel_source(const Variant& v, uint64_t period, uint32_t tpb) {
  const Program p = make_program(period, v);
  std::ostringstream prog;
  for (const auto& op : p.ops) {
    if (op.cache) {
      prog << "      data = c_dag[" << reg(op.c.src) << " & 4095u];\n      " << merge_stmt(reg(op.c.dst), "data", op.c.sel)
           << "\n";
    } else {
      prog << "      data = " << math_expr(reg(op.m.src1), reg(op.m.src2), math_op(op.m.sel1, v)) << ";\n      "
           << merge_stmt(reg(op.m.dst), "data", op.m.sel2) << "\n";
    }
  }
  const char* words[4] = {"d.x", "d.y", "d.z", "d.w"};
  std::ostringstream dag;
  for (int i = 0; i < kDagLoads; ++i) dag << "      " << merge_stmt(reg(p.dag_dst[i]), words[i], p.dag_sel[i]) << "\n";
  std::string body = kSearch;
  body.replace(body.find("PROGRAM"), 7, prog.str());
  body.replace(body.find("DAG_MERGE"), 9, dag.str());
  // The variant's Keccak: 0.9.4 pads both hashes and carries the first hash's 8 words into the final one; 0.9.2
  // uses zeros, seeds the lanes from bswap(st0) << 32 | bswap(st1) and hashes header + seed + mix at the end.
  std::ostringstream init, fin;
  for (int i = 10; i < 25; ++i) init << "    st[" << i << "] = " << (v.pad_init ? v.pad_init[i - 10] : 0) << "u;\n";
  if (v.v092) {
    fin << "  for (int i = 0; i < 8; ++i) st[i] = hdr.w[i];\n"
        << "  st[8] = bswap32(state2[1]); st[9] = bswap32(state2[0]);\n"
        << "  for (int i = 0; i < 8; ++i) st[10 + i] = digest[i];\n"
        << "  for (int i = 18; i < 25; ++i) st[i] = 0u;\n";
  } else {
    fin << "  for (int i = 0; i < 8; ++i) st[i] = state2[i];\n"
        << "  for (int i = 0; i < 8; ++i) st[8 + i] = digest[i];\n";
    for (int i = 16; i < 25; ++i) fin << "  st[" << i << "] = " << v.pad_final[i - 16] << "u;\n";
  }
  body.replace(body.find("INIT_PAD"), 8, init.str());
  body.replace(body.find("FINAL_STATE"), 11, fin.str());
  std::ostringstream src;
  src << "#define TPB " << tpb << "\n#define DAG_PARENTS " << v.parents << "u\n"
      << "#define SEED_LO " << (v.v092 ? "bswap32(state2[1])" : "state2[0]") << "\n"
      << "#define SEED_HI " << (v.v092 ? "bswap32(state2[0])" : "state2[1]") << "\n"
      << "// " << v.name << " period " << period << "\n" << kHead << body;
  return src.str();
}

std::shared_ptr<const Light> shared_light(const Variant& v, int epoch) {
  static std::mutex mu;
  static std::map<std::pair<std::string, int>, std::shared_ptr<const Light>> cache;  // (variant, epoch)
  const auto key = std::make_pair(std::string(v.name), epoch);
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  for (auto i = cache.begin(); i != cache.end();)  // keep at most the previous epoch of each variant
    i = i->first.first == key.first && i->first.second < epoch - 1 ? cache.erase(i) : std::next(i);
  info("DAG", "building the epoch " + std::to_string(epoch) + " light cache");
  auto l = make_light(epoch, v.parents, v.dataset_init);
  cache[key] = l;
  return l;
}

Engine::Engine(gpu::Device& dev, uint64_t seed, const Variant& v) : dev_(dev), v_(v), rng_(seed) {}

Engine::~Engine() {
  if (next_.valid()) next_.wait();
  dev_.free(d_found_);
  dev_.free(d_dag_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_KAWPOW_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  if (gpu::rtc_library(dev_.info().vendor).empty()) {
    err = dev_.info().vendor == gpu::Vendor::Nvidia
              ? "KawPow needs NVRTC (nvrtc64_120_0.dll / libnvrtc.so.12, shipped next to x-miner)"
              : "KawPow needs hipRTC (part of the AMD HIP runtime / ROCm)";
    return false;
  }
  return true;
}

namespace {

gpu::RtcOutput compile_period(const gpu::DeviceInfo& dev, const Variant* v, uint64_t period, uint32_t tpb) {
  gpu::RtcOutput out;
  std::vector<std::string> opts;
  if (dev.vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  if (!gpu::rtc_compile(dev, kernel_source(*v, period, tpb), std::string(v->name) + "_" + std::to_string(period) + ".cu", opts, out))
    out.code.clear();
  return out;
}

}  // namespace

bool Engine::load_program(uint64_t period, std::string& err) {
  // The next period's program is compiled in the background while this one mines (a compile takes 1-3 s; a
  // period is 3 blocks), so the switch costs only the module load.
  gpu::RtcOutput out;
  if (next_.valid() && next_period_ == period) {
    out = next_.get();
  } else {
    if (next_.valid()) next_.wait();
    out = compile_period(dev_.info(), &v_, period, tpb_);
  }
  if (out.code.empty()) {
    err = "KawPow program " + std::to_string(period) + " did not compile: " + out.log.substr(0, 2000);
    return false;
  }
  module_ = std::move(out);
  if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "kawpow"))) {
    err = dev_.last_error();
    return false;
  }
  period_ = period;
  next_period_ = period + 1;
  next_ = std::async(std::launch::async, compile_period, dev_.info(), &v_, next_period_, tpb_);
  return true;
}

bool generate_dag(gpu::Device& dev, const Light& light, gpu::DevPtr& d_dag, std::string& err) {
  dev.free(d_dag);
  d_dag = 0;
  const size_t dag_bytes = size_t(light.dataset_items) * 128;
  if (dag_bytes + (size_t(256) << 20) > dev.info().memory) {
    err = "the epoch " + std::to_string(light.epoch) + " DAG (" + std::to_string(dag_bytes >> 20) +
          " MB) does not fit in GPU memory";
    return false;
  }
  d_dag = dev.alloc(dag_bytes);
  gpu::DevPtr d_light = dev.alloc(light.cache.size() * 4);
  if (!d_dag || !d_light || !dev.upload(d_light, light.cache.data(), light.cache.size() * 4)) {
    err = dev.last_error();
    dev.free(d_light);
    return false;
  }
  const uint32_t items512 = uint32_t(light.dataset_items) * 2;
  uint32_t light_items = uint32_t(light.cache_items);
  const uint32_t chunk = 1u << 18;  // short launches: the Windows display watchdog kills kernels after ~2 s
  for (uint32_t start = 0; start < items512; start += chunk) {
    uint32_t count = std::min(chunk, items512 - start);
    void* args[] = {&d_dag, &d_light, &light_items, &start, &count};
    if (!dev.launch("xm_ethash_dag", (count + 127) / 128, 1, 1, 128, 1, 1, 0, args) || !dev.wait()) {
      err = dev.last_error();
      dev.free(d_light);
      return false;
    }
  }
  dev.free(d_light);
  return true;
}

std::string kernel_common(int parents, uint32_t tpb) {
  return "#define TPB " + std::to_string(tpb) + "\n#define DAG_PARENTS " + std::to_string(parents) + "u\n" + kHead;
}

bool Engine::build_dag(int epoch, std::string& err) {
  dag_epoch_ = -1;  // generate_dag frees the old DAG first: set again only on success
  if (epoch < 0 || epoch > 4096) {
    err = "implausible block height (DAG epoch " + std::to_string(epoch) + ")";
    return false;
  }
  light_ = shared_light(v_, epoch);
  if (!generate_dag(dev_, *light_, d_dag_, err)) return false;
  dag_epoch_ = epoch;
  return true;
}

bool Engine::set_job(const Job& job, std::string& err) {
  const int epoch = epoch_for_height(v_, job.height);
  const uint64_t period = uint64_t(job.height / v_.period);
  if (period != period_ && !load_program(period, err)) return false;
  if (epoch != dag_epoch_ && !build_dag(epoch, err)) return false;
  if (job.header != job_.header || job.nonce_prefix != job_.nonce_prefix || !have_job_) {
    // Random start below the pool prefix; batches never reach the prefix bits in practice.
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
  uint32_t items2048 = uint32_t(light_->dataset_items / 2);
  struct { uint32_t w[8]; } hdr;
  std::memcpy(hdr.w, job_.header.data(), 32);
  uint64_t target_hi = 0;
  const uint8_t* t = reinterpret_cast<const uint8_t*>(job_.target.data());
  for (int i = 0; i < 8; ++i) target_hi = target_hi << 8 | t[i];
  uint64_t base = counter_;
  uint32_t maxf = kMaxFound;
  void* args[] = {&d_dag_, &items2048, &hdr, &base, &target_hi, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + 2 * kMaxFound);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_kawpow_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args) ||
      !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += batch;
  uint32_t hits = std::min(found[0], kMaxFound);
  if (share_rate_ > 0) {
    const auto now = std::chrono::steady_clock::now();
    share_tokens_ = std::min(share_rate_, share_tokens_ + share_rate_ * std::chrono::duration<double>(now - share_refill_).count());
    share_refill_ = now;
    hits = std::min(hits, uint32_t(share_tokens_));
    share_tokens_ -= hits;
  }
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = uint64_t(found[2 + 2 * k]) << 32 | found[1 + 2 * k];
    const Result r = hash(*light_, job_.height, job_.header, s.nonce, v_);
    if (!less_or_equal(r.final_hash, job_.target)) {
      ++cpu_rejects_;  // the GPU filter only compares the top 64 bits (or the GPU erred)
      continue;
    }
    s.job_id = job_.id;
    s.header = job_.header;
    s.mix = r.mix_hash;
    s.final_hash = r.final_hash;
    s.is_block = job_.block_target != Hash256{} && less_or_equal(r.final_hash, job_.block_target);
    s.work = job_.difficulty;
    shares.push_back(s);
  }
  return batch;
}

}  // namespace xm::kawpow
