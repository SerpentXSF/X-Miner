#include "algo/octopus/octopus_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "algo/kawpow/kawpow_gpu.hpp"
#include "core/console.hpp"

namespace xm::octopus {

namespace {

// Two ways to the polynomial values, then the same DAG mix (octopus_finish): the scalar xm_octopus_search (Horner with
// Shoup's multiply, one thread per nonce) or, on NVIDIA sm_80+, the matrix path (OCT_TC): per-job Vandermonde bytes
// (xm_octopus_vand), per-batch coefficient bytes (xm_octopus_coef), int8 tensor-core products (xm_octopus_eval) and
// the mix (xm_octopus_mix).
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

// SipHash state of a nonce after its 2 + 4 setup rounds; each further round gives one coefficient.
#define SIP_SETUP(nonce)                                              \
  u64 v0 = k0, v1 = k1, v2 = k2, v3 = k3 ^ (nonce);                   \
  SIPROUND;                                                           \
  SIPROUND;                                                           \
  v0 ^= (nonce);                                                      \
  v2 ^= 0xffull;                                                      \
  SIPROUND;                                                           \
  SIPROUND;                                                           \
  SIPROUND;                                                           \
  SIPROUND
#define SIP_KEYS                                                                                    \
  const u64 k0 = (u64)hdr.w[0] | ((u64)hdr.w[1] << 32), k1 = (u64)hdr.w[2] | ((u64)hdr.w[3] << 32); \
  const u64 k2 = (u64)hdr.w[4] | ((u64)hdr.w[5] << 32), k3 = (u64)hdr.w[6] | ((u64)hdr.w[7] << 32)

// Everything after the polynomial: the Keccak seed, the Ethash-style mix of 32 dependent 256-byte DAG pages
// (OCT_LANES lanes per hash: lane k holds mix words W*k .. W*k+W-1 and loads its slice of each page; the hashes of an
// OCT_LANES group in turn), the final Keccak and the target compare. res: this thread's 32 polynomial values.
__device__ __forceinline__ void octopus_finish(const uint4* __restrict__ dag, u32 pages, u32 magic, const Hdr& hdr,
                                               u64 nonce, const u32 (&res)[32], u64 result, u64 target_hi,
                                               u32* __restrict__ found, u32 max_found) {
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

// Scalar path: one thread per nonce; each aligned group of 32 threads is one Octopus nonce group. Thread i of the
// group runs SipHash on its nonce and writes coefficients j * 32 + i (j < 32) to shared memory; every thread then
// evaluates the group's polynomial (Horner, 1024 steps) at its 32 points together, with Shoup's multiply: q = hi(acc *
// xs), acc * x - q * p is in [0, 2p) for any 32-bit acc, so the accumulators stay below 3p without a reduction inside
// the loop (one high and two low multiply-adds per step).
extern "C" __global__ __launch_bounds__(TPB) void xm_octopus_search(const uint4* __restrict__ dag, u32 pages, u32 magic,
                                                                     const uint2* __restrict__ pts, Hdr hdr,
                                                                     u64 start_nonce, u64 target_hi,
                                                                     u32* __restrict__ found, u32 max_found) {
  __shared__ u32 coef[TPB / 32][1024];
  const u32 lane = threadIdx.x & 31u, grp = threadIdx.x >> 5;
  const u64 nonce = start_nonce + (u64)blockIdx.x * TPB + threadIdx.x;  // start_nonce is a multiple of 32
  SIP_KEYS;
  {
    SIP_SETUP(nonce);
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
  octopus_finish(dag, pages, magic, hdr, nonce, res, result, target_hi, found, max_found);
}

#if OCT_TC
// Matrix path. The values a group needs, P_g(x_k) for all 1024 header points k, are the product of the group's
// coefficient row c_g (1024) and the Vandermonde matrix V[i][k] = x_k^i mod p. Both factors are below 2^20 and are
// split into three bytes, so the product is nine u8 x u8 -> s32 matrix products, summed by byte shift (T_s, s = a + b)
// and reduced: P = sum_s 2^(8s) T_s mod p. Each T_s sum is below 3 * 1024 * 255^2 < 2^28.

// V as three byte planes, point-major: vand[b][k][i] = byte b of x_k^i. One thread per (point, 32 powers).
extern "C" __global__ void xm_octopus_vand(const uint2* __restrict__ pts, unsigned char* __restrict__ vand) {
  const u32 t = blockIdx.x * blockDim.x + threadIdx.x;  // 32768 threads
  const u32 k = t >> 5, c = t & 31u;
  const u64 x = pts[k].x;
  u64 pw = 1, b = x;
  for (u32 e = 32 * c; e; e >>= 1, b = b * b % OCT_P)
    if (e & 1) pw = pw * b % OCT_P;
  for (u32 i = 32 * c; i < 32 * c + 32; ++i) {
#pragma unroll
    for (int l = 0; l < 3; ++l) vand[(size_t)l << 20 | (size_t)k << 10 | i] = (unsigned char)(pw >> (8 * l));
    pw = pw * x % OCT_P;
  }
}

// Coefficients as three byte planes, group-major: coef[a][g][j * 32 + i] = byte a of coefficient j * 32 + i of group
// g (SipHash round j + 1 of the group's nonce i). One thread per nonce; groups = nonces / 32 of this batch.
extern "C" __global__ __launch_bounds__(256) void xm_octopus_coef(Hdr hdr, u64 start_nonce, u32 groups,
                                                                  unsigned char* __restrict__ coef) {
  const u32 t = blockIdx.x * 256u + threadIdx.x;
  const u32 g = t >> 5, lane = t & 31u;
  SIP_KEYS;
  SIP_SETUP(start_nonce + t);
  unsigned char* row = coef + (size_t)g * 1024 + lane;
  const size_t plane = (size_t)groups * 1024;
#pragma unroll 4
  for (u32 j = 0; j < 32; ++j) {
    SIPROUND;
    const u32 d = fast_mod((u32)(v0 ^ v1 ^ v2 ^ v3), OCT_P, 4161u);
    row[j * 32] = (unsigned char)d;
    row[plane + j * 32] = (unsigned char)(d >> 8);
    row[2 * plane + j * 32] = (unsigned char)(d >> 16);
  }
}

#ifndef TC_BM
#define TC_BM 32   // groups per tile (16 per warp row)
#endif
#define TC_BN 64   // points per tile (32 per warp column)
#define TC_LD 48   // shared-memory row stride in bytes (32 used; 48 keeps fragment loads conflict-free)
#define TC_THREADS (TC_BM / 16 * TC_BN / 32 * 32)

__device__ __forceinline__ void mma_u8(int (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// vals[g][k] = P_g(x_k). Tiles of TC_BM groups x TC_BN points, warps as (TC_BM / 16) x 2, a warp 16 x 32; a block
// walks tiles blockIdx.x, + gridDim.x, ... (points fastest, so concurrent blocks share coefficient rows in L2). The
// host launches two blocks per SM: they stay resident beside the DAG mix running on the other stream. (A cp.async
// double-buffered version with 64-coefficient stages was slower beside the mix: 26.8 vs 31.4 MH/s on an RTX 3060;
// its 46 KB of shared memory per block takes L1 from the mix.)
extern "C" __global__ __launch_bounds__(TC_THREADS) void xm_octopus_eval(const unsigned char* __restrict__ coef,
                                                                         const unsigned char* __restrict__ vand,
                                                                         u32 groups, u32* __restrict__ vals) {
  __shared__ __align__(16) unsigned char sa[3][TC_BM][TC_LD];
  __shared__ __align__(16) unsigned char sb[3][TC_BN][TC_LD];
  const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  const u32 gid = lane >> 2, tig = lane & 3u;
  const u32 wm = warp % (TC_BM / 16), wn = warp / (TC_BM / 16);
  const size_t aplane = (size_t)groups * 1024;
  const u32 tiles = groups / TC_BM * (1024 / TC_BN);
#pragma unroll 1
  for (u32 tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const u32 g0 = tile / (1024 / TC_BN) * TC_BM, n0 = tile % (1024 / TC_BN) * TC_BN;
    int acc[5][4][4];
#pragma unroll
    for (int s = 0; s < 5; ++s)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[s][n][r] = 0;
#pragma unroll 1
    for (u32 k0 = 0; k0 < 1024; k0 += 32) {
      // 3 planes x rows x 2 halves of 16 bytes per operand.
      for (u32 q = threadIdx.x; q < 3 * TC_BM * 2; q += TC_THREADS) {
        const u32 pl = q / (TC_BM * 2), row = (q >> 1) % TC_BM, half = q & 1u;
        *(uint4*)&sa[pl][row][half * 16] = *(const uint4*)(coef + pl * aplane + (size_t)(g0 + row) * 1024 + k0 + half * 16);
      }
      for (u32 q = threadIdx.x; q < 3 * TC_BN * 2; q += TC_THREADS) {
        const u32 pl = q / (TC_BN * 2), row = (q >> 1) % TC_BN, half = q & 1u;
        *(uint4*)&sb[pl][row][half * 16] = *(const uint4*)(vand + ((size_t)pl << 20) + (size_t)(n0 + row) * 1024 + k0 + half * 16);
      }
      __syncthreads();
      u32 af[3][4];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        const u32 r0 = wm * 16 + gid;
        af[a][0] = *(const u32*)&sa[a][r0][tig * 4];
        af[a][1] = *(const u32*)&sa[a][r0 + 8][tig * 4];
        af[a][2] = *(const u32*)&sa[a][r0][16 + tig * 4];
        af[a][3] = *(const u32*)&sa[a][r0 + 8][16 + tig * 4];
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const u32 c = wn * 32 + n * 8 + gid;
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          const u32 b0 = *(const u32*)&sb[b][c][tig * 4], b1 = *(const u32*)&sb[b][c][16 + tig * 4];
#pragma unroll
          for (int a = 0; a < 3; ++a) mma_u8(acc[a + b][n], af[a], b0, b1);
        }
      }
      __syncthreads();
    }
    // C fragment: rows gid / gid + 8, columns 2 tig, 2 tig + 1.
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        u64 v = (u64)(u32)acc[0][n][r] + ((u64)(u32)acc[1][n][r] << 8) + ((u64)(u32)acc[2][n][r] << 16) +
                ((u64)(u32)acc[3][n][r] << 24) + ((u64)(u32)acc[4][n][r] << 32);
        const u32 g = g0 + wm * 16 + gid + (r >> 1) * 8, k = n0 + wn * 32 + n * 8 + tig * 2 + (r & 1);
        vals[(size_t)g * 1024 + k] = (u32)(v % OCT_P);
      }
  }
}

// The DAG part per nonce, with the polynomial values from xm_octopus_eval: nonce lane i of group g uses points
// i + 32 t (t < 32).
extern "C" __global__ __launch_bounds__(TPB) void xm_octopus_mix(const uint4* __restrict__ dag, u32 pages, u32 magic,
                                                                  const u32* __restrict__ vals, Hdr hdr,
                                                                  u64 start_nonce, u64 target_hi,
                                                                  u32* __restrict__ found, u32 max_found) {
  const u32 t = blockIdx.x * TPB + threadIdx.x;
  const u32* v = vals + (size_t)(t >> 5) * 1024 + (t & 31u);
  u32 res[32];
  u64 result = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    res[i] = v[32 * i];
    result = result * 0x01000193ull ^ res[i];
  }
  octopus_finish(dag, pages, magic, hdr, start_nonce + t, res, result, target_hi, found, max_found);
}
#endif
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
  dev_.free(d_vand_);
  for (int i = 0; i < 2; ++i) {
    dev_.free(d_coef_[i]);
    dev_.free(d_vals_[i]);
  }
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_OCTOPUS_TPB")) tpb_ = static_cast<uint32_t>(std::atoi(e));  // developer tuning
  if (const char* e = std::getenv("XM_OCTOPUS_LANES")) lanes_ = static_cast<uint32_t>(std::atoi(e));
  mul24_ = dev_.info().vendor == gpu::Vendor::Amd;
  // The matrix path needs mma.sync m16n8k32 u8 (sm_80+). XM_OCTOPUS_TC=0: the scalar path.
  const std::string& arch = dev_.info().arch;
  tc_ = dev_.info().vendor == gpu::Vendor::Nvidia && arch.rfind("sm_", 0) == 0 && std::atoi(arch.c_str() + 3) >= 80;
  if (const char* e = std::getenv("XM_OCTOPUS_TC")) tc_ = tc_ && std::atoi(e) != 0;
  // Developer tuning: groups per matrix tile (32: 4 warps, 64: 8 warps) and the matrix grid (blocks; 0 = one per tile).
  if (const char* e = std::getenv("XM_OCTOPUS_BM")) tc_bm_ = std::atoi(e) == 32 ? 32 : 64;
  if (const char* e = std::getenv("XM_OCTOPUS_EGRID")) tc_grid_ = static_cast<uint32_t>(std::atoi(e));
  if (const char* e = std::getenv("XM_OCTOPUS_MUL24")) mul24_ = mul24_ && std::atoi(e) != 0;
  if (tpb_ % 32 || lanes_ < 4 || lanes_ > 16 || (lanes_ & (lanes_ - 1))) {
    err = "XM_OCTOPUS_TPB must be a multiple of 32 and XM_OCTOPUS_LANES 4, 8 or 16";
    return false;
  }
  d_found_ = dev_.alloc((1 + 2 * kMaxFound) * 4);
  d_pts_ = dev_.alloc(sizeof(uint32_t) * 2 * kN);
  if (tc_) d_vand_ = dev_.alloc(3u << 20);
  if (!d_found_ || !d_pts_ || (tc_ && !d_vand_)) {
    err = dev_.last_error();
    return false;
  }
  std::vector<std::string> opts;
  if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
  const std::string src =
      "#define OCT_LANES " + std::to_string(lanes_) + "\n#define OCT_MUL24 " + (mul24_ ? "1" : "0") +
      "\n#define OCT_TC " + (tc_ ? "1" : "0") + "\n#define TC_BM " + std::to_string(tc_bm_) + "\n" +
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
    void* vargs[] = {&d_pts_, &d_vand_};
    if (!dev_.upload(d_pts_, packed.data(), packed.size() * 4) ||
        (tc_ && !dev_.launch("xm_octopus_vand", kN * 32 / 256, 1, 1, 256, 1, 1, 0, vargs))) {
      err = dev_.last_error();
      return false;
    }
    prepared_ = false;  // values computed ahead belong to the old header
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
  // The matrix path works in blocks of 64 groups (2048 nonces), the scalar one in blocks of tpb_ nonces.
  const uint32_t unit = tc_ ? std::max(2048u, tpb_) : tpb_;
  batch = std::max(batch / unit, 1u) * unit;
  if (tc_ && batch > tc_batch_) {  // coefficient bytes (3 KB) and values (4 KB) per group of 32 nonces
    bool okb = true;
    for (int i = 0; i < 2; ++i) {
      dev_.free(d_coef_[i]);
      dev_.free(d_vals_[i]);
      d_coef_[i] = dev_.alloc(size_t(batch) / 32 * 3072);
      d_vals_[i] = dev_.alloc(size_t(batch) / 32 * 4096);
      okb = okb && d_coef_[i] && d_vals_[i];
    }
    tc_batch_ = okb ? batch : 0;
    prepared_ = false;
    if (!tc_batch_) {
      err = dev_.last_error();
      return 0;
    }
  }
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
  bool ok = dev_.memset(d_found_, 0, 4);
  if (tc_) {
    static const bool prof = std::getenv("XM_OCTOPUS_PROF") != nullptr;  // developer: per-kernel times (serial)
    static double tsum[3];
    static int tn;
    auto t = std::chrono::steady_clock::now();
    auto lap = [&](int i) {
      if (!prof) return true;
      const bool w = dev_.wait();
      const auto now = std::chrono::steady_clock::now();
      tsum[i] += std::chrono::duration<double, std::milli>(now - t).count();
      t = now;
      return w;
    };
    uint32_t groups = batch / 32;
    const uint32_t tiles = groups / tc_bm_ * (kN / 64);
    // Default: two resident matrix blocks per SM beside the mix (RTX 3060: 31.4 MH/s vs 26.3 with one block per tile).
    const uint32_t want = tc_grid_ ? tc_grid_ : 2 * uint32_t(std::max(dev_.info().compute_units, 1));
    const uint32_t egrid = std::min(want, tiles);
    // Polynomial values of the batch at `at` into buffer i: coefficients, then the matrix products.
    auto values = [&](uint64_t at, int i) {
      void* cargs[] = {&hdr, &at, &groups, &d_coef_[i]};
      void* eargs[] = {&d_coef_[i], &d_vand_, &groups, &d_vals_[i]};
      return dev_.launch("xm_octopus_coef", batch / 256, 1, 1, 256, 1, 1, 0, cargs) && lap(0) &&
             dev_.launch("xm_octopus_eval", egrid, 1, 1, tc_bm_ / 16 * 64, 1, 1, 0, eargs) && lap(1);
    };
    if (!prepared_ || prep_base_ != base || prep_batch_ != batch) ok = ok && values(base, cur_);
    void* margs[] = {&d_dag_, &pages, &magic, &d_vals_[cur_], &hdr, &base, &target_hi, &d_found_, &maxf};
    ok = ok && dev_.launch("xm_octopus_mix", batch / tpb_, 1, 1, tpb_, 1, 1, 0, margs) && lap(2);
    // The next batch's values on the second stream, overlapping the mix (in order when there is no second stream).
    const bool two = !prof && dev_.set_stream(1);
    ok = ok && values(base + batch, cur_ ^ 1);
    if (two) dev_.set_stream(0);
    prepared_ = ok;
    prep_base_ = base + batch;
    prep_batch_ = batch;
    cur_ ^= 1;
    if (prof && ++tn % 20 == 0) {
      char b[160];
      std::snprintf(b, sizeof b, "per batch of %u: coef %.2f ms, eval %.2f ms, mix %.2f ms", batch, tsum[0] / tn,
                    tsum[1] / tn, tsum[2] / tn);
      info("PROF", b);
    }
  } else {
    ok = ok && dev_.launch("xm_octopus_search", batch / tpb_, 1, 1, tpb_, 1, 1, 0, args);
  }
  if (!ok || !dev_.wait() || !dev_.download(found.data(), d_found_, found.size() * 4)) {
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
