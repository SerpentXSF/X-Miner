// Beam BeamHash III solver (compiled at run time; the host prepends NB_BITS, CAP, TPB, COOP_STORE and LDS_BUCKET).
// See src/algo/beam/beam.hpp for the algorithm. 2^25 elements, five rounds of 24-bit collisions.
//
// Elements live in NB buckets of CAP slots, by the low NB_BITS bits of their 24-bit collision key (the low bits of the
// MIXED low word: every element is stored already mixed for the round that reads it, so its bucket is known when it is
// written). One block per bucket links the bucket's elements by the other key bits and combines every colliding pair.
// - NVIDIA (LDS_BUCKET): 2^16 buckets of ~512 elements. The block copies its bucket into shared memory with coalesced
//   loads, lists the colliding pairs, and combines them a warp at a time. With 4096 buckets of ~8192 elements read
//   from global memory, the partner loads went to DRAM again (the buckets in flight were ~10x an RTX 3060's L2).
// - AMD: 4096 buckets of ~8192 elements read from global memory; its large last-level cache serves the re-reads and
//   merges the scattered stores, and 2^16 buckets were slower there.
// Element layout (32-bit words): work bits, then the leaf indices the next round needs (rounds 1-4: the whole
// subtree, 1/2/4/8; packed in round 3). Round 5 keeps only the mixed low word, the first leaf index and the positions
// of its two parents in the round-4 buffer, from which the host rebuilds the 32 indices of a solution.
typedef unsigned int u32;
typedef unsigned long long u64;

#ifndef NB_BITS
#define NB_BITS 16
#endif
#define NB (1u << NB_BITS)
#define SLOTS (1u << (24 - NB_BITS))
#ifndef LDS_BUCKET
#define LDS_BUCKET 1  // NVIDIA: small buckets processed in shared memory; AMD (host sets 0): large buckets, see round_body
#endif
#ifndef COOP_STORE
#define COOP_STORE 1  // NVIDIA: whole elements per warp store (see warp_store); AMD (host sets 0): per-thread stores
#endif
#define PCAP 1024  // colliding pairs per bucket (mean ~512)

#if defined(__HIPCC__)
// Two funnel shifts (v_alignbit_b32) per rotation (n is a constant everywhere here).
__device__ __forceinline__ u64 rotl(u64 x, int n) {
  u32 lo = (u32)x, hi = (u32)(x >> 32);
  if (n >= 32) {
    const u32 t = lo;
    lo = hi;
    hi = t;
    n -= 32;
  }
  if (n == 0) return ((u64)hi << 32) | lo;
  return ((u64)__builtin_amdgcn_alignbit(hi, lo, 32 - n) << 32) | __builtin_amdgcn_alignbit(lo, hi, 32 - n);
}
#else
__device__ __forceinline__ u64 rotl(u64 x, int n) { return (x << n) | (x >> (64 - n)); }
#endif

#define SIPROUND                                                                   \
  do {                                                                             \
    v0 += v1; v2 += v3; v1 = rotl(v1, 13); v3 = rotl(v3, 16); v1 ^= v0; v3 ^= v2; \
    v0 = rotl(v0, 32); v2 += v1; v0 += v3; v1 = rotl(v1, 17); v3 = rotl(v3, 21);  \
    v1 ^= v2; v3 ^= v0; v2 = rotl(v2, 32);                                         \
  } while (0)

#ifdef __HIPCC__  // wave32 on RDNA
#define SHFL(x, l) __shfl((x), (l), 32)
#define WARP_SYNC() __builtin_amdgcn_wave_barrier()
#else
#define SHFL(x, l) __shfl_sync(0xffffffffu, (x), (l))
#define WARP_SYNC() __syncwarp()
#endif

struct PrePow {
  u64 v[4];
};

__device__ __forceinline__ u64 siphash(const PrePow& p, u64 nonce) {
  u64 v0 = p.v[0], v1 = p.v[1], v2 = p.v[2], v3 = p.v[3] ^ nonce;
  SIPROUND;
  SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND;
  SIPROUND;
  SIPROUND;
  SIPROUND;
  return v0 ^ v1 ^ v2 ^ v3;
}

// Round geometry (r = 1..5): bits before round r's mix, bits after its combination, leaf indices its mix reads.
template <int R> struct G {
  static constexpr int MIX_LEN = 448 - (R - 1) * 24 - (R == 5 ? 64 : 0);
  static constexpr int MIX_IDX = R == 1 ? 1 : R == 2 ? 2 : R == 3 ? 4 : R == 4 ? 6 : 9;
  static constexpr int OUT_LEN = R == 5 ? 24 : 448 - R * 24 - (R == 4 ? 64 : 0);
  static constexpr int WK = (MIX_LEN + 31) / 32;          // work words of round R's input
  static constexpr int CI = R == 5 ? 9 : (1 << (R - 1));   // leaf indices carried by round R's input
  // Round 3's leaf indices are packed (25 bits each, at MIX_LEN + 25 k as the mix appends them), so its elements take
  // 64 bytes instead of 80.
  static constexpr bool PACKED = R == 3;
  // words per stored element (16-byte multiple); round 5 keeps only what beam_final and the host read (R5 below)
  static constexpr int STRIDE = R <= 3 ? 16 : R == 5 ? 8 : 20;
};

// 25 bits of x from bit `bit` (constant after unrolling).
__device__ __forceinline__ u32 bits25(const u32* x, int bit) {
  const int w = bit >> 5, sh = bit & 31;
  u32 v = x[w] >> sh;
  if (sh > 7) v |= x[w + 1] << (32 - sh);
  return v & 0x1ffffffu;
}
__device__ __forceinline__ void put25(u32* x, int bit, u32 v) {
  const int w = bit >> 5, sh = bit & 31;
  x[w] |= v << sh;
  if (sh > 7) x[w + 1] |= v >> (32 - sh);
}

// Round 5's stored element (words): the mixed low word, the first leaf index, the parents' positions.
#define R5_LOW 0
#define R5_FIRST 2
#define R5_REF 3

// The mixed low word of w (work bits of length LEN, words above zero) with n indices appended at LEN + 25 k.
template <int LEN, int N>
__device__ __forceinline__ u64 mix(const u64* w, const u32* idx) {
  u64 t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = i * 64 < LEN ? w[i] : 0;
#pragma unroll
  for (int k = 0; k < N; ++k) {
    const int pos = LEN + 25 * k;
    if (pos < 512) {
      const u64 v = idx[k];
      t[pos >> 6] |= v << (pos & 63);
      if ((pos & 63) > 39 && (pos >> 6) < 7) t[(pos >> 6) + 1] |= v >> (64 - (pos & 63));
    }
  }
  u64 r = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) r += rotl(t[i], (29 * (i + 1)) & 63);
  return rotl(r, 24);
}

// Element I/O: 32-bit words to 64-bit work words.
template <int R>
__device__ __forceinline__ void load_elem(const u32* __restrict__ e, u64* w, u32* idx) {
  constexpr int WK = G<R>::WK;
  u32 x[G<R>::STRIDE];
  const uint4* q = reinterpret_cast<const uint4*>(e);
#pragma unroll
  for (int i = 0; i < G<R>::STRIDE / 4; ++i) {
    const uint4 v = q[i];
    x[4 * i] = v.x;
    x[4 * i + 1] = v.y;
    x[4 * i + 2] = v.z;
    x[4 * i + 3] = v.w;
  }
#pragma unroll
  for (int i = 0; i < 7; ++i)
    w[i] = (2 * i < WK ? (u64)x[2 * i] : 0) | (2 * i + 1 < WK ? (u64)x[2 * i + 1] << 32 : 0);
  if (G<R>::PACKED) {
#pragma unroll
    for (int i = 0; i < G<R>::CI; ++i) idx[i] = bits25(x, G<R>::MIX_LEN + 25 * i);
  } else {
#pragma unroll
    for (int i = 0; i < G<R>::CI; ++i) idx[i] = x[WK + i];
  }
}

// Round 1 input: 2^25 seeds, already mixed for round 1. Layout: 14 work words, the index, a zero word.
#define GEN_TPB 256
extern "C" __global__ __launch_bounds__(GEN_TPB) void beam_gen(PrePow p, u32* __restrict__ out, u32* __restrict__ cnt) {
#if COOP_STORE
  // [5]: 80-byte lane stride, so the 16-byte writes and reads are free of bank conflicts
  __shared__ uint4 stage[GEN_TPB / 32][32][5];
#endif
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  u64 w[8];
#pragma unroll
  for (u32 j = 0; j < 7; ++j) w[j] = siphash(p, (u64)((i << 3) + j));
  w[7] = 0;
  const u32 idx[1] = {i};
  w[0] = mix<G<1>::MIX_LEN, G<1>::MIX_IDX>(w, idx);
  const u32 b = (u32)w[0] & (NB - 1);
  const u32 pos = atomicAdd(&cnt[b], 1u);
  const uint4 v[4] = {make_uint4((u32)w[0], (u32)(w[0] >> 32), (u32)w[1], (u32)(w[1] >> 32)),
                      make_uint4((u32)w[2], (u32)(w[2] >> 32), (u32)w[3], (u32)(w[3] >> 32)),
                      make_uint4((u32)w[4], (u32)(w[4] >> 32), (u32)w[5], (u32)(w[5] >> 32)),
                      make_uint4((u32)w[6], (u32)(w[6] >> 32), i, 0u)};
#if !COOP_STORE
  if (pos < CAP) {
    uint4* q = reinterpret_cast<uint4*>(out + ((size_t)b * CAP + pos) * G<1>::STRIDE);
#pragma unroll
    for (int k = 0; k < 4; ++k) q[k] = v[k];
  }
#else
  // Whole elements per warp store (see warp_store), through shared memory here.
  uint4* mine = stage[threadIdx.x >> 5][threadIdx.x & 31];
#pragma unroll
  for (int k = 0; k < 4; ++k) mine[k] = v[k];
  WARP_SYNC();
  const u32 lane = threadIdx.x & 31, part = lane & 3;
#pragma unroll
  for (u32 k = 0; k < 4; ++k) {
    const u32 e = 8 * k + (lane >> 2);
    const u32 eb = SHFL(b, e), ep = SHFL(pos, e);
    if (ep < CAP)
      reinterpret_cast<uint4*>(out + ((size_t)eb * CAP + ep) * G<1>::STRIDE)[part] = stage[threadIdx.x >> 5][e][part];
  }
#endif
}

// Store every lane's element x (ST words) at its dst (0: nothing) with ST / 4 lanes per element, so one warp store
// writes 32 / (ST / 4) whole elements instead of 32 scattered 16-byte pieces (the scattered stores held the rounds at
// the memory pipeline's queue: round 1 25 -> 7.8 ms without them on an RTX 3060). Data moves by shuffles.
template <int ST>
__device__ __forceinline__ void warp_store(u64 dst, const u32 (&x)[ST]) {
  constexpr int P = ST / 4;
  if (!COOP_STORE) {
    if (dst)
#pragma unroll
      for (int q = 0; q < P; ++q)
        reinterpret_cast<uint4*>(dst)[q] = make_uint4(x[4 * q], x[4 * q + 1], x[4 * q + 2], x[4 * q + 3]);
    return;
  }
  const u32 lane = threadIdx.x & 31;
#pragma unroll
  for (int k = 0; k < P; ++k) {
    const u32 g = k * 32 + lane, e = g / P, part = g - e * P;
    const u64 d = ((u64)SHFL((u32)(dst >> 32), e) << 32) | SHFL((u32)dst, e);
    u32 v[4] = {0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < P; ++r)
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const u32 y = SHFL(x[4 * r + t], e);
        if (part == (u32)r) v[t] = y;
      }
    if (d) reinterpret_cast<uint4*>(d)[part] = make_uint4(v[0], v[1], v[2], v[3]);
  }
}

#if LDS_BUCKET
// A bucket in shared memory: the first LW words of every element (stride LS, odd: thread-per-element reads are free of
// bank conflicts), head[slot] = the last element with that slot, nxt[i] = the previous one, and the list of the
// bucket's colliding pairs (i | j << 16).
template <int R> struct Lds {
  static constexpr int ST = G<R>::STRIDE;
  // words kept: the whole element if packed, else the work words, plus the indices if they fit
  static constexpr int LW = G<R>::PACKED ? ST : G<R>::WK + G<R>::CI <= 17 ? G<R>::WK + G<R>::CI : G<R>::WK;
  static constexpr int LS = LW | 1;
  u32 w[CAP * LS];
  u32 head[SLOTS];
  unsigned short nxt[CAP];
  u32 npairs;
  u32 pairs[PCAP];
};

// Copy bucket `base` (n elements) into shared memory (16-byte loads over the contiguous bucket) and link the chains.
template <int R>
__device__ __forceinline__ void load_bucket(Lds<R>& s, const u32* __restrict__ base, u32 n) {
  constexpr int ST = Lds<R>::ST, LW = Lds<R>::LW, LS = Lds<R>::LS, Q = ST / 4;
  for (u32 k = threadIdx.x; k < SLOTS; k += blockDim.x) s.head[k] = 0xffffu;
  if (threadIdx.x == 0) s.npairs = 0;
  const uint4* q = reinterpret_cast<const uint4*>(base);
  for (u32 t = threadIdx.x; t < n * Q; t += blockDim.x) {
    const u32 e = t / Q, w = (t - e * Q) * 4;
    if (w >= LW) continue;
    const uint4 v = q[t];
    u32* d = s.w + e * LS + w;
    d[0] = v.x;
    if (w + 1 < LW) d[1] = v.y;
    if (w + 2 < LW) d[2] = v.z;
    if (w + 3 < LW) d[3] = v.w;
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    const u32 slot = (s.w[i * LS] >> NB_BITS) & (SLOTS - 1);
    s.nxt[i] = (unsigned short)atomicExch(&s.head[slot], i);
  }
  __syncthreads();
}

// Element i of the bucket: work words from shared memory, leaf indices from shared memory or (if they did not fit)
// from the bucket in global memory (just read, so cached).
template <int R>
__device__ __forceinline__ void get_elem(const Lds<R>& s, const u32* __restrict__ base, u32 i, u64* w, u32* idx) {
  constexpr int WK = G<R>::WK, LW = Lds<R>::LW, LS = Lds<R>::LS;
  const u32* x = s.w + i * LS;
#pragma unroll
  for (int k = 0; k < 7; ++k)
    w[k] = (2 * k < WK ? (u64)x[2 * k] : 0) | (2 * k + 1 < WK ? (u64)x[2 * k + 1] << 32 : 0);
  w[7] = 0;  // (packed: w also holds index bits above MIX_LEN; the combination masks them off)
  if (G<R>::PACKED) {
#pragma unroll
    for (int k = 0; k < G<R>::CI; ++k) idx[k] = bits25(x, G<R>::MIX_LEN + 25 * k);
  } else if (LW > WK) {
#pragma unroll
    for (int k = 0; k < G<R>::CI; ++k) idx[k] = x[WK + k];
  } else {
    const uint4* g = reinterpret_cast<const uint4*>(base + (size_t)i * G<R>::STRIDE + WK);  // WK % 4 == 0 here
#pragma unroll
    for (int k = 0; k < G<R>::CI / 4; ++k) {
      const uint4 v = g[k];
      idx[4 * k] = v.x;
      idx[4 * k + 1] = v.y;
      idx[4 * k + 2] = v.z;
      idx[4 * k + 3] = v.w;
    }
  }
}
#endif

// Combine element i (wa, ia) and element j (wb, ib) of bucket bk into round R+1's element x; returns its address in
// round R+1's buckets (0: no output).
template <int R>
__device__ __forceinline__ u64 combine(const u64* wa, const u32* ia, const u64* wb, const u32* ib, u32 bk, u32 i, u32 j,
                                       u32* __restrict__ out, u32* __restrict__ cout, u32 (&x)[G<R + 1>::STRIDE]) {
  // (a ^ b) >> 24, masked to OUT_LEN bits
  u64 xx[8];
#pragma unroll
  for (int k = 0; k < 7; ++k) xx[k] = wa[k] ^ wb[k];
  xx[7] = 0;
  u64 nw[8];
  u64 any = 0;
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    nw[k] = (xx[k] >> 24) | (xx[k + 1] << 40);
    constexpr int L = G<R>::OUT_LEN;
    const int lo = 64 * k;
    if (L <= lo) nw[k] = 0;
    else if (L < lo + 64) nw[k] &= (1ull << (L - lo)) - 1;
    any |= nw[k];
  }
  nw[7] = 0;
  if (!any) return 0;  // identical subtrees: duplicate indices, never a valid solution
  const bool a_first = ia[0] < ib[0];
  u32 ni[G<R + 1>::CI];
#pragma unroll
  for (int k = 0; k < G<R + 1>::CI; ++k) {
    const int fromA = k < G<R>::CI;
    const u32 va = fromA ? ia[k] : ib[k - G<R>::CI];
    const u32 vb = fromA ? ib[k] : ia[k - G<R>::CI];
    ni[k] = a_first ? va : vb;
  }
  nw[0] = mix<G<R + 1>::MIX_LEN, G<R + 1>::MIX_IDX>(nw, ni);
  const u32 ob = (u32)nw[0] & (NB - 1);
  const u32 pos = atomicAdd(&cout[ob], 1u);
  // Element layout: work words, then the leaf indices. Round 5 (R5_*): the mixed low word, the first leaf index and
  // the parents' positions in the round-4 buffer (the rest of an element only fed the mix).
#pragma unroll
  for (int k = 0; k < G<R + 1>::STRIDE; ++k) x[k] = 0;
  if (R + 1 == 5) {
    x[R5_LOW] = (u32)nw[0];
    x[R5_LOW + 1] = (u32)(nw[0] >> 32);
    x[R5_FIRST] = ni[0];
    x[R5_REF] = bk * CAP + (a_first ? i : j);
    x[R5_REF + 1] = bk * CAP + (a_first ? j : i);
  } else {
    constexpr int WK1 = R + 1 == 5 ? 0 : G<R + 1>::WK, CI1 = R + 1 == 5 ? 0 : G<R + 1>::CI;  // (round 5: dead code)
#pragma unroll
    for (int k = 0; k < WK1; ++k) x[k] = (u32)(nw[k >> 1] >> (32 * (k & 1)));
    if (G<R + 1>::PACKED) {
#pragma unroll
      for (int k = 0; k < CI1; ++k) put25(x, G<R + 1>::MIX_LEN + 25 * k, ni[k]);
    } else {
#pragma unroll
      for (int k = 0; k < CI1; ++k) x[WK1 + k] = ni[k];
    }
  }
  return pos < CAP ? (u64)(out + ((size_t)ob * CAP + pos) * G<R + 1>::STRIDE) : 0;
}

#if LDS_BUCKET
// Rounds 1-4: combine every colliding pair of each input bucket into round R+1's buckets.
template <int R>
__device__ __forceinline__ void round_body(const u32* __restrict__ in, const u32* __restrict__ cin,
                                           u32* __restrict__ out, u32* __restrict__ cout) {
  static_assert(Lds<R>::LW > G<R>::WK || (G<R>::WK % 4 == 0 && G<R>::CI % 4 == 0), "index words must be uint4s");
  __shared__ Lds<R> s;
  const u32 b = blockIdx.x;
  const u32 n = min(cin[b], (u32)CAP);
  const u32* base = in + (size_t)b * CAP * G<R>::STRIDE;
  load_bucket<R>(s, base, n);
  for (u32 i = threadIdx.x; i < n; i += blockDim.x)
    for (u32 j = s.nxt[i]; j != 0xffffu; j = s.nxt[j]) {
      const u32 k = atomicAdd(&s.npairs, 1u);
      if (k < PCAP) s.pairs[k] = i | j << 16;
    }
  __syncthreads();
  const u32 np = min(s.npairs, (u32)PCAP);
  for (u32 p0 = threadIdx.x & ~31u; p0 < np; p0 += blockDim.x) {  // warp-uniform loop: every lane shuffles
    const u32 p = p0 + (threadIdx.x & 31);
    u64 dst = 0;
    u32 x[G<R + 1>::STRIDE];
    if (p < np) {
      const u32 i = s.pairs[p] & 0xffffu, j = s.pairs[p] >> 16;
      u64 wa[8], wb[8];
      u32 ia[G<R>::CI], ib[G<R>::CI];
      get_elem<R>(s, base, i, wa, ia);
      get_elem<R>(s, base, j, wb, ib);
      dst = combine<R>(wa, ia, wb, ib, b, i, j, out, cout, x);
    }
    warp_store<G<R + 1>::STRIDE>(dst, x);
  }
}
#else
// Large buckets (AMD: 4096 of ~8192 elements, the partner re-reads hit its large last-level cache): chains in shared
// memory, elements read from global memory, per-thread stores.
struct Chains {
  u32 head[SLOTS];
  unsigned short nxt[CAP];
};

template <int R>
__device__ __forceinline__ void build_chains(Chains& s, const u32* __restrict__ base, u32 n) {
  for (u32 k = threadIdx.x; k < SLOTS; k += blockDim.x) s.head[k] = 0xffffu;
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    const u32 slot = (base[(size_t)i * G<R>::STRIDE] >> NB_BITS) & (SLOTS - 1);
    s.nxt[i] = (unsigned short)atomicExch(&s.head[slot], i);
  }
  __syncthreads();
}

template <int R>
__device__ __forceinline__ void round_body(const u32* __restrict__ in, const u32* __restrict__ cin,
                                           u32* __restrict__ out, u32* __restrict__ cout) {
  __shared__ Chains s;
  const u32 b = blockIdx.x;
  const u32 n = min(cin[b], (u32)CAP);
  const u32* base = in + (size_t)b * CAP * G<R>::STRIDE;
  build_chains<R>(s, base, n);
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    u32 j = s.nxt[i];
    if (j == 0xffffu) continue;
    u64 wa[8];
    u32 ia[G<R>::CI];
    load_elem<R>(base + (size_t)i * G<R>::STRIDE, wa, ia);
    wa[7] = 0;
    for (; j != 0xffffu; j = s.nxt[j]) {
      u64 wb[8];
      u32 ib[G<R>::CI];
      load_elem<R>(base + (size_t)j * G<R>::STRIDE, wb, ib);
      wb[7] = 0;
      u32 x[G<R + 1>::STRIDE];
      const u64 dst = combine<R>(wa, ia, wb, ib, b, i, j, out, cout, x);
      if (dst)
#pragma unroll
        for (int q = 0; q < G<R + 1>::STRIDE / 4; ++q)
          reinterpret_cast<uint4*>(dst)[q] = make_uint4(x[4 * q], x[4 * q + 1], x[4 * q + 2], x[4 * q + 3]);
    }
  }
}
#endif

extern "C" __global__ __launch_bounds__(TPB) void beam_round1(const u32* in, const u32* cin, u32* out, u32* cout) {
  round_body<1>(in, cin, out, cout);
}
extern "C" __global__ __launch_bounds__(TPB) void beam_round2(const u32* in, const u32* cin, u32* out, u32* cout) {
  round_body<2>(in, cin, out, cout);
}
extern "C" __global__ __launch_bounds__(TPB) void beam_round3(const u32* in, const u32* cin, u32* out, u32* cout) {
  round_body<3>(in, cin, out, cout);
}
extern "C" __global__ __launch_bounds__(TPB) void beam_round4(const u32* in, const u32* cin, u32* out, u32* cout) {
  round_body<4>(in, cin, out, cout);
}

// Round 5: pairs that also agree on key bits 24..47 of the mixed word are solutions (the combination leaves 24 zero
// bits). Writes (position a, position b) in the round-5 buffer, a = the subtree with the smaller first index.
extern "C" __global__ __launch_bounds__(TPB) void beam_final(const u32* __restrict__ in, const u32* __restrict__ cin,
                                                            u32* __restrict__ found, u32 max_found) {
#if LDS_BUCKET
  __shared__ u32 key[CAP], first[CAP], head[SLOTS];
  __shared__ unsigned short nxt[CAP];
  const u32 b = blockIdx.x;
  const u32 n = min(cin[b], (u32)CAP);
  const u32* base = in + (size_t)b * CAP * G<5>::STRIDE;
  for (u32 k = threadIdx.x; k < SLOTS; k += blockDim.x) head[k] = 0xffffu;
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    const u32* e = base + (size_t)i * G<5>::STRIDE;
    const uint2 w01 = *reinterpret_cast<const uint2*>(e + R5_LOW);
    key[i] = w01.x >> 24 | w01.y << 8;  // key bits 24..55 of the mixed word
    first[i] = e[R5_FIRST];             // first leaf index
    nxt[i] = (unsigned short)atomicExch(&head[(w01.x >> NB_BITS) & (SLOTS - 1)], i);
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    const u32 ka = key[i];
    for (u32 j = nxt[i]; j != 0xffffu; j = nxt[j]) {
      if (((ka ^ key[j]) & 0xffffffu) != 0) continue;
      const u32 fa = first[i], fb = first[j];
      const u32 slot = atomicAdd(&found[0], 1u);
      if (slot < max_found) {
        found[1 + 2 * slot] = b * CAP + (fa < fb ? i : j);
        found[2 + 2 * slot] = b * CAP + (fa < fb ? j : i);
      }
    }
  }
#else
  __shared__ Chains s;
  const u32 b = blockIdx.x;
  const u32 n = min(cin[b], (u32)CAP);
  const u32* base = in + (size_t)b * CAP * G<5>::STRIDE;
  build_chains<5>(s, base, n);
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    u32 j = s.nxt[i];
    if (j == 0xffffu) continue;
    const u32* ea = base + (size_t)i * G<5>::STRIDE;
    const u32 ka = ea[R5_LOW] >> 24 | ea[R5_LOW + 1] << 8;  // key bits 24..55 of the mixed word
    for (; j != 0xffffu; j = s.nxt[j]) {
      const u32* eb = base + (size_t)j * G<5>::STRIDE;
      if (((ka ^ (eb[R5_LOW] >> 24 | eb[R5_LOW + 1] << 8)) & 0xffffffu) != 0) continue;
      const u32 fa = ea[R5_FIRST], fb = eb[R5_FIRST];  // first leaf indices
      const u32 slot = atomicAdd(&found[0], 1u);
      if (slot < max_found) {
        found[1 + 2 * slot] = b * CAP + (fa < fb ? i : j);
        found[2 + 2 * slot] = b * CAP + (fa < fb ? j : i);
      }
    }
  }
#endif
}
