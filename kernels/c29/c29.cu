// Tari Cuckaroo29 edge trimming (compiled at run time; the host prepends NB_BITS, CAPB, TPB, GEN_SB).
// Edges are (u, v) pairs. Early rounds keep them in NB = 2^NB_BITS fine buckets by the top bits of the current side's
// node: one block per bucket counts that side's degrees in shared memory (two bitmaps over the bucket's 2^(29 - NB_BITS)
// nodes: seen once, seen twice) and keeps the edges whose node was seen twice. Scattering straight into 4096 buckets
// does not coalesce, so edges move in two steps: into NC = 64 coarse buckets of the other side, where every producing
// block owns a fixed segment (no reservation pass: the rare edge past a segment's end is dropped), then a split kernel
// spreads each coarse bucket over its 64 fine buckets (written back over the round's input). Once
// few edges are left they go into one flat list and the late rounds use global bitmaps (small, cache-resident) with a
// device-side edge count. Edge indices are not kept: a cycle's indices are recovered at the end by regenerating every
// edge.
typedef unsigned int u32;
typedef unsigned long long u64;

#define NB (1 << NB_BITS)
#define NODE_BITS 29
#define NODE_MASK ((1u << NODE_BITS) - 1u)
#define BSHIFT (NODE_BITS - NB_BITS)
#define LOCAL_WORDS (1 << (BSHIFT - 5))  // bitmap words of one bucket's node range
#define NC_BITS 6
#define NC (1 << NC_BITS)
#define CSHIFT (NODE_BITS - NC_BITS)
#define CCAP ((size_t)CAPB << (NB_BITS - NC_BITS))  // a coarse bucket spans the memory of its fine buckets
#define GEN_BLOCKS ((1u << 23) / (TPB * GEN_SB))
#define GEN_SEG ((u32)(CCAP / GEN_BLOCKS))  // a generating block's segment of a coarse bucket (mean 2^29 / 64 / GEN_BLOCKS)
#define ROUND_SEG ((u32)(CCAP / NB))        // a round block's segment (at most ~1300 used, in round 0)
#define MAX_SEGS 256                        // segments per split block

#ifdef __HIPCC__  // wave32 on RDNA
#define BALLOT(p) ((u32)__ballot(p))
#define SHFL0(x) __shfl((x), 0, 32)
#else
#define BALLOT(p) __ballot_sync(0xffffffffu, (p))
#define SHFL0(x) __shfl_sync(0xffffffffu, (x), 0)
#endif

__device__ __forceinline__ u64 rotl(u64 x, int n) { return (x << n) | (x >> (64 - n)); }

#define SIPROUND                                                                   \
  do {                                                                             \
    v0 += v1; v2 += v3; v1 = rotl(v1, 13); v3 = rotl(v3, 16); v1 ^= v0; v3 ^= v2; \
    v0 = rotl(v0, 32); v2 += v1; v0 += v3; v1 = rotl(v1, 17); v3 = rotl(v3, 21);  \
    v1 ^= v2; v3 ^= v0; v2 = rotl(v2, 32);                                         \
  } while (0)

struct Keys {
  u64 k0, k1, k2, k3;
};

// The 64 block hashes of sipblock b (edge indices 64 b .. 64 b + 63), already xored with the last.
__device__ __forceinline__ void sipblock(const Keys& k, u32 b, u64 h[64]) {
  u64 v0 = k.k0, v1 = k.k1, v2 = k.k2, v3 = k.k3;
  const u64 e0 = (u64)b << 6;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const u64 n = e0 + i;
    v3 ^= n;
    SIPROUND; SIPROUND;
    v0 ^= n;
    v2 ^= 0xff;
    SIPROUND; SIPROUND; SIPROUND; SIPROUND;
    h[i] = (v0 ^ v1) ^ (v2 ^ v3);
  }
  const u64 last = h[63];
#pragma unroll
  for (int i = 0; i < 63; ++i) h[i] ^= last;
}

// Generation into the coarse U buckets: each thread GEN_SB sipblocks, written into the block's segments; segs[c *
// GEN_BLOCKS + block] = the edges in segment c.
extern "C" __global__ __launch_bounds__(TPB) void c29_gen(Keys k, uint2* __restrict__ out, u32* __restrict__ segs) {
  __shared__ u32 cnt[NC];
  for (u32 i = threadIdx.x; i < NC; i += TPB) cnt[i] = 0;
  __syncthreads();
  const u32 b0 = (blockIdx.x * TPB + threadIdx.x) * GEN_SB;
  uint2* o = out + (size_t)blockIdx.x * GEN_SEG;
  u64 h[64];
#pragma unroll 1
  for (int s = 0; s < GEN_SB; ++s) {
    sipblock(k, b0 + s, h);
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const u32 u = (u32)h[i] & NODE_MASK, v = (u32)(h[i] >> 32) & NODE_MASK;
      const u32 c = u >> CSHIFT;
      const u32 pos = atomicAdd(&cnt[c], 1u);
      if (pos < GEN_SEG) o[c * CCAP + pos] = make_uint2(u, v);
    }
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NC; i += TPB) segs[i * GEN_BLOCKS + blockIdx.x] = min(cnt[i], GEN_SEG);
}

// Coarse -> fine buckets of side `side`. A coarse bucket is `nseg` segments of `seg_cap` edges (segs[c * nseg + s]
// used); `parts` blocks per coarse bucket (both powers of two), each max(1, nseg / parts) (<= MAX_SEGS) segments, or
// a slice of one segment when parts > nseg: count per fine bucket, reserve with one global atomic per fine bucket,
// write.
extern "C" __global__ __launch_bounds__(TPB) void c29_split(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ segs, u32 nseg, u32 seg_cap,
                                                            uint2* __restrict__ out, u32* __restrict__ counts,
                                                            u32 parts) {
  __shared__ u32 cnt[NB / NC], pre[MAX_SEGS + 1];
  const u32 c = blockIdx.x / parts, part = blockIdx.x % parts;
  const u32 per = max(nseg / parts, 1u), sub = max(parts / nseg, 1u), s0 = part / sub * per;
  for (u32 i = threadIdx.x; i < NB / NC; i += TPB) cnt[i] = 0;
  for (u32 i = threadIdx.x; i < per; i += TPB) pre[i + 1] = segs[c * nseg + s0 + i];
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 t = 0;
    pre[0] = 0;
    for (u32 i = 1; i <= per; ++i) pre[i] = t += pre[i];
  }
  __syncthreads();
  const u32 total = pre[per], slice = (total + sub - 1) / sub, beg = part % sub * slice, n = min(total, beg + slice);
  const uint2* e = in + c * CCAP + (size_t)s0 * seg_cap;
  const u32 smask = NB / NC - 1;
  // flat index i -> (segment, offset) by binary search in the prefix sums
#define SPLIT_EDGE(i, x)                                     \
  {                                                          \
    u32 lo = 0, hi = per;                                    \
    while (hi - lo > 1) {                                    \
      const u32 mid = (lo + hi) >> 1;                        \
      if (pre[mid] <= (i)) lo = mid; else hi = mid;          \
    }                                                        \
    x = e[(size_t)lo * seg_cap + ((i) - pre[lo])];           \
  }
  for (u32 i = beg + threadIdx.x; i < n; i += TPB) {
    uint2 x;
    SPLIT_EDGE(i, x);
    atomicAdd(&cnt[((side ? x.y : x.x) >> BSHIFT) & smask], 1u);
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NB / NC; i += TPB) {
    const u32 m = cnt[i];
    cnt[i] = m ? atomicAdd(&counts[c * (NB / NC) + i], m) : 0u;
  }
  __syncthreads();
  for (u32 i = beg + threadIdx.x; i < n; i += TPB) {
    uint2 x;
    SPLIT_EDGE(i, x);
    const u32 f = (side ? x.y : x.x) >> BSHIFT;
    const u32 pos = atomicAdd(&cnt[f & smask], 1u);
    if (pos < CAPB) out[(size_t)f * CAPB + pos] = x;
  }
#undef SPLIT_EDGE
}

// One trimming round on side `side` (0: u, 1: v), one block per fine bucket of that side; the survivors go to the
// block's segments of the other side's coarse buckets (segs[c * NB + bucket]).
extern "C" __global__ __launch_bounds__(TPB) void c29_round(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ counts, uint2* __restrict__ out,
                                                            u32* __restrict__ segs) {
  __shared__ u32 s1[LOCAL_WORDS], s2[LOCAL_WORDS], cnt[NC];
  for (u32 i = threadIdx.x; i < LOCAL_WORDS; i += TPB) { s1[i] = 0; s2[i] = 0; }
  for (u32 i = threadIdx.x; i < NC; i += TPB) cnt[i] = 0;
  __syncthreads();
  const u32 bk = blockIdx.x;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  const u32 lmask = (1u << BSHIFT) - 1u;
  for (u32 i = threadIdx.x; i < n; i += TPB) {
    const uint2 x = e[i];
    const u32 l = (side ? x.y : x.x) & lmask;
    const u32 m = 1u << (l & 31);
    if (atomicOr(&s1[l >> 5], m) & m) atomicOr(&s2[l >> 5], m);
  }
  __syncthreads();
  uint2* o = out + (size_t)bk * ROUND_SEG;
  for (u32 i = threadIdx.x; i < n; i += TPB) {
    const uint2 x = e[i];
    const u32 l = (side ? x.y : x.x) & lmask;
    if (!(s2[l >> 5] >> (l & 31) & 1u)) continue;
    const u32 co = (side ? x.x : x.y) >> CSHIFT;
    const u32 pos = atomicAdd(&cnt[co], 1u);
    if (pos < ROUND_SEG) o[co * CCAP + pos] = x;
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NC; i += TPB) segs[i * NB + bk] = min(cnt[i], ROUND_SEG);
}

// The same round for fewer edges: the survivors go to whole coarse buckets (one segment each, ccounts[c]), slots
// reserved with one global atomic per bucket (a pass more over the input, but no per-block segments to split).
extern "C" __global__ __launch_bounds__(TPB) void c29_round_r(u32 side, const uint2* __restrict__ in,
                                                              const u32* __restrict__ counts, uint2* __restrict__ out,
                                                              u32* __restrict__ ccounts) {
  __shared__ u32 s1[LOCAL_WORDS], s2[LOCAL_WORDS], cnt[NC];
  for (u32 i = threadIdx.x; i < LOCAL_WORDS; i += TPB) { s1[i] = 0; s2[i] = 0; }
  for (u32 i = threadIdx.x; i < NC; i += TPB) cnt[i] = 0;
  __syncthreads();
  const u32 bk = blockIdx.x;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  const u32 lmask = (1u << BSHIFT) - 1u;
  for (u32 i = threadIdx.x; i < n; i += TPB) {
    const uint2 x = e[i];
    const u32 l = (side ? x.y : x.x) & lmask;
    const u32 m = 1u << (l & 31);
    if (atomicOr(&s1[l >> 5], m) & m) atomicOr(&s2[l >> 5], m);
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += TPB) {
    const uint2 x = e[i];
    const u32 l = (side ? x.y : x.x) & lmask;
    if (s2[l >> 5] >> (l & 31) & 1u) atomicAdd(&cnt[(side ? x.x : x.y) >> CSHIFT], 1u);
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NC; i += TPB) {
    const u32 c = cnt[i];
    cnt[i] = c ? atomicAdd(&ccounts[i], c) : 0u;
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < n; i += TPB) {
    const uint2 x = e[i];
    const u32 l = (side ? x.y : x.x) & lmask;
    if (!(s2[l >> 5] >> (l & 31) & 1u)) continue;
    const u32 co = (side ? x.x : x.y) >> CSHIFT;
    const u32 pos = atomicAdd(&cnt[co], 1u);
    if (pos < CCAP) out[co * CCAP + pos] = x;
  }
}

// Buckets -> one flat list at the given offsets (one block per bucket).
extern "C" __global__ __launch_bounds__(TPB) void c29_flatten(const uint2* __restrict__ in,
                                                              const u32* __restrict__ counts,
                                                              const u32* __restrict__ offsets,
                                                              uint2* __restrict__ flat) {
  const u32 bk = blockIdx.x, n = min(counts[bk], (u32)CAPB), o = offsets[bk];
  const uint2* e = in + (size_t)bk * CAPB;
  for (u32 i = threadIdx.x; i < n; i += TPB) flat[o + i] = e[i];
}

// Late rounds on the flat list: count (global bitmaps), keep (warp-aggregated append), clear the bitmap words used.
extern "C" __global__ __launch_bounds__(TPB) void c29_fcount(u32 side, const uint2* __restrict__ in,
                                                             const u32* __restrict__ nin, u32* __restrict__ bm1,
                                                             u32* __restrict__ bm2) {
  const u32 n = *nin;
  for (u32 i = blockIdx.x * TPB + threadIdx.x; i < n; i += gridDim.x * TPB) {
    const u32 node = side ? in[i].y : in[i].x;
    const u32 w = node >> 5, m = 1u << (node & 31);
    if (atomicOr(&bm1[w], m) & m) atomicOr(&bm2[w], m);
  }
}

extern "C" __global__ __launch_bounds__(TPB) void c29_ffilter(u32 side, const uint2* __restrict__ in,
                                                              const u32* __restrict__ nin,
                                                              const u32* __restrict__ bm2, uint2* __restrict__ out,
                                                              u32* __restrict__ nout) {
  const u32 n = *nin;
  const u32 lane = threadIdx.x & 31;
  for (u32 base = blockIdx.x * TPB; base < n; base += gridDim.x * TPB) {  // the whole block loops together
    const u32 i = base + threadIdx.x;
    uint2 x = make_uint2(0, 0);
    bool keep = false;
    if (i < n) {
      x = in[i];
      const u32 node = side ? x.y : x.x;
      keep = bm2[node >> 5] >> (node & 31) & 1u;
    }
    const u32 mask = BALLOT(keep);
    u32 slot = 0;
    if (lane == 0 && mask) slot = atomicAdd(nout, (u32)__popc(mask));
    slot = SHFL0(slot);
    if (keep) out[slot + __popc(mask & ((1u << lane) - 1u))] = x;
  }
}

extern "C" __global__ __launch_bounds__(TPB) void c29_fclear(u32 side, const uint2* __restrict__ in,
                                                             const u32* __restrict__ nin, u32* __restrict__ bm1,
                                                             u32* __restrict__ bm2) {
  const u32 n = *nin;
  for (u32 i = blockIdx.x * TPB + threadIdx.x; i < n; i += gridDim.x * TPB) {
    const u32 node = side ? in[i].y : in[i].x;
    bm1[node >> 5] = 0;
    bm2[node >> 5] = 0;
  }
}

// Edge index recovery: every edge is regenerated and compared with the cycle's (u, v) pairs (a 1024-bit filter on u
// first).
extern "C" __global__ __launch_bounds__(TPB) void c29_recover(Keys k, const uint2* __restrict__ want, u32 nwant,
                                                              u32* __restrict__ found) {
  __shared__ uint2 w[64];
  __shared__ u32 filt[32];
  if (threadIdx.x < 32) filt[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x < nwant) {
    w[threadIdx.x] = want[threadIdx.x];
    atomicOr(&filt[(w[threadIdx.x].x >> 5) & 31], 1u << (w[threadIdx.x].x & 31));
  }
  __syncthreads();
  const u32 b = blockIdx.x * TPB + threadIdx.x;
  u64 h[64];
  sipblock(k, b, h);
  for (int i = 0; i < 64; ++i) {
    const u32 u = (u32)h[i] & NODE_MASK;
    if (!(filt[(u >> 5) & 31] >> (u & 31) & 1u)) continue;
    const u32 v = (u32)(h[i] >> 32) & NODE_MASK;
    for (u32 j = 0; j < nwant; ++j)
      if (w[j].x == u && w[j].y == v) found[j] = (b << 6) + i;
  }
}
