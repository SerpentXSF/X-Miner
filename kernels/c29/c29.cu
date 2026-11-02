// Tari Cuckaroo29 edge trimming (compiled at run time; the host prepends NB_BITS, CAPB and TPB).
// Edges are (u, v) pairs kept in NB = 2^NB_BITS buckets by the top bits of the current side's node. A round counts
// the degrees of that side's nodes (two bitmaps: seen once, seen twice), then keeps the edges whose node was seen
// twice, re-bucketed by the other side's node. Bucket-major block order keeps a round's working set of the bitmaps
// (one bucket's node range) in the L2 cache. Edge indices are not kept: a cycle's indices are recovered at the end by
// regenerating every edge.
typedef unsigned int u32;
typedef unsigned long long u64;

#define NB (1 << NB_BITS)
#define NODE_BITS 29
#define NODE_MASK ((1u << NODE_BITS) - 1u)
#define BSHIFT (NODE_BITS - NB_BITS)

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

// Generation: each thread one sipblock (64 edges) into the U buckets. Per-block staging: count per bucket in shared
// memory, one global atomic per bucket reserves the block's slots, then the edges are written.
extern "C" __global__ __launch_bounds__(TPB) void c29_gen(Keys k, uint2* __restrict__ out, u32* __restrict__ counts) {
  __shared__ u32 cnt[NB], base[NB];
  for (u32 i = threadIdx.x; i < NB; i += TPB) cnt[i] = 0;
  __syncthreads();
  const u32 b = blockIdx.x * TPB + threadIdx.x;
  u64 h[64];
  sipblock(k, b, h);
#pragma unroll 8
  for (int i = 0; i < 64; ++i) atomicAdd(&cnt[((u32)h[i] & NODE_MASK) >> BSHIFT], 1u);
  __syncthreads();
  for (u32 i = threadIdx.x; i < NB; i += TPB) {
    base[i] = cnt[i] ? atomicAdd(&counts[i], cnt[i]) : 0u;
    cnt[i] = 0;
  }
  __syncthreads();
#pragma unroll 8
  for (int i = 0; i < 64; ++i) {
    const u32 u = (u32)h[i] & NODE_MASK, v = (u32)(h[i] >> 32) & NODE_MASK;
    const u32 bk = u >> BSHIFT;
    const u32 pos = base[bk] + atomicAdd(&cnt[bk], 1u);
    if (pos < CAPB) out[(size_t)bk * CAPB + pos] = make_uint2(u, v);
  }
}

// Degree count on side `side` (0: u, 1: v). Block x handles chunk (x % CHUNKS) of bucket (x / CHUNKS).
extern "C" __global__ __launch_bounds__(TPB) void c29_count(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ counts, u32* __restrict__ bm1,
                                                            u32* __restrict__ bm2, u32 chunks) {
  const u32 bk = blockIdx.x / chunks, ch = blockIdx.x % chunks;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  for (u32 i = ch * CHUNK + threadIdx.x; i < n && i < (ch + 1) * CHUNK; i += TPB) {
    const uint2 x = e[i];
    const u32 node = side ? x.y : x.x;
    const u32 w = node >> 5, m = 1u << (node & 31);
    if (atomicOr(&bm1[w], m) & m) atomicOr(&bm2[w], m);
  }
}

// Keep the edges whose node on `side` has degree >= 2, into the buckets of the other side. (Counting the other side's
// degrees here as well, to read each round's edges once, is 4x slower: those nodes are random over the whole bitmap,
// while the count kernel works one bucket's node range at a time, which stays in the cache.)
extern "C" __global__ __launch_bounds__(TPB) void c29_filter(u32 side, const uint2* __restrict__ in,
                                                             const u32* __restrict__ counts,
                                                             const u32* __restrict__ bm2, uint2* __restrict__ out,
                                                             u32* __restrict__ out_counts, u32 chunks) {
  __shared__ u32 cnt[NB], base[NB];
  for (u32 i = threadIdx.x; i < NB; i += TPB) cnt[i] = 0;
  __syncthreads();
  const u32 bk = blockIdx.x / chunks, ch = blockIdx.x % chunks;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  const u32 lo = ch * CHUNK, hi = min(n, (ch + 1) * CHUNK);
  // per thread: up to CHUNK / TPB edges, kept in registers between the two passes
  uint2 keep[CHUNK / TPB];
  u32 nk = 0;
#pragma unroll
  for (int r = 0; r < CHUNK / TPB; ++r) {
    const u32 i = lo + r * TPB + threadIdx.x;
    keep[r] = make_uint2(0xffffffffu, 0);
    if (i < hi) {
      const uint2 x = e[i];
      const u32 node = side ? x.y : x.x;
      if (bm2[node >> 5] >> (node & 31) & 1u) {
        keep[r] = x;
        atomicAdd(&cnt[(side ? x.x : x.y) >> BSHIFT], 1u);
        ++nk;
      }
    }
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NB; i += TPB) {
    base[i] = cnt[i] ? atomicAdd(&out_counts[i], cnt[i]) : 0u;
    cnt[i] = 0;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < CHUNK / TPB; ++r) {
    if (keep[r].x == 0xffffffffu) continue;
    const u32 other = side ? keep[r].x : keep[r].y;
    const u32 bo = other >> BSHIFT;
    const u32 pos = base[bo] + atomicAdd(&cnt[bo], 1u);
    if (pos < CAPB) out[(size_t)bo * CAPB + pos] = keep[r];
  }
  (void)nk;
}

// Zero the bitmap words of this round's nodes (instead of clearing the whole bitmaps once few edges are left).
extern "C" __global__ __launch_bounds__(TPB) void c29_clear(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ counts, u32* __restrict__ bm1,
                                                            u32* __restrict__ bm2, u32 chunks) {
  const u32 bk = blockIdx.x / chunks, ch = blockIdx.x % chunks;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  for (u32 i = ch * CHUNK + threadIdx.x; i < n && i < (ch + 1) * CHUNK; i += TPB) {
    const u32 node = side ? e[i].y : e[i].x;
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
