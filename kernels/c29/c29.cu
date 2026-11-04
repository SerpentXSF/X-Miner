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
#ifndef GEN_TPB
#define GEN_TPB TPB
#endif
#define GEN_BLOCKS ((1u << 23) / (TPB * GEN_SB))
#define GEN_PER (TPB * GEN_SB / GEN_TPB)  // sipblocks per generating thread (GEN_TPB threads per block)
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

// Staged scatter into 64 buckets: put() parks an edge in its bucket's row of shared memory (or, when the row is full,
// writes it straight out); flush() writes every row out as one contiguous run at the bucket's next position
// (cnt[c], shared: block-local or pre-reserved) and empties the rows. Destination of bucket c: out + c * stride,
// positions below cap. One scattered 8-byte store per edge costs several times the bandwidth of the runs.
template <int SLOTS>
__device__ __forceinline__ void stage_put(uint2 (*stage)[SLOTS], u32* fill, u32* cnt, uint2* out, size_t stride,
                                          u32 cap, u32 c, uint2 x) {
  const u32 slot = atomicAdd(&fill[c], 1u);
  if (slot < SLOTS) {
    stage[c][slot] = x;
  } else {
    const u32 pos = atomicAdd(&cnt[c], 1u);
    if (pos < cap) out[c * stride + pos] = x;
  }
}
template <int SLOTS, int TB>
__device__ __forceinline__ void stage_flush(uint2 (*stage)[SLOTS], u32* fill, u32* cnt, uint2* out, size_t stride,
                                            u32 cap) {
  __syncthreads();
  const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (u32 c = warp; c < 64; c += TB / 32) {
    const u32 n = min(fill[c], (u32)SLOTS);
    u32 base = 0;
    if (lane == 0) base = atomicAdd(&cnt[c], n);
    base = SHFL0(base);
    for (u32 j = lane; j < n; j += 32)
      if (base + j < cap) out[c * stride + base + j] = stage[c][j];
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < 64; i += TB) fill[i] = 0;
  __syncthreads();
}
// Staged survivor writes in c29_round (ROUND_STAGE: compiled in, NVIDIA; the host enables them per round, round 0
// by default: RTX 3060 3.01 -> 3.06 g/s; all four segment rounds 3.02).
#ifndef ROUND_STAGE
#define ROUND_STAGE 1
#endif
#ifndef GEN_DIRECT
#define GEN_DIRECT 0
#endif
#ifndef SPLIT_DIRECT
#define SPLIT_DIRECT 0
#endif
#if SPLIT_DIRECT
#undef SPLIT_ONEPASS
#define SPLIT_ONEPASS 0  // direct stores need the count pass's reservation
#endif
#ifndef SPLIT_ONEPASS
#define SPLIT_ONEPASS 1
#endif
// The rounds' staging: edges per thread between flushes, staged edges per coarse bucket. With the nibble counters
// (dynamic shared memory) 4 and 48 (RTX 3060 3.305 -> 3.32 g/s); next to the 32 KB of bitmaps only 1 and 24 fit
// under 48 KB of static shared memory.
#if defined(ROUND_NIBBLE) && ROUND_NIBBLE
#define ROUND_PER_DEF 4
#define RSLOTS_DEF 48
#else
#define ROUND_PER_DEF 1
#define RSLOTS_DEF 24
#endif
#ifndef ROUND_PER
#define ROUND_PER ROUND_PER_DEF
#endif
#ifndef RSLOTS
#define RSLOTS RSLOTS_DEF
#endif

// Generation into the coarse U buckets: GEN_BLOCKS blocks of GEN_TPB threads, GEN_PER sipblocks per thread, written
// into the block's segments; segs[c * GEN_BLOCKS + block] = the edges in segment c.
// A sipblock's edges are its 64 hashes xored with the last one. Holding the 64 hashes needs 128 registers and a fully
// unrolled block (~11k instructions: the warps then wait on instruction fetch), so each sipblock is hashed twice in a
// rolled loop: once for the last hash, once to emit the edges. Edges are staged per coarse bucket in shared memory
// (GEN_WAVE edges per thread at a time) and each bucket's run is written out contiguously: one scattered 8-byte store
// per edge cost ~100 of ~150 ms on an RTX 3060. Moving the hash's 64-bit adds and rotations to the multiply pipe
// (IMAD.WIDE.U32 by 1 / by 2^n, multipliers from shared memory) was slower: 116 vs 102 ms.
// Edges per thread and wave, and staged edges per bucket (mean GEN_TPB * GEN_WAVE / NC; a full row writes directly).
// Longer runs per wave: RTX 3060 (512 threads) waves of 4 / 64 slots 3.33 g/s, 8/88 3.35, 16/92 3.36, 32/92 3.33.
#ifndef GEN_WAVE
#define GEN_WAVE (8192 / GEN_TPB)  // a mean of 128 edges per bucket and wave
#endif
#ifndef GEN_SLOTS
#define GEN_SLOTS 92  // (46 KB of the 48 KB of static shared memory)
#endif
__device__ __forceinline__ u64 siphash_next(u64& v0, u64& v1, u64& v2, u64& v3, u64 n) {
  v3 ^= n;
  SIPROUND; SIPROUND;
  v0 ^= n;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1) ^ (v2 ^ v3);
}
#if GEN_DIRECT
// AMD: one pass, the 64 hashes in registers, one store per edge straight into the segment (the Infinity Cache merges
// the scattered stores; staging was slower there: 9070 XT generation + split 34 -> 54 ms).
extern "C" __global__ __launch_bounds__(GEN_TPB) void c29_gen(Keys k, uint2* __restrict__ out, u32* __restrict__ segs) {
  __shared__ u32 cnt[NC];
  for (u32 i = threadIdx.x; i < NC; i += GEN_TPB) cnt[i] = 0;
  __syncthreads();
  const u32 b0 = (blockIdx.x * GEN_TPB + threadIdx.x) * GEN_PER;
  uint2* o = out + (size_t)blockIdx.x * GEN_SEG;
  u64 h[64];
#pragma unroll 1
  for (int s = 0; s < GEN_PER; ++s) {
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
  for (u32 i = threadIdx.x; i < NC; i += GEN_TPB) segs[i * GEN_BLOCKS + blockIdx.x] = min(cnt[i], GEN_SEG);
}
#else
extern "C" __global__ __launch_bounds__(GEN_TPB) void c29_gen(Keys k, uint2* __restrict__ out, u32* __restrict__ segs) {
  __shared__ u32 cnt[NC], fill[NC];
  __shared__ uint2 stage[NC][GEN_SLOTS];
  for (u32 i = threadIdx.x; i < NC; i += GEN_TPB) cnt[i] = fill[i] = 0;
  __syncthreads();
  const u32 b0 = (blockIdx.x * GEN_TPB + threadIdx.x) * GEN_PER;
  uint2* o = out + (size_t)blockIdx.x * GEN_SEG;
  const u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int s = 0; s < GEN_PER; ++s) {
    const u64 e0 = (u64)(b0 + s) << 6;
    u64 last;
    {
      u64 v0 = k.k0, v1 = k.k1, v2 = k.k2, v3 = k.k3;
#pragma unroll 1
      for (int i = 0; i < 64; ++i) last = siphash_next(v0, v1, v2, v3, e0 + i);
    }
    u64 v0 = k.k0, v1 = k.k1, v2 = k.k2, v3 = k.k3;
#pragma unroll 1
    for (int w = 0; w < 64; w += GEN_WAVE) {
#pragma unroll 1
      for (int i = w; i < w + GEN_WAVE; ++i) {
        u64 h = siphash_next(v0, v1, v2, v3, e0 + i);
        if (i < 63) h ^= last;
        const u32 u = (u32)h & NODE_MASK, v = (u32)(h >> 32) & NODE_MASK;
        const u32 c = u >> CSHIFT;
        const u32 slot = atomicAdd(&fill[c], 1u);
        if (slot < GEN_SLOTS) {
          stage[c][slot] = make_uint2(u, v);
        } else {  // rare: a bucket's staging row is full; write the edge directly
          const u32 pos = atomicAdd(&cnt[c], 1u);
          if (pos < GEN_SEG) o[c * CCAP + pos] = make_uint2(u, v);
        }
      }
      __syncthreads();
      // each warp writes out whole buckets: their runs are contiguous in the block's segments
      for (u32 c = warp; c < NC; c += GEN_TPB / 32) {
        const u32 n = min(fill[c], (u32)GEN_SLOTS);
        u32 base = 0;
        if (lane == 0) base = atomicAdd(&cnt[c], n);
        base = SHFL0(base);
        for (u32 j = lane; j < n; j += 32)
          if (base + j < GEN_SEG) o[c * CCAP + base + j] = stage[c][j];
      }
      __syncthreads();
      for (u32 i = threadIdx.x; i < NC; i += GEN_TPB) fill[i] = 0;
      __syncthreads();
    }
  }
  for (u32 i = threadIdx.x; i < NC; i += GEN_TPB) segs[i * GEN_BLOCKS + blockIdx.x] = min(cnt[i], GEN_SEG);
}
#endif

// Coarse -> fine buckets of side `side`. A coarse bucket is `nseg` segments of `seg_cap` edges (segs[c * nseg + s]
// used); `parts` blocks per coarse bucket (both powers of two), each max(1, nseg / parts) (<= MAX_SEGS) segments, or
// a slice of one segment when parts > nseg: count per fine bucket, reserve with one global atomic per fine bucket,
// write.
#ifndef SPLIT_TPB
#define SPLIT_TPB TPB
#endif
// Edges per thread between flushes, and the staged edges per fine bucket: longer runs per flush (RTX 3060: 1 edge
// and 24 slots 3.06, 2/48 3.27, 3/64 3.31, 4/80 3.31 g/s; generation's split 36 -> 28 ms, round 0's 25 -> 19 ms).
#ifndef SPLIT_PER
#define SPLIT_PER 3
#endif
#ifndef SSLOTS
#define SSLOTS 64
#endif
extern "C" __global__ __launch_bounds__(SPLIT_TPB) void c29_split(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ segs, u32 nseg, u32 seg_cap,
                                                            uint2* __restrict__ out, u32* __restrict__ counts,
                                                            u32 parts) {
  __shared__ u32 cnt[NB / NC], pre[MAX_SEGS + 1], fill[NB / NC];
#if !SPLIT_DIRECT
  __shared__ uint2 stage[NB / NC][SSLOTS];
#endif
  const u32 c = blockIdx.x / parts, part = blockIdx.x % parts;
  const u32 per = max(nseg / parts, 1u), sub = max(parts / nseg, 1u), s0 = part / sub * per;
  for (u32 i = threadIdx.x; i < NB / NC; i += SPLIT_TPB) cnt[i] = fill[i] = 0;
  for (u32 i = threadIdx.x; i < per; i += SPLIT_TPB) pre[i + 1] = segs[c * nseg + s0 + i];
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
#if SPLIT_ONEPASS
  // One pass: each staged run reserves its slots in the fine bucket's global count when it is written out (one
  // global atomic per run of ~SSLOTS edges), so the input is read once instead of twice (count, then write).
  u32* gc = counts + c * (NB / NC);
#else
  for (u32 i = beg + threadIdx.x; i < n; i += SPLIT_TPB) {
    uint2 x;
    SPLIT_EDGE(i, x);
    atomicAdd(&cnt[((side ? x.y : x.x) >> BSHIFT) & smask], 1u);
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < NB / NC; i += SPLIT_TPB) {
    const u32 m = cnt[i];
    cnt[i] = m ? atomicAdd(&counts[c * (NB / NC) + i], m) : 0u;
  }
  __syncthreads();
  u32* gc = cnt;
#endif
  uint2* ob = out + (size_t)c * (NB / NC) * CAPB;  // this coarse bucket's fine buckets
#if SPLIT_DIRECT
  // AMD: direct stores after the count pass (see GEN_DIRECT)
  for (u32 i = beg + threadIdx.x; i < n; i += SPLIT_TPB) {
    uint2 x;
    SPLIT_EDGE(i, x);
    const u32 f = ((side ? x.y : x.x) >> BSHIFT) & smask;
    const u32 pos = atomicAdd(&gc[f], 1u);
    if (pos < CAPB) ob[(size_t)f * CAPB + pos] = x;
  }
#else
  // Large segments (generation's ~8.5K edges): segment by segment, no per-edge search for the segment of a flat
  // index; the block's flat range [beg, n) intersected with each segment's [pre[s], pre[s + 1]). Small segments (the
  // rounds' ~100) would leave most of a pass idle: flat index and binary search.
  if (n - beg >= 2u * SPLIT_TPB * per) {
    for (u32 sg = 0; sg < per; ++sg) {
      const u32 lo = max(beg, pre[sg]), hi = min(n, pre[sg + 1]);
      if (lo >= hi) continue;  // (uniform: every thread sees the same bounds)
      const uint2* es = e + (size_t)sg * seg_cap - pre[sg];
      for (u32 i0 = lo; i0 < hi; i0 += SPLIT_TPB * SPLIT_PER) {
#pragma unroll
        for (u32 k = 0; k < SPLIT_PER; ++k) {
          const u32 i = i0 + k * SPLIT_TPB + threadIdx.x;
          if (i < hi) {
            const uint2 x = es[i];
            stage_put<SSLOTS>(stage, fill, gc, ob, CAPB, CAPB, ((side ? x.y : x.x) >> BSHIFT) & smask, x);
          }
        }
        stage_flush<SSLOTS, SPLIT_TPB>(stage, fill, gc, ob, CAPB, CAPB);
      }
    }
  } else {
    for (u32 i0 = beg; i0 < n; i0 += SPLIT_TPB * SPLIT_PER) {
#pragma unroll
      for (u32 k = 0; k < SPLIT_PER; ++k) {
        const u32 i = i0 + k * SPLIT_TPB + threadIdx.x;
        if (i < n) {
          uint2 x;
          SPLIT_EDGE(i, x);
          stage_put<SSLOTS>(stage, fill, gc, ob, CAPB, CAPB, ((side ? x.y : x.x) >> BSHIFT) & smask, x);
        }
      }
      stage_flush<SSLOTS, SPLIT_TPB>(stage, fill, gc, ob, CAPB, CAPB);
    }
  }
#endif
#undef SPLIT_EDGE
}

// Visits the n edges at e (16-byte aligned) as BODY(x) for each uint2 x: 16-byte loads, two in flight per thread.
// (Same speed as one 8-byte load per iteration on an RTX 3060; the rounds wait on the bitmap atomics' results:
// short-scoreboard and MIO stalls. Warp-aggregated atomics on the 64 bucket counters (match.any) were slower.)
// FOR_EDGES_REV: the same, last edges first. A round reads its bucket twice (count, then filter); read back to front
// the second time, its first reads are the ones the cache most likely still holds.
#ifndef REV_PASS
#define REV_PASS 1
#endif
#define FOR_EDGES(e, n, BODY) FOR_EDGES_D(e, n, TPB, 0, BODY)
#define FOR_EDGES_TB(e, n, TB, BODY) FOR_EDGES_D(e, n, TB, 0, BODY)
#define FOR_EDGES_REV(e, n, TB, BODY) FOR_EDGES_D(e, n, TB, REV_PASS, BODY)
#define FOR_EDGES_D(e, n, TB, REV, BODY)                                              \
  {                                                                                   \
    const uint4* e4_ = reinterpret_cast<const uint4*>(e);                             \
    const u32 n2_ = (n) >> 1;                                                         \
    for (u32 j0_ = threadIdx.x; j0_ < n2_; j0_ += 2 * (TB)) {                         \
      const u32 j_ = (REV) ? n2_ - 1 - j0_ : j0_;                                     \
      const uint4 a_ = e4_[j_];                                                       \
      const bool hb_ = j0_ + (TB) < n2_;                                              \
      const uint4 b_ = hb_ ? e4_[(REV) ? j_ - (TB) : j_ + (TB)] : make_uint4(0, 0, 0, 0); \
      { const uint2 x = make_uint2(a_.x, a_.y); BODY }                                \
      { const uint2 x = make_uint2(a_.z, a_.w); BODY }                                \
      if (hb_) {                                                                      \
        { const uint2 x = make_uint2(b_.x, b_.y); BODY }                              \
        { const uint2 x = make_uint2(b_.z, b_.w); BODY }                              \
      }                                                                               \
    }                                                                                 \
    if (((n) & 1) && threadIdx.x == 0) { const uint2 x = (e)[(n) - 1]; BODY }         \
  }

// One trimming round on side `side` (0: u, 1: v), one block per fine bucket of that side; the survivors go to the
// block's segments of the other side's coarse buckets (segs[c * NB + bucket]).
// Degree counts of a bucket's 2^BSHIFT nodes. ROUND_NIBBLE (NVIDIA; 64 KB of dynamic shared memory): a 4-bit counter
// per node, one atomic add per edge whose result nobody waits for (a count wraps only at 16, never seen in practice).
// Otherwise two bitmaps (seen once, seen twice): the second atomic depends on the first's result, and the rounds
// waited on those results (short-scoreboard and MIO stalls).
#ifndef ROUND_NIBBLE
#define ROUND_NIBBLE 0
#endif
#if ROUND_NIBBLE
#define DEG_DECL extern __shared__ u32 nib[]; /* LOCAL_WORDS * 4 words */
#define DEG_CLEAR for (u32 i = threadIdx.x; i < 4 * LOCAL_WORDS; i += TPB) nib[i] = 0;
#define DEG_COUNT(l) atomicAdd(&nib[(l) >> 3], 1u << (((l) & 7u) << 2));
#define DEG_KEEP(l) ((nib[(l) >> 3] >> (((l) & 7u) << 2) & 15u) >= 2u)
#else
#define DEG_DECL __shared__ u32 s1[LOCAL_WORDS], s2[LOCAL_WORDS];
#define DEG_CLEAR for (u32 i = threadIdx.x; i < LOCAL_WORDS; i += TPB) { s1[i] = 0; s2[i] = 0; }
#define DEG_COUNT(l) { const u32 m_ = 1u << ((l) & 31); if (atomicOr(&s1[(l) >> 5], m_) & m_) atomicOr(&s2[(l) >> 5], m_); }
#define DEG_KEEP(l) (s2[(l) >> 5] >> ((l) & 31) & 1u)
#endif
// One trimming round on side `side` (0: u, 1: v), one block per fine bucket of that side; the survivors go to the
// block's segments of the other side's coarse buckets (segs[c * NB + bucket]). stage != 0 (the host's choice per
// round): the survivors are staged in shared memory and written as runs (RTX 3060 round 0: 41.0 -> 36.5 ms; with
// fewer edges per bucket the barriers cost more than the scattered stores).
extern "C" __global__ __launch_bounds__(TPB) void c29_round(u32 side, const uint2* __restrict__ in,
                                                            const u32* __restrict__ counts, uint2* __restrict__ out,
                                                            u32* __restrict__ segs, u32 stage_on) {
  DEG_DECL
  __shared__ u32 cnt[NC], fill[NC];
#if ROUND_STAGE
  __shared__ uint2 stage[NC][RSLOTS];
#endif
  DEG_CLEAR
  for (u32 i = threadIdx.x; i < NC; i += TPB) cnt[i] = fill[i] = 0;
  __syncthreads();
  const u32 bk = blockIdx.x;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  const u32 lmask = (1u << BSHIFT) - 1u;
  FOR_EDGES(e, n, {
    const u32 l = (side ? x.y : x.x) & lmask;
    DEG_COUNT(l)
  })
  __syncthreads();
  uint2* o = out + (size_t)bk * ROUND_SEG;
#if ROUND_STAGE
  if (stage_on) {
    for (u32 i0 = 0; i0 < n; i0 += TPB * ROUND_PER) {
#pragma unroll
      for (u32 k = 0; k < ROUND_PER; ++k) {
        const u32 i = i0 + k * TPB + threadIdx.x;
        if (i < n) {
          const uint2 x = e[REV_PASS ? n - 1 - i : i];
          const u32 l = (side ? x.y : x.x) & lmask;
          if (DEG_KEEP(l))
            stage_put<RSLOTS>(stage, fill, cnt, o, CCAP, ROUND_SEG, (side ? x.x : x.y) >> CSHIFT, x);
        }
      }
      stage_flush<RSLOTS, TPB>(stage, fill, cnt, o, CCAP, ROUND_SEG);
    }
  } else
#endif
  {
    FOR_EDGES_REV(e, n, TPB, {
      const u32 l = (side ? x.y : x.x) & lmask;
      if (DEG_KEEP(l)) {
        const u32 co = (side ? x.x : x.y) >> CSHIFT;
        const u32 pos = atomicAdd(&cnt[co], 1u);
        if (pos < ROUND_SEG) o[co * CCAP + pos] = x;
      }
    })
    __syncthreads();
  }
  for (u32 i = threadIdx.x; i < NC; i += TPB) segs[i * NB + bk] = min(cnt[i], ROUND_SEG);
}

// The same round for fewer edges: the survivors go to whole coarse buckets (one segment each, ccounts[c]), slots
// reserved with one global atomic per bucket (a pass more over the input, but no per-block segments to split).
#ifndef RS_TPB
#define RS_TPB 256
#endif
template <int RTPB>
__device__ __forceinline__ void round_r(u32 side, const uint2* __restrict__ in,
                                        const u32* __restrict__ counts, uint2* __restrict__ out,
                                        u32* __restrict__ ccounts) {
  __shared__ u32 s1[LOCAL_WORDS], s2[LOCAL_WORDS], cnt[NC], fill[NC];
  for (u32 i = threadIdx.x; i < LOCAL_WORDS; i += RTPB) { s1[i] = 0; s2[i] = 0; }
  for (u32 i = threadIdx.x; i < NC; i += RTPB) cnt[i] = fill[i] = 0;
  __syncthreads();
  const u32 bk = blockIdx.x;
  const u32 n = min(counts[bk], (u32)CAPB);
  const uint2* e = in + (size_t)bk * CAPB;
  const u32 lmask = (1u << BSHIFT) - 1u;
  FOR_EDGES_TB(e, n, RTPB, {
    const u32 l = (side ? x.y : x.x) & lmask;
    const u32 m = 1u << (l & 31);
    if (atomicOr(&s1[l >> 5], m) & m) atomicOr(&s2[l >> 5], m);
  })
  __syncthreads();
  FOR_EDGES_REV(e, n, RTPB, {
    const u32 l = (side ? x.y : x.x) & lmask;
    if (s2[l >> 5] >> (l & 31) & 1u) atomicAdd(&cnt[(side ? x.x : x.y) >> CSHIFT], 1u);
  })
  __syncthreads();
  for (u32 i = threadIdx.x; i < NC; i += RTPB) {
    const u32 c = cnt[i];
    cnt[i] = c ? atomicAdd(&ccounts[i], c) : 0u;
  }
  __syncthreads();
  FOR_EDGES_TB(e, n, RTPB, {
    const u32 l = (side ? x.y : x.x) & lmask;
    if (s2[l >> 5] >> (l & 31) & 1u) {
      const u32 co = (side ? x.x : x.y) >> CSHIFT;
      const u32 pos = atomicAdd(&cnt[co], 1u);
      if (pos < CCAP) out[co * CCAP + pos] = x;
    }
  })

}
extern "C" __global__ __launch_bounds__(TPB) void c29_round_r(u32 side, const uint2* __restrict__ in,
                                                              const u32* __restrict__ counts, uint2* __restrict__ out,
                                                              u32* __restrict__ ccounts) {
  round_r<TPB>(side, in, counts, out, ccounts);
}
// Rounds below 2^24 edges (a few thousand edges per bucket and fewer): smaller blocks, several resident per SM (the
// 32 KB of bitmaps, not the threads, limited residency). RTX 3060: 4.08 -> 4.20 g/s; at ~1M edges 0.45 -> 0.22 ms per
// round kernel, at ~20M slower (the 2^24 cutoff measured best: 2^22 4.17, 2^25 4.16). RX 9070 XT unchanged.
extern "C" __global__ __launch_bounds__(RS_TPB) void c29_round_rs(u32 side, const uint2* __restrict__ in,
                                                                 const u32* __restrict__ counts,
                                                                 uint2* __restrict__ out, u32* __restrict__ ccounts) {
  round_r<RS_TPB>(side, in, counts, out, ccounts);
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
