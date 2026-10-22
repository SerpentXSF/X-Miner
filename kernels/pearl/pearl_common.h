// Pearl device code shared by the AMD (HIP) and NVIDIA (CUDA) builds: BLAKE3, noise generation, noising into
// the packed GEMM layout, and the per-tile jackpot check. Bit-exact with src/algo/pearl/pearl.cpp, which is
// checked against Pearl's official code; --selftest compares every kernel here with that CPU reference.
#pragma once

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

#define XM_DEV __device__ __forceinline__

struct Key8 {
  uint32_t w[8];
};
struct U256w {
  uint32_t w[8];  // little-endian limbs
};

// ------------------------------------------------------------------------------------------------ BLAKE3
__constant__ static const uint32_t kB3IV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                                               0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
enum : uint32_t { B3_CHUNK_START = 1, B3_CHUNK_END = 2, B3_PARENT = 4, B3_ROOT = 8, B3_KEYED = 16 };

XM_DEV uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

XM_DEV void b3_g(uint32_t* v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx;
  v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my;
  v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 7);
}

// Compress one 64-byte block (16 LE words). Writes the 8-word chaining value into out.
XM_DEV void b3_compress(const uint32_t cv[8], const uint32_t m_in[16], uint64_t counter, uint32_t len,
                        uint32_t flags, uint32_t out[8]) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = m_in[i];
  uint32_t v[16] = {cv[0],   cv[1],   cv[2],   cv[3],   cv[4],   cv[5],   cv[6],   cv[7],
                    kB3IV[0], kB3IV[1], kB3IV[2], kB3IV[3], (uint32_t)counter, (uint32_t)(counter >> 32), len, flags};
  const int perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    b3_g(v, 0, 4, 8, 12, m[0], m[1]);
    b3_g(v, 1, 5, 9, 13, m[2], m[3]);
    b3_g(v, 2, 6, 10, 14, m[4], m[5]);
    b3_g(v, 3, 7, 11, 15, m[6], m[7]);
    b3_g(v, 0, 5, 10, 15, m[8], m[9]);
    b3_g(v, 1, 6, 11, 12, m[10], m[11]);
    b3_g(v, 2, 7, 8, 13, m[12], m[13]);
    b3_g(v, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      uint32_t t[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) t[i] = m[perm[i]];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}

XM_DEV uint32_t ld32(const uint8_t* p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

// Keyed BLAKE3 of a single 64-byte message (one chunk, one block, root): the noise and jackpot hashes.
XM_DEV void b3_keyed64(const Key8& key, const uint32_t msg[16], uint32_t out[8]) {
  b3_compress(key.w, msg, 0, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT | B3_KEYED, out);
}

// ------------------------------------------------------------------------------------------------ kernels

// Chaining values of full 1024-byte chunks (keyed, non-root). One thread per chunk.
extern "C" __global__ void xm_b3_chunk_cvs(const uint8_t* __restrict__ data, uint32_t nchunks, Key8 key,
                                           uint32_t* __restrict__ cvs) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nchunks) return;
  const uint8_t* p = data + (size_t)c * 1024;
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key.w[i];
  for (int b = 0; b < 16; ++b) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = ld32(p + b * 64 + i * 4);
    uint32_t f = B3_KEYED | (b == 0 ? B3_CHUNK_START : 0) | (b == 15 ? B3_CHUNK_END : 0);
    b3_compress(cv, m, c, 64, f, cv);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) cvs[(size_t)c * 8 + i] = cv[i];
}

// Noise hash H(index, slot) = keyed_blake3(key, i32le(index+1) at slot || zeros || label32).
XM_DEV void noise_hash(const Key8& key, uint32_t index, int slot, uint32_t side_b, uint32_t out[8]) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = 0;
  m[slot] = index + 1;
  // "A_tensor" / "B_tensor" as little-endian words
  m[8] = side_b ? 0x65745f42u : 0x65745f41u;  // bytes 'A'|'B', '_', 't', 'e'
  m[9] = 0x726f736eu;                          // "nsor"
  b3_keyed64(key, m, out);
}

// Uniform noise matrix (rows x rank) in [-32, 31]: one thread per 32 values (one hash).
extern "C" __global__ void xm_noise_uniform(Key8 key, uint32_t side_b, uint32_t nhash, int8_t* __restrict__ out) {
  uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nhash) return;
  uint32_t d[8];
  noise_hash(key, h, 0, side_b, d);
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)h * 32 + i * 4 + j] = (int8_t)((int)((d[i] >> (8 * j)) & 63) - 32);
}

// Sparse +1/-1 permutation: k index pairs (p0, p1). One thread per 8 lines (one hash).
extern "C" __global__ void xm_noise_perm(Key8 key, uint32_t side_b, uint32_t k, uint32_t rank,
                                         uint32_t* __restrict__ pairs) {
  uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h * 8 >= k) return;
  uint32_t d[8];
  noise_hash(key, h, 1, side_b, d);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint32_t l = h * 8 + j;
    if (l >= k) break;
    uint32_t u = d[j];
    uint32_t first = u & (rank - 1);
    uint32_t mulhi = (uint32_t)(((uint64_t)(rank - 1) * u) >> 32);
    pairs[2 * l] = first;
    pairs[2 * l + 1] = first ^ (1 + mulhi);
  }
}

// Packed fragment-major layout used by the GEMM (see pearl.hip): for 16-row block rb and 32-k pair kp, 512
// contiguous bytes; lane l holds row (rb*16 + l%16), k in [kp*32 + 8*(l/16), +8) then [kp*32 + 16 + 8*(l/16), +8).
XM_DEV size_t packed_offset(uint32_t row, uint32_t k, uint32_t K) {
  uint32_t rb = row >> 4, r = row & 15, kp = k >> 5, kin = k & 31;
  uint32_t half = kin >> 4, kk = kin & 15, lane = r + 16 * (kk >> 3);
  return ((size_t)rb * (K >> 5) + kp) * 512 + lane * 16 + half * 8 + (kk & 7);
}

// X' = X + U[row][p0[l]] - U[row][p1[l]], written in the packed layout. One thread per (row, 8 consecutive k).
extern "C" __global__ void xm_apply_noise(const int8_t* __restrict__ sig, const int8_t* __restrict__ uni,
                                          const uint32_t* __restrict__ pairs, int8_t* __restrict__ packed,
                                          uint32_t rows, uint32_t K, uint32_t rank) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t groups = K / 8;
  if (t >= (size_t)rows * groups) return;
  uint32_t row = (uint32_t)(t / groups), k0 = (uint32_t)(t % groups) * 8;
  const int8_t* s = sig + (size_t)row * K + k0;
  const int8_t* u = uni + (size_t)row * rank;
  int8_t v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t l = k0 + i;
    v[i] = (int8_t)(s[i] + u[pairs[2 * l]] - u[pairs[2 * l + 1]]);
  }
#ifdef XM_PEARL_ROWMAJOR
  int8_t* dst = packed + (size_t)row * K + k0;  // NVIDIA: plain row-major (K-major), what mma.sync expects
#else
  // k0..k0+7 stay inside one 8-byte run of the packed layout.
  int8_t* dst = packed + packed_offset(row, k0, K);
#endif
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[i] = v[i];
}

XM_DEV bool u256_le(const uint32_t a[8], const U256w& b) {
  for (int i = 7; i >= 0; --i)
    if (a[i] != b.w[i]) return a[i] < b.w[i];
  return true;
}

// One thread per 16x16 tile: combine the GEMM's two half-transcripts, hash with a_noise_seed, compare with the
// bound, and record hits (tile index + transcript) for the host.
extern "C" __global__ void xm_jackpot_check(const uint32_t* __restrict__ tr, uint32_t tiles, uint32_t tiles_n,
                                            Key8 a_seed, U256w bound,
                                            uint32_t* __restrict__ found, uint32_t max_found) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tiles) return;
  uint32_t m[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) m[w] = tr[(size_t)t * 32 + w] ^ tr[(size_t)t * 32 + w + 16];
  uint32_t h[8];
  b3_keyed64(a_seed, m, h);
  if (!u256_le(h, bound)) return;
  uint32_t slot = atomicAdd(&found[0], 1u);
  if (slot >= max_found) return;
  uint32_t* rec = found + 1 + slot * 18;
  rec[0] = t / tiles_n * 16;  // t_rows
  rec[1] = t % tiles_n * 16;  // t_cols
#pragma unroll
  for (int w = 0; w < 16; ++w) rec[2 + w] = m[w];
}
