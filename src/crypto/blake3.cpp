#include "crypto/blake3.hpp"

#include <cstring>

namespace xm::blake3 {

const Cv kIV = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

namespace {

constexpr uint8_t kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void g(uint32_t* v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx;
  v[d] = rotr(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my;
  v[d] = rotr(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];
  v[b] = rotr(v[b] ^ v[c], 7);
}

inline uint32_t load32(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

// Output of a node, kept unfinalised so it can become either a chaining value or the root.
struct Output {
  Cv cv;
  uint8_t block[64];
  uint64_t counter;
  uint32_t block_len, flags;
};

Output chunk_output(const Cv& key, const uint8_t* data, size_t len, uint64_t index, uint32_t flags) {
  Cv cv = key;
  size_t nblocks = len == 0 ? 1 : (len + kBlockLen - 1) / kBlockLen;
  Output out{};
  for (size_t b = 0; b < nblocks; ++b) {
    uint8_t block[64] = {};
    size_t off = b * kBlockLen;
    size_t n = len - off < kBlockLen ? len - off : kBlockLen;
    if (len) std::memcpy(block, data + off, n);
    uint32_t f = flags | (b == 0 ? CHUNK_START : 0);
    if (b + 1 == nblocks) {
      out.cv = cv;
      std::memcpy(out.block, block, 64);
      out.counter = index;
      out.block_len = static_cast<uint32_t>(len ? n : 0);
      out.flags = f | CHUNK_END;
      return out;
    }
    auto o = compress(cv, block, index, kBlockLen, f);
    std::copy(o.begin(), o.begin() + 8, cv.begin());
  }
  return out;
}

Output parent_output(const Cv& key, const Cv& left, const Cv& right, uint32_t flags) {
  Output out{};
  out.cv = key;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      out.block[i * 4 + j] = static_cast<uint8_t>(left[i] >> (8 * j));
      out.block[32 + i * 4 + j] = static_cast<uint8_t>(right[i] >> (8 * j));
    }
  }
  out.counter = 0;
  out.block_len = kBlockLen;
  out.flags = flags | PARENT;
  return out;
}

Cv output_cv(const Output& o) {
  auto r = compress(o.cv, o.block, o.counter, o.block_len, o.flags);
  Cv cv;
  std::copy(r.begin(), r.begin() + 8, cv.begin());
  return cv;
}

void output_root(const Output& o, uint8_t* out, size_t out_len) {
  uint64_t counter = 0;
  size_t done = 0;
  while (done < out_len) {
    auto r = compress(o.cv, o.block, counter++, o.block_len, o.flags | ROOT);
    for (int i = 0; i < 16 && done < out_len; ++i)
      for (int j = 0; j < 4 && done < out_len; ++j) out[done++] = static_cast<uint8_t>(r[i] >> (8 * j));
  }
}

// Largest power of two strictly less than n (n >= 2).
uint64_t left_len(uint64_t n) {
  uint64_t p = 1;
  while (p * 2 < n) p *= 2;
  return p;
}

// CV of a subtree covering chunks [first, first + count) of data (not the root).
Cv subtree_cv(const Cv& key, const uint8_t* data, size_t len, uint64_t first, uint64_t count, uint32_t flags) {
  if (count == 1) {
    size_t off = first * kChunkLen;
    size_t n = len - off < kChunkLen ? len - off : kChunkLen;
    return output_cv(chunk_output(key, data + off, n, first, flags));
  }
  uint64_t l = left_len(count);
  return output_cv(parent_output(key, subtree_cv(key, data, len, first, l, flags),
                                 subtree_cv(key, data, len, first + l, count - l, flags), flags));
}

Output root_output(const Cv& key, const uint8_t* data, size_t len, uint32_t flags) {
  uint64_t chunks = len == 0 ? 1 : (len + kChunkLen - 1) / kChunkLen;
  if (chunks == 1) return chunk_output(key, data, len, 0, flags);
  uint64_t l = left_len(chunks);
  return parent_output(key, subtree_cv(key, data, len, 0, l, flags), subtree_cv(key, data, len, l, chunks - l, flags),
                       flags);
}

}  // namespace

std::array<uint32_t, 16> compress(const Cv& cv, const uint8_t block[64], uint64_t counter, uint32_t block_len,
                                  uint32_t flags) {
  uint32_t m[16];
  for (int i = 0; i < 16; ++i) m[i] = load32(block + 4 * i);
  uint32_t v[16] = {cv[0],  cv[1],  cv[2],  cv[3],  cv[4],  cv[5],
                    cv[6],  cv[7],  kIV[0], kIV[1], kIV[2], kIV[3],
                    static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), block_len, flags};
  for (int round = 0; round < 7; ++round) {
    g(v, 0, 4, 8, 12, m[0], m[1]);
    g(v, 1, 5, 9, 13, m[2], m[3]);
    g(v, 2, 6, 10, 14, m[4], m[5]);
    g(v, 3, 7, 11, 15, m[6], m[7]);
    g(v, 0, 5, 10, 15, m[8], m[9]);
    g(v, 1, 6, 11, 12, m[10], m[11]);
    g(v, 2, 7, 8, 13, m[12], m[13]);
    g(v, 3, 4, 9, 14, m[14], m[15]);
    if (round < 6) {
      uint32_t t[16];
      for (int i = 0; i < 16; ++i) t[i] = m[kPerm[i]];
      std::memcpy(m, t, sizeof m);
    }
  }
  std::array<uint32_t, 16> out;
  for (int i = 0; i < 8; ++i) {
    out[i] = v[i] ^ v[i + 8];
    out[i + 8] = v[i + 8] ^ cv[i];
  }
  return out;
}

Cv key_words(const uint8_t key[32]) {
  Cv k;
  for (int i = 0; i < 8; ++i) k[i] = load32(key + 4 * i);
  return k;
}

Hash cv_bytes(const Cv& cv) {
  Hash h;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) h[i * 4 + j] = static_cast<uint8_t>(cv[i] >> (8 * j));
  return h;
}

Cv chunk_cv(const Cv& key, const uint8_t* data, size_t len, uint64_t chunk_index, uint32_t flags) {
  return output_cv(chunk_output(key, data, len, chunk_index, flags));
}

Cv parent_cv(const Cv& key, const Cv& left, const Cv& right, uint32_t flags) {
  return output_cv(parent_output(key, left, right, flags));
}

Hash parent_root(const Cv& key, const Cv& left, const Cv& right, uint32_t flags) {
  Hash h;
  output_root(parent_output(key, left, right, flags), h.data(), 32);
  return h;
}

std::vector<uint8_t> hash_xof(const uint8_t* data, size_t len, size_t out_len, const uint8_t* key) {
  std::vector<uint8_t> out(out_len);
  Cv k = key ? key_words(key) : kIV;
  output_root(root_output(k, data, len, key ? KEYED_HASH : 0), out.data(), out_len);
  return out;
}

Hash hash(const uint8_t* data, size_t len) {
  Hash h;
  output_root(root_output(kIV, data, len, 0), h.data(), 32);
  return h;
}

Hash keyed_hash(const uint8_t key[32], const uint8_t* data, size_t len) {
  Hash h;
  output_root(root_output(key_words(key), data, len, KEYED_HASH), h.data(), 32);
  return h;
}

Hash derive_key(const char* context, const uint8_t* material, size_t len) {
  Hash ctx;
  output_root(root_output(kIV, reinterpret_cast<const uint8_t*>(context), std::strlen(context), DERIVE_KEY_CONTEXT),
              ctx.data(), 32);
  Hash h;
  output_root(root_output(key_words(ctx.data()), material, len, DERIVE_KEY_MATERIAL), h.data(), 32);
  return h;
}

}  // namespace xm::blake3
