// BLAKE3, written from the specification (https://github.com/BLAKE3-team/BLAKE3-specs).
// Portable and simple rather than SIMD-fast: X-Miner hashes small messages on the CPU and does bulk hashing on
// the GPU. The chunk/parent building blocks are exposed because Pearl's matrix commitments are BLAKE3 trees whose
// inner nodes appear in shares (Merkle multiproofs).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm::blake3 {

using Hash = std::array<uint8_t, 32>;
using Cv = std::array<uint32_t, 8>;

enum : uint32_t {
  CHUNK_START = 1,
  CHUNK_END = 2,
  PARENT = 4,
  ROOT = 8,
  KEYED_HASH = 16,
  DERIVE_KEY_CONTEXT = 32,
  DERIVE_KEY_MATERIAL = 64,
};
constexpr size_t kBlockLen = 64;
constexpr size_t kChunkLen = 1024;

extern const Cv kIV;

// The compression function. Returns the full 16-word output (first 8 = new chaining value).
std::array<uint32_t, 16> compress(const Cv& cv, const uint8_t block[64], uint64_t counter, uint32_t block_len,
                                  uint32_t flags);

Cv key_words(const uint8_t key[32]);
Hash cv_bytes(const Cv& cv);

// Chaining value of one chunk (len <= 1024) that is NOT the root.
Cv chunk_cv(const Cv& key, const uint8_t* data, size_t len, uint64_t chunk_index, uint32_t flags);
// Chaining value of a parent node that is NOT the root.
Cv parent_cv(const Cv& key, const Cv& left, const Cv& right, uint32_t flags);
// Root output of a parent node.
Hash parent_root(const Cv& key, const Cv& left, const Cv& right, uint32_t flags);

// Hashes (32-byte output).
Hash hash(const uint8_t* data, size_t len);
Hash keyed_hash(const uint8_t key[32], const uint8_t* data, size_t len);
Hash derive_key(const char* context, const uint8_t* material, size_t len);
// Extendable output of the hash (XOF), for test vectors.
std::vector<uint8_t> hash_xof(const uint8_t* data, size_t len, size_t out_len, const uint8_t* key = nullptr);

}  // namespace xm::blake3
