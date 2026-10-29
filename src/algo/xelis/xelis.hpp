// XelisHash v3 (Xelis, "xel/v3") CPU reference, written from xelis-project/xelis-hash src/v3.rs + src/v2.rs (MIT).
// Per hash: a 543,744-byte scratchpad filled with ChaCha8 keystreams keyed by BLAKE3 chains of the 112-byte input
// (stage 1), 2 x 33,984 sequential steps of random reads / writes over its two halves with 16 branch kinds of 64- and
// 128-bit arithmetic and one AES round per pass (stage 3), and BLAKE3 of the whole scratchpad (stage 4).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm::xelis {

constexpr int kMemoryWords = 531 * 128;        // u64 scratchpad words
constexpr int kMemoryBytes = kMemoryWords * 8;  // 543,744
constexpr int kBufferWords = kMemoryWords / 2;  // each half
constexpr int kIters = 2;
constexpr int kInputSize = 112;

using Hash = std::array<uint8_t, 32>;

// The 112-byte miner work: header work hash | timestamp (BE) | nonce (BE) | extra nonce (32) | miner public key (32).
struct Work {
  uint8_t bytes[kInputSize] = {};
  void set_nonce(uint64_t nonce) {
    for (int i = 0; i < 8; ++i) bytes[40 + i] = static_cast<uint8_t>(nonce >> (56 - 8 * i));
  }
};

// XelisHash v3 of the input; `scratch` is resized to kMemoryWords and may be reused.
Hash hash(const uint8_t* input, size_t len, std::vector<uint64_t>& scratch);

// Pieces exposed for tests.
void chacha8_keystream(const uint8_t key[32], const uint8_t nonce[12], uint8_t* out, size_t len);
void aes_round(uint8_t block[16], const uint8_t key[16]);
uint64_t isqrt(uint64_t n);

// The pool's share rule: hash (big-endian) <= 2^256 / difficulty, i.e. hash * difficulty fits in 256 bits.
bool meets_difficulty(const Hash& h, uint64_t difficulty);

}  // namespace xm::xelis
