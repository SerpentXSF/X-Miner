// Quantus (QTC) QPoW, miner side: CPU reference.
//
// Quantus-Network/chain qpow-math (lib.rs, qp-poseidon-core 3.1.0):
//   hash   = Poseidon2 sponge over Goldilocks (p = 2^64 - 2^32 + 1), width 12, rate 8, 8 full + 22 partial rounds,
//            S-box x^7, of the 96 bytes mining_hash(32) || nonce(64, big-endian U512), squeezed twice (64 bytes)
//   bytes  -> field elements 4 bytes at a time (little-endian u32), then a terminator element 1 and the sponge's own
//            padding element 1: absorbed blocks are header | nonce[0..32] | nonce[32..64] | [1, 1, 0 x 6]
//   output = state[0..4] as little-endian u64s after permutation 4, then again after permutation 5
//   valid  when the 64 output bytes read as a big-endian U512 are < U512::MAX / difficulty (strict).
// The pool's 128-hex target is that bound, so it is used as is. Checked against the official miner's
// NONCE_HASH_KVS vectors (quantus-miner pow-core) and qp-poseidon's permutation vectors.
#pragma once

#include <array>
#include <cstdint>

namespace xm::qtc {

constexpr uint64_t kP = 0xffffffff00000001ull;  // Goldilocks prime
using Hash = std::array<uint8_t, 64>;
using Target = std::array<uint8_t, 64>;  // big-endian U512
using Nonce = std::array<uint8_t, 64>;   // big-endian U512, as submitted

void permute(uint64_t s[12]);  // Poseidon2, canonical elements in and out
Hash nonce_hash(const uint8_t header[32], const Nonce& nonce);
bool meets_target(const Hash& h, const Target& t);  // h < t, both big-endian

// GPU per-job state: the sponge after permutations 1 and 2, nonce bytes 32..55 absorbed, the initial external
// linear layer and the first round constants applied. Bytes 56..63 (the low 64 bits) are left out: a kernel adds
// their linear contribution per nonce and resumes at the first S-box (the official miner's prestate split).
void prestate_low64(const uint8_t header[32], const Nonce& nonce, uint64_t out[12]);

}  // namespace xm::qtc
