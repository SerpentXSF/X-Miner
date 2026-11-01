// NexaPow (Nexa, NEXA) CPU reference, written from the Nexa node source (gitlab.com/nexa/nexa, MIT: src/pow.cpp,
// src/primitives/block.cpp, src/key.cpp and the libsecp256k1 schnorr module):
//   miningHash = SHA256d(headerCommitment | u8 len(nonce) | nonce)
//   h1         = SHA256(miningHash)
//   sig        = BCH Schnorr signature of h1 with miningHash as the private key (big-endian; 0 or >= n: invalid)
//   powHash    = SHA256(sig), compared as a little-endian number with the target.
// Pools send the header commitment; the nonce is a byte vector of 1..16 bytes (12 on Miningcore pools: extranonce1
// (4) | 8 miner bytes). The Tailstorm sub-block form of h1 (Hard Fork 2) is not active on mainnet (2026-11-01).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace xm::nexa {

using Hash = std::array<uint8_t, 32>;    // as SHA-256 outputs it
using Target = std::array<uint8_t, 32>;  // little-endian 256-bit number

Hash mining_hash(const Hash& commitment, const uint8_t* nonce, size_t len);
// powHash for a nonce; false if the mining hash is not a valid private key (probability ~2^-128).
bool pow_hash(const Hash& commitment, const uint8_t* nonce, size_t len, Hash& out);

// Pool share difficulty d: target = floor(0xFFFF * 2^208 / d) (Bitcoin difficulty 1, as Miningcore's Nexa pools).
Target target_from_difficulty(double d);
Target target_from_nbits(uint32_t nbits);
bool meets(const Hash& h, const Target& t);  // h as a little-endian number <= t
uint64_t target_top64(const Target& t);

}  // namespace xm::nexa
