// Bitcoin Knots BLAKE2b (RDTS) proof of work, miner side: CPU reference.
//
// The DATUM gateway hands hashers classic Sia-style work (datum_pow.c, datum_stratum.c):
//   work_root = blake2b-256(0x00 || coinb1(39) || extranonce1 || extranonce2)   (extranonce = 12 bytes)
//   header    = prevhash(32, already transformed by the gateway) || nonce(8, LE) || ntime(8) || work_root(32)
//   digest    = blake2b-256(header); valid when the digest read big-endian is <= the target.
// Targets are kept as 32-byte little-endian integers, as the gateway stores them. Share difficulty is Sia-style:
// target = (2^224 - 1) / difficulty. An 80-byte header is one BLAKE2b block, so each hash is one compression.
// Checked against the BLAKE2b activation block on Knots regtest and against hashlib.blake2b.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace xm::b2b {

using Digest = std::array<uint8_t, 32>;
using Target = std::array<uint8_t, 32>;  // little-endian
using Header = std::array<uint8_t, 80>;
constexpr size_t kCoinb1Size = 39;
constexpr size_t kExtranonceSize = 12;

Digest blake2b256(const uint8_t* data, size_t len);  // unkeyed, 32-byte digest; len <= 128 (one block)
Digest work_root(const uint8_t coinb1[kCoinb1Size], const uint8_t extranonce[kExtranonceSize]);
Header work_header(const uint8_t prevhash[32], uint64_t nonce, const uint8_t ntime[8], const Digest& root);

bool meets_target(const Digest& d, const Target& t);  // digest big-endian <= target (equality passes)
Target target_for_difficulty(double difficulty);      // exact floor((2^224 - 1) / difficulty); 0 -> all zero
Target target_from_nbits(uint32_t nbits);              // network target from compact bits (0 on bad input)
double target_work(const Target& t);                   // hashes per share: 2^256 / target

}  // namespace xm::b2b
