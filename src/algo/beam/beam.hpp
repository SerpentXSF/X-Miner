// Beam (BEAM) BeamHash III, miner side: CPU reference (verifier and a slow solver for tests), written from Beam's
// source (BeamMW/beam, Apache-2.0: 3rdparty/crypto/beamHashIII_impl.cpp, pow/beamHash.cpp, core/difficulty.cpp):
//   prePow   = blake2b-256 personalised "Beam-PoW" | u32 448 | u32 5, over input(32) | nonce(8) | extra nonce(4)
//              (input = the pool's job "input", the header hash for PoW; nonce = pool prefix + miner bytes; extra
//              nonce = solution bytes 100..103), read as four little-endian u64 = the SipHash-2-4 state (no key
//              constants)
//   element  = 448 work bits: 64-bit word j of element i is siphash(prePow, 8 i + j), j = 0..6 (low word first);
//              2^25 elements (indices of 25 bits)
//   round r  = 1..5: every element is "mixed" (its low 64 bits replaced by a sum of rotated 64-bit words of its work
//              bits with its first leaf indices appended, see mix()), then pairs that agree on the low 24 bits are
//              combined: (a xor b) >> 24, masked to the round's remaining length; the subtree with the smaller first
//              index goes first. Round 5's combination must leave 24 zero bits.
//   solution = the 32 leaf indices packed as 25-bit little-endian fields (100 bytes) + the 4-byte extra nonce
//   share    = sha256(solution, 104 bytes) read big-endian, times the difficulty's mantissa, below 2^(280 - order)
//              (Beam's packed difficulty: order = top 8 bits, mantissa = 2^24 | low 24 bits)
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm::beam {

constexpr int kWorkBits = 448;
constexpr int kCollisionBits = 24;
constexpr int kRounds = 5;
constexpr int kIndexBits = kCollisionBits + 1;  // 25
constexpr uint32_t kElements = 1u << kIndexBits;
constexpr int kIndices = 32;
constexpr size_t kSolutionBytes = 104;

using Solution = std::array<uint8_t, kSolutionBytes>;
using Indices = std::array<uint32_t, kIndices>;

struct PrePow {
  uint64_t v[4];
};
PrePow pre_pow(const uint8_t input[32], const uint8_t nonce[8], const uint8_t extra[4]);
uint64_t siphash(const PrePow& p, uint64_t nonce);

// A 512-bit value (little-endian 64-bit words); work bits occupy the low remLen bits.
struct Elem {
  uint64_t w[8];
};
Elem seed(const PrePow& p, uint32_t index);
// Bits that remain before round r's mix, and after round r's combination (r = 1..5).
constexpr int mix_len(int r) { return kWorkBits - (r - 1) * kCollisionBits - (r == 5 ? 64 : 0); }
constexpr int out_len(int r) { return r == 5 ? kCollisionBits : kWorkBits - r * kCollisionBits - (r == 4 ? 64 : 0); }
// Leaf indices the mix of round r reads: min((512 - mix_len + 24) / 25, 2^(r-1)) = 1, 2, 4, 6, 9.
constexpr int mix_indices(int r) {
  return ((512 - mix_len(r)) + kCollisionBits) / (kCollisionBits + 1) < (1 << (r - 1))
             ? ((512 - mix_len(r)) + kCollisionBits) / (kCollisionBits + 1)
             : (1 << (r - 1));
}
// The mixed low word: work bits (rem_len of them) with idx[0..n) appended at rem_len + 25 i (bits >= 512 dropped),
// sum of rotl(word j, 29 (j + 1) mod 64) over the eight words, rotated left by 24.
uint64_t mix(const Elem& e, int rem_len, const uint32_t* idx, int n);
// (a xor b) >> 24, masked to rem_len bits.
Elem combine(const Elem& a, const Elem& b, int rem_len);

Indices unpack_indices(const uint8_t* sol100);
std::array<uint8_t, 100> pack_indices(const Indices& idx);

// Full BeamHash III check of a 104-byte solution (structure only, not the share difficulty).
bool verify(const uint8_t input[32], const uint8_t nonce[8], const Solution& sol);

// Beam packed difficulty.
bool target_reached(const uint8_t sha256_digest[32], uint32_t packed);
double difficulty_value(uint32_t packed);  // solutions per share on average: mantissa * 2^order / 2^24
// The share check: sha256(solution) against the packed difficulty.
bool meets_difficulty(const Solution& sol, uint32_t packed);

// Slow CPU solver (several GB of memory, tens of seconds): every solution of (input, nonce, extra nonce 0), each
// verified. For tests and development only.
std::vector<Solution> solve_cpu(const uint8_t input[32], const uint8_t nonce[8]);

}  // namespace xm::beam
