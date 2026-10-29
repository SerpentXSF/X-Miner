// QHash (Qubitcoin, QTC) CPU reference, written from super-quantum/qubitcoin src/crypto/qhash.cpp (MIT).
// hash = SHA256(h | fixed16(<Z_0>) .. fixed16(<Z_15>)), h = SHA256(80-byte header): the 64 nibbles of h set the angles
// of a 16-qubit circuit (2 layers of RY + RZ on every qubit, then CNOT 0->1, 1->2, .., 14->15), <Z_q> are its Z
// expectations as Q15 fixed point (little-endian int16). Hashes with too many zero bytes in those 32 bytes are
// invalid (all 0xFF). The circuit is evaluated exactly without a 2^16 state vector: after the first layer and CNOT
// chain the state is a bond-dimension-2 chain, so each <Z_j> is a contraction of 4-component transfer steps (see
// qhash.cpp); the result matches a full state-vector simulation to ~1e-14 and real block hashes bit for bit.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xm::qhash {

constexpr int kQubits = 16;
constexpr int kHeaderSize = 80;
// Consensus switches by block time (nTime): odd angles (2n + 1) pi / 32 from 1758762000; zero-byte rules from
// 1753105444 (all 32 zero), 1753305380 (>= 24) and 1754220531 (>= 8).
constexpr uint32_t kOddAnglesTime = 1758762000;

using Hash = std::array<uint8_t, 32>;    // as SHA-256 outputs it; the block hash is these bytes as a LE number
using Target = std::array<uint8_t, 32>;  // little-endian 256-bit number

bool odd_angles(uint32_t ntime);
// Zero bytes (of the 32 fixed-point bytes) at which a hash becomes invalid; 33 = never.
uint32_t zero_limit(uint32_t ntime);

// <Z_q> of the circuit for 64 nibbles (0..15).
std::array<double, kQubits> expectations(const uint8_t nibbles[64], bool odd);
// The reference's double -> Q15 conversion: round half away from zero, then truncate to 16 bits.
int16_t to_fixed(double x);

Hash hash(const uint8_t header[kHeaderSize]);

// Pool share difficulty d (Bitcoin units): target = floor(0xFFFF * 2^208 / d).
Target target_from_difficulty(double d);
// Compact nBits -> target.
Target target_from_nbits(uint32_t nbits);
// hash (as a LE number) <= target
bool meets(const Hash& h, const Target& t);
// Top 64 bits of a target (GPU pre-filter).
uint64_t target_top64(const Target& t);

// Bitcoin stratum v1 work: header = version | prevhash | merkle root | ntime | nbits | nonce (all LE), merkle root =
// SHA256d(coinb1 | extranonce1 | extranonce2 | coinb2) folded with the branches.
struct Template {
  std::vector<uint8_t> coinb1, coinb2, extranonce1;
  int extranonce2_size = 4;
  std::vector<std::array<uint8_t, 32>> branches;
  std::array<uint8_t, 32> prevhash{};  // header order
  uint32_t version = 0, nbits = 0, ntime = 0;
};

// The extranonce2 bytes for a counter value (little-endian, zero-padded / truncated to the pool's size).
std::vector<uint8_t> extranonce2_bytes(const Template& t, uint64_t en2);
std::string extranonce2_hex(const Template& t, uint64_t en2);
// The 80-byte header with nonce 0.
void header(const Template& t, uint64_t en2, uint8_t out[kHeaderSize]);
// Stratum prevhash (8 words, each byte-swapped) -> header order.
bool prevhash_from_stratum(const std::string& hex, std::array<uint8_t, 32>& out);

}  // namespace xm::qhash
