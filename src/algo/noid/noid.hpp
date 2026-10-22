// Parano1d (NOID) proof-of-work: CPU reference.
//
// digest = Poseidon2b sponge over GF(2^128) (t = 4, rate 2, x^7 S-box, 8 full + 58 partial rounds), POWHDR__ domain,
// absorbing the 16-field header schedule (8 rate blocks, no padding); valid when digest < target as 256-bit
// little-endian integers. Field values are tower-basis GF(2^128) elements; the permutation runs in the flat
// (polynomial, x^128 + x^7 + x^2 + x + 1) basis. Constants: noid_consts.hpp, generated from the official sources
// (github.com/ignotusnemo/parano1d, Apache-2.0). Checked against 64 header -> digest vectors.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace xm::noid {

struct U128 {
  uint64_t lo = 0, hi = 0;
  U128 operator^(const U128& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
  U128& operator^=(const U128& o) {
    lo ^= o.lo;
    hi ^= o.hi;
    return *this;
  }
  bool operator==(const U128& o) const { return lo == o.lo && hi == o.hi; }
  bool operator!=(const U128& o) const { return !(*this == o); }
};

using State = std::array<U128, 4>;
using Fields = std::array<U128, 16>;   // tower basis, field 10 = nonce
using Digest = std::array<uint8_t, 32>;
constexpr int kNonceField = 10;

U128 gf_mul(U128 a, U128 b);             // flat basis
U128 tower_to_flat(U128 v);
U128 flat_to_tower(U128 v);
void permute_flat(State& s);

Digest digest(const Fields& tower_fields);
// Per-job midstate: the sponge state (flat) after absorbing fields 0..9 (5 permutations).
State midstate(const Fields& tower_fields);
// Finish from a midstate with a given nonce (tower basis) and fields 11..15: 3 permutations.
Digest digest_from_midstate(const State& mid, const Fields& tower_fields, U128 nonce);

bool less_le256(const Digest& d, const uint8_t target_le[32]);  // strict: equality fails

U128 parse_u128_be_hex(const std::string& hex);  // "dc1b..." (32 hex, big-endian printout)
U128 u128_from_le_bytes(const uint8_t* p);

}  // namespace xm::noid
