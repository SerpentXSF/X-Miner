// Ergo Autolykos v2 CPU reference (k = 32 elements, n = 26), from ergoplatform/ergo ergo-core
// mining/AutolykosPowScheme.scala (@c1a69f6, CC0):
//   N(h)    table size: 2^26 until height 614400, then x 105/100 (integer, divide first) every 51200 blocks, capped at
//           height 4198400.
//   elem(j) = Blake2b256(j as 4 bytes BE || h as 4 bytes BE || M).drop(1), a 31-byte big-endian number;
//           M = the 64-bit big-endian values 0..1023 (8192 bytes).
//   hit     i = (last 8 bytes of Blake2b256(msg || nonce), BE) mod N; seed = elem(i) || msg || nonce (71 bytes);
//           32 indexes = BE 4-byte windows 0..31 of (Blake2b256(seed) || its first 3 bytes), mod N;
//           hit = Blake2b256(32-byte BE sum of the 32 elements), a valid share when hit < b.
// msg = Blake2b256 of the header without the PoW (the pool sends it), nonce = 8 bytes (pool extranonce first).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace xm::autolykos {

using Bytes32 = std::array<uint8_t, 32>;  // big-endian numbers / digests, bytes in order

Bytes32 blake2b256(const uint8_t* data, size_t len);  // unkeyed, any length

uint32_t table_size(uint32_t height);  // N(h)
// Element j at height h as 32 bytes: a zero byte, then the digest's last 31 bytes (the big-endian number).
Bytes32 element(uint32_t j, uint32_t height);
// The hit (Blake2b256 of the element sum) for msg, an 8-byte nonce and the height.
Bytes32 hit(const Bytes32& msg, uint64_t nonce, uint32_t height);

bool less_than(const Bytes32& a, const Bytes32& b);  // both big-endian
Bytes32 from_decimal(const std::string& dec);  // a pool's decimal b (0 if it does not fit or is not a number)
// Hashes per share for target b: about 2^256 / b.
double work_for(const Bytes32& b);

}  // namespace xm::autolykos
