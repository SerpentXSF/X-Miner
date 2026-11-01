#include "algo/nexa/nexa.hpp"

#include <cmath>
#include <cstring>

#include "crypto/secp256k1.hpp"
#include "crypto/sha256.hpp"

namespace xm::nexa {

Hash mining_hash(const Hash& commitment, const uint8_t* nonce, size_t len) {
  uint8_t buf[32 + 1 + 255];
  std::memcpy(buf, commitment.data(), 32);
  buf[32] = uint8_t(len);
  std::memcpy(buf + 33, nonce, len);
  return crypto::sha256d(buf, 33 + len);
}

bool pow_hash(const Hash& commitment, const uint8_t* nonce, size_t len, Hash& out) {
  const Hash mh = mining_hash(commitment, nonce, len);
  const Hash h1 = crypto::sha256(mh.data(), 32);
  uint8_t sig[64];
  if (!crypto::secp256k1::schnorr_sign_bch(h1.data(), mh.data(), sig)) return false;
  out = crypto::sha256(sig, sizeof sig);
  return true;
}

Target target_from_difficulty(double d) {
  // floor(0xFFFF * 2^240 / m) with m = round(d * 2^32): exact for d with up to 32 fraction bits.
  if (!(d > 0)) d = 1;
  if (d > 4294967295.0) d = 4294967295.0;
  const uint64_t m = static_cast<uint64_t>(std::llround(d * 4294967296.0));
  uint32_t num[8] = {}, q[8] = {};
  num[7] = 0xFFFF0000u;
  uint64_t rem = 0;
  for (int bit = 255; bit >= 0; --bit) {
    const bool carry = rem >> 63;
    rem = rem << 1 | ((num[bit / 32] >> (bit % 32)) & 1);
    if (carry || rem >= m) {
      rem -= m;
      q[bit / 32] |= 1u << (bit % 32);
    }
  }
  Target t{};
  for (int i = 0; i < 32; ++i) t[i] = uint8_t(q[i / 4] >> (8 * (i % 4)));
  return t;
}

Target target_from_nbits(uint32_t nbits) {
  Target t{};
  const int exp = int(nbits >> 24);
  const uint32_t mant = nbits & 0x007fffff;
  for (int i = 0; i < 3; ++i) {
    const int byte = exp - 3 + i;
    if (byte >= 0 && byte < 32) t[byte] = uint8_t(mant >> (8 * i));
  }
  return t;
}

bool meets(const Hash& h, const Target& t) {
  for (int i = 31; i >= 0; --i) {
    if (h[i] != t[i]) return h[i] < t[i];
  }
  return true;
}

uint64_t target_top64(const Target& t) {
  uint64_t v = 0;
  for (int i = 31; i >= 24; --i) v = v << 8 | t[i];
  return v;
}

}  // namespace xm::nexa
