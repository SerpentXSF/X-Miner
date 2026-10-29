#include "crypto/sha256.hpp"

#include <cstring>

namespace xm::crypto {

const uint32_t kSha256Init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

const uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

namespace {
inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

void sha256_compress(uint32_t s[8], const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = uint32_t(block[4 * i]) << 24 | uint32_t(block[4 * i + 1]) << 16 | uint32_t(block[4 * i + 2]) << 8 | block[4 * i + 3];
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + kSha256K[i] + w[i];
    const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

Sha256Digest sha256(const uint8_t* data, size_t len) {
  uint32_t s[8];
  std::memcpy(s, kSha256Init, sizeof s);
  size_t i = 0;
  for (; i + 64 <= len; i += 64) sha256_compress(s, data + i);
  uint8_t tail[128] = {};
  const size_t rest = len - i;
  if (rest) std::memcpy(tail, data + i, rest);
  tail[rest] = 0x80;
  const size_t blocks = rest + 9 <= 64 ? 1 : 2;
  const uint64_t bits = uint64_t(len) * 8;
  for (int k = 0; k < 8; ++k) tail[64 * blocks - 1 - k] = uint8_t(bits >> (8 * k));
  for (size_t b = 0; b < blocks; ++b) sha256_compress(s, tail + 64 * b);
  Sha256Digest out;
  for (int k = 0; k < 8; ++k)
    for (int j = 0; j < 4; ++j) out[4 * k + j] = uint8_t(s[k] >> (24 - 8 * j));
  return out;
}

Sha256Digest sha256d(const uint8_t* data, size_t len) {
  const Sha256Digest h = sha256(data, len);
  return sha256(h.data(), h.size());
}

}  // namespace xm::crypto
