#include "algo/blake2b/blake2b.hpp"

#include <cmath>
#include <cstring>

namespace xm::b2b {

namespace {

constexpr uint64_t kIv[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                             0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                             0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
constexpr uint8_t kSigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

inline void g(uint64_t* v, int a, int b, int c, int d, uint64_t x, uint64_t y) {
  v[a] = v[a] + v[b] + x;
  v[d] = rotr(v[d] ^ v[a], 32);
  v[c] = v[c] + v[d];
  v[b] = rotr(v[b] ^ v[c], 24);
  v[a] = v[a] + v[b] + y;
  v[d] = rotr(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr(v[b] ^ v[c], 63);
}

}  // namespace

Digest blake2b256(const uint8_t* data, size_t len) {
  uint64_t h[8], m[16] = {}, v[16];
  std::memcpy(h, kIv, sizeof h);
  h[0] ^= 0x01010000ull ^ 32;  // parameter block: fanout 1, depth 1, no key, 32-byte digest
  for (size_t i = 0; i < len && i < 128; ++i) m[i >> 3] |= uint64_t(data[i]) << (8 * (i & 7));
  std::memcpy(v, h, sizeof h);
  std::memcpy(v + 8, kIv, sizeof kIv);
  v[12] ^= len;  // byte counter; single (final) block
  v[14] = ~v[14];
  for (const auto& s : kSigma) {
    g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    g(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    g(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
  }
  Digest out;
  for (int i = 0; i < 32; ++i) out[i] = static_cast<uint8_t>((h[i >> 3] ^ v[i >> 3] ^ v[8 + (i >> 3)]) >> (8 * (i & 7)));
  return out;
}

Digest work_root(const uint8_t coinb1[kCoinb1Size], const uint8_t extranonce[kExtranonceSize]) {
  uint8_t leaf[1 + kCoinb1Size + kExtranonceSize] = {0};
  std::memcpy(leaf + 1, coinb1, kCoinb1Size);
  std::memcpy(leaf + 1 + kCoinb1Size, extranonce, kExtranonceSize);
  return blake2b256(leaf, sizeof leaf);
}

Header work_header(const uint8_t prevhash[32], uint64_t nonce, const uint8_t ntime[8], const Digest& root) {
  Header h{};
  std::memcpy(h.data(), prevhash, 32);
  for (int i = 0; i < 8; ++i) h[32 + i] = static_cast<uint8_t>(nonce >> (8 * i));
  std::memcpy(h.data() + 40, ntime, 8);
  std::memcpy(h.data() + 48, root.data(), 32);
  return h;
}

bool meets_target(const Digest& d, const Target& t) {
  // The gateway byte-reverses the digest and compares little-endian (datum_pow.c, compare_hashes): the digest's
  // first byte is the most significant.
  for (int i = 0; i < 32; ++i) {
    if (d[i] != t[31 - i]) return d[i] < t[31 - i];
  }
  return true;
}

Target target_for_difficulty(double difficulty) {
  Target t{};
  if (!(difficulty > 0) || !std::isfinite(difficulty)) return t;
  // difficulty = m * 2^(e - 53) exactly, so target = floor(N / m) with N = (2^224 - 1) * 2^(53 - e). Bitwise long
  // division by the 53-bit m keeps it exact (pools send values like 16 - 2^-12 that floats would round).
  int e;
  double f = std::frexp(difficulty, &e);
  uint64_t m = static_cast<uint64_t>(std::ldexp(f, 53));
  const int shift = 53 - e;  // N has ones in bits [shift, shift + 224), clipped at 0
  const int top = 224 + shift;
  uint64_t rem = 0;
  for (int bit = top - 1; bit >= 0; --bit) {
    rem = rem << 1 | (bit >= shift ? 1u : 0u);
    if (rem >= m) {
      rem -= m;
      if (bit >= 256) {  // quotient beyond 256 bits: difficulty far below 1, saturate
        t.fill(0xff);
        return t;
      }
      t[bit >> 3] |= static_cast<uint8_t>(1u << (bit & 7));
    }
  }
  return t;
}

Target target_from_nbits(uint32_t nbits) {
  Target t{};
  const int exponent = static_cast<int>(nbits >> 24);
  const uint32_t mantissa = nbits & 0x007fffff;
  if ((nbits & 0x00800000) || mantissa == 0) return t;
  for (int i = 0; i < 3; ++i) {
    const int pos = exponent - 3 + i;  // byte index (little-endian) of mantissa byte i
    if (pos >= 32) return Target{};
    if (pos >= 0) t[pos] = static_cast<uint8_t>(mantissa >> (8 * i));
  }
  return t;
}

double target_work(const Target& t) {
  double v = 0;
  for (int i = 31; i >= 0; --i) v = v * 256.0 + t[i];
  return v > 0 ? std::ldexp(1.0, 256) / v : 0.0;
}

}  // namespace xm::b2b
