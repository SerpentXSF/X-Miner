#include "algo/autolykos/autolykos.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace xm::autolykos {

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

uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

void compress(uint64_t h[8], const uint8_t block[128], uint64_t t, bool last) {
  uint64_t m[16], v[16];
  std::memcpy(m, block, 128);
  for (int i = 0; i < 8; ++i) v[i] = h[i], v[i + 8] = kIv[i];
  v[12] ^= t;
  if (last) v[14] = ~v[14];
  auto g = [&](int a, int b, int c, int d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x; v[d] = rotr(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];     v[b] = rotr(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y; v[d] = rotr(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];     v[b] = rotr(v[b] ^ v[c], 63);
  };
  for (int r = 0; r < 12; ++r) {
    const uint8_t* s = kSigma[r];
    g(0, 4, 8, 12, m[s[0]], m[s[1]]);
    g(1, 5, 9, 13, m[s[2]], m[s[3]]);
    g(2, 6, 10, 14, m[s[4]], m[s[5]]);
    g(3, 7, 11, 15, m[s[6]], m[s[7]]);
    g(0, 5, 10, 15, m[s[8]], m[s[9]]);
    g(1, 6, 11, 12, m[s[10]], m[s[11]]);
    g(2, 7, 8, 13, m[s[12]], m[s[13]]);
    g(3, 4, 9, 14, m[s[14]], m[s[15]]);
  }
  for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

const std::vector<uint8_t>& table_suffix() {  // M: the 64-bit big-endian values 0..1023
  static const std::vector<uint8_t> m = [] {
    std::vector<uint8_t> v(8192, 0);
    for (int i = 0; i < 1024; ++i) {
      v[8 * i + 6] = static_cast<uint8_t>(i >> 8);
      v[8 * i + 7] = static_cast<uint8_t>(i);
    }
    return v;
  }();
  return m;
}

// Adds a 32-byte big-endian number into an accumulator (no overflow below 2^256 here).
void add_be(Bytes32& acc, const Bytes32& x) {
  unsigned carry = 0;
  for (int i = 31; i >= 0; --i) {
    const unsigned s = unsigned(acc[i]) + x[i] + carry;
    acc[i] = static_cast<uint8_t>(s);
    carry = s >> 8;
  }
}

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

}  // namespace

Bytes32 blake2b256(const uint8_t* data, size_t len) {
  uint64_t h[8];
  std::memcpy(h, kIv, sizeof h);
  h[0] ^= 0x01010000ull ^ 32;  // digest length 32, no key, fanout 1, depth 1
  uint8_t block[128];
  size_t done = 0;
  while (len - done > 128) {
    compress(h, data + done, done + 128, false);
    done += 128;
  }
  std::memset(block, 0, sizeof block);
  if (len > done) std::memcpy(block, data + done, len - done);
  compress(h, block, len, true);
  Bytes32 out;
  std::memcpy(out.data(), h, 32);
  return out;
}

uint32_t table_size(uint32_t height) {
  const uint32_t h = height < 4198400u ? height : 4198400u;
  uint64_t n = 1u << 26;
  if (h < 614400u) return uint32_t(n);
  const uint32_t iters = (h - 614400u) / 51200u + 1;
  for (uint32_t k = 0; k < iters; ++k) n = n / 100 * 105;
  return uint32_t(n);
}

Bytes32 element(uint32_t j, uint32_t height) {
  std::vector<uint8_t> in(8 + 8192);
  const uint8_t jb[4] = {uint8_t(j >> 24), uint8_t(j >> 16), uint8_t(j >> 8), uint8_t(j)};
  const uint8_t hb[4] = {uint8_t(height >> 24), uint8_t(height >> 16), uint8_t(height >> 8), uint8_t(height)};
  std::memcpy(in.data(), jb, 4);
  std::memcpy(in.data() + 4, hb, 4);
  std::memcpy(in.data() + 8, table_suffix().data(), 8192);
  Bytes32 d = blake2b256(in.data(), in.size());
  d[0] = 0;  // drop(1)
  return d;
}

Bytes32 hit(const Bytes32& msg, uint64_t nonce, uint32_t height) {
  const uint32_t n = table_size(height);
  uint8_t nb[8];
  for (int i = 0; i < 8; ++i) nb[i] = static_cast<uint8_t>(nonce >> (56 - 8 * i));
  uint8_t mn[40];
  std::memcpy(mn, msg.data(), 32);
  std::memcpy(mn + 32, nb, 8);
  const Bytes32 h1 = blake2b256(mn, 40);
  uint64_t prei8 = 0;
  for (int i = 24; i < 32; ++i) prei8 = prei8 << 8 | h1[i];
  const Bytes32 f = element(uint32_t(prei8 % n), height);
  uint8_t seed[71];
  std::memcpy(seed, f.data() + 1, 31);
  std::memcpy(seed + 31, msg.data(), 32);
  std::memcpy(seed + 63, nb, 8);
  const Bytes32 hs = blake2b256(seed, 71);
  uint8_t ext[35];
  std::memcpy(ext, hs.data(), 32);
  std::memcpy(ext + 32, hs.data(), 3);
  Bytes32 sum{};
  for (int k = 0; k < 32; ++k) add_be(sum, element(be32(ext + k) % n, height));
  return blake2b256(sum.data(), 32);
}

bool less_than(const Bytes32& a, const Bytes32& b) {
  for (int i = 0; i < 32; ++i)
    if (a[i] != b[i]) return a[i] < b[i];
  return false;
}

Bytes32 from_decimal(const std::string& dec) {
  Bytes32 v{};
  if (dec.empty()) return v;
  for (char c : dec) {
    if (c < '0' || c > '9') return Bytes32{};
    unsigned carry = unsigned(c - '0');
    for (int i = 31; i >= 0; --i) {  // v = v * 10 + digit
      const unsigned x = unsigned(v[i]) * 10 + carry;
      v[i] = static_cast<uint8_t>(x);
      carry = x >> 8;
    }
    if (carry) return Bytes32{};  // overflow
  }
  return v;
}

double work_for(const Bytes32& b) {
  int i = 0;
  while (i < 24 && b[i] == 0) ++i;
  double top = 0;
  for (int k = 0; k < 8; ++k) top = top * 256 + b[i + k];
  return top > 0 ? std::ldexp(1.0, 64 + 8 * i) / top : 0;  // 2^256 / b
}

}  // namespace xm::autolykos
