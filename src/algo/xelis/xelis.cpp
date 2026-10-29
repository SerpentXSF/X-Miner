#include "algo/xelis/xelis.hpp"

#include <cmath>
#include <cstring>

#include "crypto/blake3.hpp"

namespace xm::xelis {

namespace {

// ---------------------------------------------------------------- 128-bit arithmetic (portable: MSVC has no __int128)
struct U128 {
  uint64_t hi, lo;
};
bool operator>(U128 a, U128 b) { return a.hi != b.hi ? a.hi > b.hi : a.lo > b.lo; }

uint64_t mulhi(uint64_t a, uint64_t b) {
  const uint64_t a0 = uint32_t(a), a1 = a >> 32, b0 = uint32_t(b), b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + uint32_t(p01) + uint32_t(p10);
  return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

int clz64(uint64_t x) {
  int n = 0;
  if (!x) return 64;
  while (!(x & (1ull << 63))) x <<= 1, ++n;
  return n;
}

// q = n / d, r = n % d for d != 0 (shift-subtract from the top set bit; the CPU only re-checks shares).
void divmod(U128 n, U128 d, U128& q, U128& r) {
  q = {0, 0};
  r = {0, 0};
  const int nb = n.hi ? 128 - clz64(n.hi) : 64 - clz64(n.lo);
  for (int i = nb - 1; i >= 0; --i) {
    r.hi = r.hi << 1 | r.lo >> 63;
    r.lo = r.lo << 1 | ((i >= 64 ? n.hi >> (i - 64) : n.lo >> i) & 1);
    if (!(d > r)) {
      const uint64_t borrow = r.lo < d.lo;
      r.lo -= d.lo;
      r.hi -= d.hi + borrow;
      if (i >= 64) q.hi |= 1ull << (i - 64);
      else q.lo |= 1ull << i;
    }
  }
}
uint64_t rem_lo(U128 n, U128 d) {
  U128 q, r;
  divmod(n, d, q, r);
  return r.lo;
}
uint64_t div_lo(U128 n, U128 d) {
  U128 q, r;
  divmod(n, d, q, r);
  return q.lo;
}

uint64_t rotl(uint64_t x, uint32_t r) {
  r &= 63;
  return r ? (x << r) | (x >> (64 - r)) : x;
}
uint64_t rotr(uint64_t x, uint32_t r) {
  r &= 63;
  return r ? (x >> r) | (x << (64 - r)) : x;
}

uint64_t murmurhash3(uint64_t s) {
  s ^= s >> 55;
  s *= 0xff51afd7ed558ccdull;
  s ^= s >> 32;
  s *= 0xc4ceb9fe1a85ec53ull;
  s ^= s >> 15;
  return s;
}

size_t map_index(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  return size_t(mulhi(x, kBufferWords));
}

bool pick_half(uint64_t seed) { return (murmurhash3(seed) & (1ull << 58)) != 0; }

uint64_t modular_power(uint64_t base, uint64_t exp, uint64_t mod) {
  uint64_t result = 1;
  if (mod == 0) return 0;  // the reference divides by zero (a zero 64-bit stage result); the GPU does the same
  const U128 m{0, mod};
  base = rem_lo({0, base}, m);
  while (exp > 0) {
    if (exp & 1) result = rem_lo({mulhi(result, base), result * base}, m);
    base = rem_lo({mulhi(base, base), base * base}, m);
    exp /= 2;
  }
  return result;
}

// ---------------------------------------------------------------- AES
constexpr uint8_t kSbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9,
    0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f,
    0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07,
    0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3,
    0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58,
    0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3,
    0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f,
    0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac,
    0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a,
    0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70,
    0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42,
    0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

uint8_t xtime(uint8_t x) { return static_cast<uint8_t>((x << 1) ^ ((x >> 7) * 0x1b)); }

// ---------------------------------------------------------------- ChaCha8 (IETF: 32-bit counter from 0, 96-bit nonce)
uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
void chacha_block(const uint32_t in[16], uint8_t out[64]) {
  uint32_t x[16];
  std::memcpy(x, in, 64);
  auto qr = [&](int a, int b, int c, int d) {
    x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16);
    x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12);
    x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);
    x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);
  };
  for (int i = 0; i < 4; ++i) {  // 8 rounds
    qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15);
    qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14);
  }
  for (int i = 0; i < 16; ++i) {
    const uint32_t v = x[i] + in[i];
    std::memcpy(out + 4 * i, &v, 4);  // little-endian
  }
}

}  // namespace

void chacha8_keystream(const uint8_t key[32], const uint8_t nonce[12], uint8_t* out, size_t len) {
  uint32_t st[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
  std::memcpy(st + 4, key, 32);
  st[12] = 0;
  std::memcpy(st + 13, nonce, 12);
  uint8_t block[64];
  for (size_t off = 0; off < len; off += 64, ++st[12]) {
    chacha_block(st, block);
    std::memcpy(out + off, block, len - off < 64 ? len - off : 64);
  }
}

void aes_round(uint8_t s[16], const uint8_t key[16]) {
  uint8_t t[16];
  for (int c = 0; c < 4; ++c)  // SubBytes + ShiftRows: row r of column c comes from column (c + r) % 4
    for (int r = 0; r < 4; ++r) t[4 * c + r] = kSbox[s[4 * ((c + r) % 4) + r]];
  for (int c = 0; c < 4; ++c) {  // MixColumns + AddRoundKey
    const uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
    const uint8_t all = a0 ^ a1 ^ a2 ^ a3;
    s[4 * c] = static_cast<uint8_t>(a0 ^ all ^ xtime(a0 ^ a1) ^ key[4 * c]);
    s[4 * c + 1] = static_cast<uint8_t>(a1 ^ all ^ xtime(a1 ^ a2) ^ key[4 * c + 1]);
    s[4 * c + 2] = static_cast<uint8_t>(a2 ^ all ^ xtime(a2 ^ a3) ^ key[4 * c + 2]);
    s[4 * c + 3] = static_cast<uint8_t>(a3 ^ all ^ xtime(a3 ^ a0) ^ key[4 * c + 3]);
  }
}

uint64_t isqrt(uint64_t n) {
  if (n < 2) return n;
  const uint64_t approx = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
  if (approx * approx > n) return approx - 1;  // wrapping products, as the reference
  if ((approx + 1) * (approx + 1) <= n) return approx + 1;
  return approx;
}

Hash hash(const uint8_t* input, size_t len, std::vector<uint64_t>& scratch) {
  scratch.assign(kMemoryWords, 0);
  uint8_t* bytes = reinterpret_cast<uint8_t*>(scratch.data());
  // Stage 1 (v2): ChaCha8 keystreams, one share of the scratchpad per 32-byte input chunk.
  Hash input_hash = blake3::hash(input, len);
  uint8_t nonce[12];
  std::memcpy(nonce, input_hash.data(), 12);
  const size_t chunks = (len + 31) / 32;
  size_t out_off = 0;
  for (size_t ci = 0; ci < chunks; ++ci) {
    uint8_t tmp[64] = {};
    std::memcpy(tmp, input_hash.data(), 32);
    const size_t clen = len - ci * 32 < 32 ? len - ci * 32 : 32;
    std::memcpy(tmp + 32, input + ci * 32, clen);
    input_hash = blake3::hash(tmp, 64);
    const size_t remaining = kMemoryBytes - out_off, left = chunks - ci;
    const size_t cur = remaining < remaining / left ? remaining : remaining / left;
    uint8_t* part = bytes + ci * cur;
    chacha8_keystream(input_hash.data(), nonce, part, cur);
    out_off += cur;
    std::memcpy(nonce, part + (cur >= 12 ? cur - 12 : 0), 12);
  }

  // Stage 3 (v3).
  uint64_t* a_buf = scratch.data();
  uint64_t* b_buf = scratch.data() + kBufferWords;
  static const uint8_t key[16] = {'x', 'e', 'l', 'i', 's', 'h', 'a', 's', 'h', '-', 'p', 'o', 'w', '-', 'v', '3'};
  uint64_t addr_a = b_buf[kBufferWords - 1];
  uint64_t addr_b = a_buf[kBufferWords - 1] >> 32;
  size_t r = 0;
  for (uint64_t i = 0; i < kIters; ++i) {
    const uint64_t mem_a = a_buf[map_index(addr_a)];
    const uint64_t mem_b = b_buf[map_index(mem_a ^ addr_b)];
    uint8_t block[16];
    std::memcpy(block, &mem_b, 8);
    std::memcpy(block + 8, &mem_a, 8);
    aes_round(block, key);
    uint64_t h1, h2;
    std::memcpy(&h1, block, 8);
    std::memcpy(&h2, block + 8, 8);
    uint64_t result = ~(h1 ^ h2);
    for (uint64_t j = 0; j < uint64_t(kBufferWords); ++j) {
      const uint64_t a = a_buf[map_index(result)];
      const uint64_t b = b_buf[map_index(a ^ ~rotr(result, uint32_t(r)))];
      const uint64_t c = r < size_t(kBufferWords) ? a_buf[r] : b_buf[r - kBufferWords];
      r = r < size_t(kMemoryWords) - 1 ? r + 1 : 0;
      uint64_t v;
      switch (rotl(result, uint32_t(c)) & 0xf) {
        case 0: {
          const U128 t1{a + i, isqrt(b + j)};
          v = rem_lo(t1, {0, murmurhash3(c ^ result ^ i ^ j) | 1});
          break;
        }
        case 1: {
          const uint64_t t1 = (c + i) % isqrt(b | 2);
          v = rotl(t1, uint32_t(i + j)) * isqrt(a + j);
          break;
        }
        case 2: v = (isqrt(a + i) * isqrt(c + j)) ^ (b + i + j); break;
        case 3: v = (a + b) * c; break;
        case 4: v = (b - c) * a; break;
        case 5: v = c - a + b; break;
        case 6: v = a - b + c; break;
        case 7: v = b * c + a; break;
        case 8: v = c * a + b; break;
        case 9: v = a * b * c; break;
        case 10: v = rem_lo({a, b}, {0, c | 1}); break;
        case 11: {
          const U128 t1{b, c}, t2{rotl(result, uint32_t(r)), a | 2};
          v = t2 > t1 ? c : rem_lo(t1, t2);
          break;
        }
        case 12: v = div_lo({c, a}, {0, b | 4}); break;
        case 13: {
          const U128 t1{rotl(result, uint32_t(r)), b}, t2{a, c | 8};
          v = t1 > t2 ? div_lo(t1, t2) : a ^ b;
          break;
        }
        case 14: v = b * c + mulhi(a, c); break;  // ((b:a) * c mod 2^128) >> 64
        default: {  // 15: ((a:c) * (rotr(result, r):b) mod 2^128) >> 64
          const uint64_t rh = rotr(result, uint32_t(r));
          v = a * b + c * rh + mulhi(c, b);
          break;
        }
      }
      const uint64_t seed = v ^ result;
      result = rotl(seed, uint32_t(r));
      const bool use_b = pick_half(v);
      const size_t it = map_index(seed);
      const uint64_t t = (use_b ? b_buf[it] : a_buf[it]) ^ result;
      const size_t ia = map_index(t ^ result ^ 0x9e3779b97f4a7c15ull);
      const size_t ib = map_index(uint64_t(ia) ^ ~result ^ 0xd2b74407b1ce6e93ull);
      const uint64_t old = a_buf[ia];
      a_buf[ia] = t;
      b_buf[ib] ^= old ^ rotr(t, uint32_t(i + j));
    }
    addr_a = modular_power(addr_a, addr_b, result);
    addr_b = isqrt(result) * (uint64_t(r) + 1) * isqrt(addr_a);
  }

  // Stage 4.
  return blake3::hash(bytes, kMemoryBytes);
}

bool meets_difficulty(const Hash& h, uint64_t difficulty) {
  if (difficulty == 0) return false;
  uint8_t target[32] = {};
  uint64_t rem = 0;
  for (int bit = 255; bit >= 0; --bit) {  // floor((2^256 - 1) / difficulty), big-endian
    const bool carry = rem >> 63;
    rem = rem << 1 | 1;
    if (carry || rem >= difficulty) {
      rem -= difficulty;
      target[31 - bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
    }
  }
  return std::memcmp(h.data(), target, 32) <= 0;
}

}  // namespace xm::xelis
