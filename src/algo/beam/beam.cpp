#include "algo/beam/beam.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "algo/blake2b/blake2b.hpp"
#include "crypto/sha256.hpp"

namespace xm::beam {

namespace {

inline uint64_t rotl(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }

inline void sip_round(uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {
  v0 += v1;
  v2 += v3;
  v1 = rotl(v1, 13);
  v3 = rotl(v3, 16);
  v1 ^= v0;
  v3 ^= v2;
  v0 = rotl(v0, 32);
  v2 += v1;
  v0 += v3;
  v1 = rotl(v1, 17);
  v3 = rotl(v3, 21);
  v1 ^= v2;
  v3 ^= v0;
  v2 = rotl(v2, 32);
}

// e >> n for 0 < n < 64 (512 bits).
Elem shr(const Elem& e, int n) {
  Elem r{};
  for (int i = 0; i < 8; ++i) r.w[i] = (e.w[i] >> n) | (i < 7 ? e.w[i + 1] << (64 - n) : 0);
  return r;
}

void mask_to(Elem& e, int bits) {
  for (int i = 0; i < 8; ++i) {
    const int lo = 64 * i;
    if (bits <= lo) e.w[i] = 0;
    else if (bits < lo + 64) e.w[i] &= (1ull << (bits - lo)) - 1;
  }
}

bool is_zero(const Elem& e) {
  for (uint64_t w : e.w)
    if (w) return false;
  return true;
}

}  // namespace

PrePow pre_pow(const uint8_t input[32], const uint8_t nonce[8], const uint8_t extra[4]) {
  uint8_t personal[16] = {'B', 'e', 'a', 'm', '-', 'P', 'o', 'W'};
  const uint32_t work_bits = kWorkBits, rounds = kRounds;
  for (int i = 0; i < 4; ++i) {
    personal[8 + i] = static_cast<uint8_t>(work_bits >> (8 * i));
    personal[12 + i] = static_cast<uint8_t>(rounds >> (8 * i));
  }
  uint8_t msg[44];
  std::memcpy(msg, input, 32);
  std::memcpy(msg + 32, nonce, 8);
  std::memcpy(msg + 40, extra, 4);
  const auto d = b2b::blake2b256_personal(msg, sizeof msg, personal);
  PrePow p;
  for (int i = 0; i < 4; ++i) {
    p.v[i] = 0;
    for (int b = 0; b < 8; ++b) p.v[i] |= uint64_t(d[8 * i + b]) << (8 * b);
  }
  return p;
}

uint64_t siphash(const PrePow& p, uint64_t nonce) {
  uint64_t v0 = p.v[0], v1 = p.v[1], v2 = p.v[2], v3 = p.v[3] ^ nonce;
  sip_round(v0, v1, v2, v3);
  sip_round(v0, v1, v2, v3);
  v0 ^= nonce;
  v2 ^= 0xff;
  for (int i = 0; i < 4; ++i) sip_round(v0, v1, v2, v3);
  return v0 ^ v1 ^ v2 ^ v3;
}

Elem seed(const PrePow& p, uint32_t index) {
  Elem e{};
  for (uint32_t j = 0; j < 7; ++j) e.w[j] = siphash(p, uint64_t((index << 3) + j));
  return e;
}

uint64_t mix(const Elem& e, int rem_len, const uint32_t* idx, int n) {
  Elem t = e;
  for (int i = 0; i < n; ++i) {
    const int pos = rem_len + i * (kCollisionBits + 1);
    if (pos >= 512) break;
    const uint64_t v = idx[i];  // < 2^25
    t.w[pos >> 6] |= v << (pos & 63);
    if ((pos & 63) > 64 - kIndexBits && (pos >> 6) < 7) t.w[(pos >> 6) + 1] |= v >> (64 - (pos & 63));
  }
  uint64_t r = 0;
  for (int i = 0; i < 8; ++i) r += rotl(t.w[i], (29 * (i + 1)) & 63);
  return rotl(r, 24);
}

Elem combine(const Elem& a, const Elem& b, int rem_len) {
  Elem x;
  for (int i = 0; i < 8; ++i) x.w[i] = a.w[i] ^ b.w[i];
  x = shr(x, kCollisionBits);
  mask_to(x, rem_len);
  return x;
}

Indices unpack_indices(const uint8_t* sol100) {
  Indices r{};
  for (int i = 0; i < kIndices; ++i) {
    uint32_t v = 0;
    for (int b = 0; b < kIndexBits; ++b) {
      const int bit = i * kIndexBits + b;
      v |= uint32_t((sol100[bit >> 3] >> (bit & 7)) & 1) << b;
    }
    r[i] = v;
  }
  return r;
}

std::array<uint8_t, 100> pack_indices(const Indices& idx) {
  std::array<uint8_t, 100> r{};
  for (int i = 0; i < kIndices; ++i)
    for (int b = 0; b < kIndexBits; ++b)
      if ((idx[i] >> b) & 1) {
        const int bit = i * kIndexBits + b;
        r[bit >> 3] |= uint8_t(1u << (bit & 7));
      }
  return r;
}

bool verify(const uint8_t input[32], const uint8_t nonce[8], const Solution& sol) {
  const PrePow p = pre_pow(input, nonce, sol.data() + 100);
  const Indices idx = unpack_indices(sol.data());
  struct Node {
    Elem e;
    std::vector<uint32_t> tree;
  };
  std::vector<Node> x(kIndices);
  for (int i = 0; i < kIndices; ++i) {
    if (idx[i] >= kElements) return false;
    x[i].e = seed(p, idx[i]);
    x[i].tree = {idx[i]};
  }
  for (int r = 1; x.size() > 1; ++r) {
    std::vector<Node> next;
    for (size_t i = 0; i < x.size(); i += 2) {
      Node& a = x[i];
      Node& b = x[i + 1];
      a.e.w[0] = mix(a.e, mix_len(r), a.tree.data(), std::min<int>(mix_indices(r), int(a.tree.size())));
      b.e.w[0] = mix(b.e, mix_len(r), b.tree.data(), std::min<int>(mix_indices(r), int(b.tree.size())));
      if (((a.e.w[0] ^ b.e.w[0]) & ((1u << kCollisionBits) - 1)) != 0) return false;  // collision
      for (uint32_t ia : a.tree)
        for (uint32_t ib : b.tree)
          if (ia == ib) return false;  // distinct indices
      if (!(a.tree[0] < b.tree[0])) return false;  // canonical order
      Node c;
      c.e = combine(a.e, b.e, out_len(r));
      c.tree = a.tree;
      c.tree.insert(c.tree.end(), b.tree.begin(), b.tree.end());
      next.push_back(std::move(c));
    }
    x = std::move(next);
  }
  return is_zero(x[0].e);
}

bool target_reached(const uint8_t sha256_digest[32], uint32_t packed) {
  constexpr uint32_t kInf = uint32_t(256 - 24) << 24;
  if (packed > kInf) return false;
  const uint32_t order = packed >> 24, mantissa = (1u << 24) | (packed & 0xFFFFFF);
  // hv (big-endian) * mantissa must have no bit at or above 280 - order.
  uint32_t limb[9] = {};  // little-endian 32-bit limbs of the product
  uint64_t carry = 0;
  for (int i = 0; i < 8; ++i) {
    const uint8_t* q = sha256_digest + 28 - 4 * i;
    const uint32_t w = uint32_t(q[0]) << 24 | uint32_t(q[1]) << 16 | uint32_t(q[2]) << 8 | q[3];
    const uint64_t t = uint64_t(w) * mantissa + carry;
    limb[i] = static_cast<uint32_t>(t);
    carry = t >> 32;
  }
  limb[8] = static_cast<uint32_t>(carry);
  const int bound = 280 - int(order);
  for (int bit = 9 * 32 - 1; bit >= bound; --bit)
    if ((limb[bit >> 5] >> (bit & 31)) & 1) return false;
  return true;
}

double difficulty_value(uint32_t packed) {
  const uint32_t order = packed >> 24, mantissa = (1u << 24) | (packed & 0xFFFFFF);
  return std::ldexp(double(mantissa), int(order) - 24);
}

bool meets_difficulty(const Solution& sol, uint32_t packed) {
  const auto d = crypto::sha256(sol.data(), sol.size());
  return target_reached(d.data(), packed);
}

std::vector<Solution> solve_cpu(const uint8_t input[32], const uint8_t nonce[8]) {
  const uint8_t extra[4] = {0, 0, 0, 0};
  const PrePow p = pre_pow(input, nonce, extra);
  // Round r's elements: work bits + all 2^(r-1) leaf indices (flat).
  std::vector<Elem> e(kElements);
  std::vector<uint32_t> tree(kElements);
  for (uint32_t i = 0; i < kElements; ++i) {
    e[i] = seed(p, i);
    tree[i] = i;
  }
  std::vector<Solution> out;
  for (int r = 1; r <= kRounds; ++r) {
    const size_t n = e.size(), leaves = size_t(1) << (r - 1);
    const int ml = mix_len(r), mi = mix_indices(r);
    std::vector<uint64_t> order(n);
    for (size_t i = 0; i < n; ++i) {
      e[i].w[0] = mix(e[i], ml, &tree[i * leaves], mi);
      order[i] = (e[i].w[0] & ((1u << kCollisionBits) - 1)) << 32 | i;
    }
    std::sort(order.begin(), order.end());
    std::vector<Elem> ne;
    std::vector<uint32_t> nt;
    ne.reserve(n + n / 8);
    nt.reserve((n + n / 8) * leaves * 2);
    for (size_t i = 0; i < n;) {
      size_t j = i + 1;
      while (j < n && (order[j] >> 32) == (order[i] >> 32)) ++j;
      for (size_t a = i; a < j; ++a)
        for (size_t b = a + 1; b < j; ++b) {
          uint32_t ia = static_cast<uint32_t>(order[a]), ib = static_cast<uint32_t>(order[b]);
          if (tree[ib * leaves] < tree[ia * leaves]) std::swap(ia, ib);
          const Elem c = combine(e[ia], e[ib], out_len(r));
          if (r < kRounds) {
            if (is_zero(c)) continue;  // identical subtrees (duplicate indices): never part of a valid solution
            ne.push_back(c);
            nt.insert(nt.end(), &tree[ia * leaves], &tree[ia * leaves] + leaves);
            nt.insert(nt.end(), &tree[ib * leaves], &tree[ib * leaves] + leaves);
          } else if (is_zero(c)) {
            Indices idx;
            std::copy(&tree[ia * leaves], &tree[ia * leaves] + leaves, idx.begin());
            std::copy(&tree[ib * leaves], &tree[ib * leaves] + leaves, idx.begin() + leaves);
            Solution s{};
            const auto packed = pack_indices(idx);
            std::copy(packed.begin(), packed.end(), s.begin());
            if (verify(input, nonce, s)) out.push_back(s);
          }
        }
      i = j;
    }
    e = std::move(ne);
    tree = std::move(nt);
  }
  return out;
}

}  // namespace xm::beam
