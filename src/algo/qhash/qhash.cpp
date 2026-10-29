#include "algo/qhash/qhash.hpp"

#include <cmath>
#include <cstring>

#include "crypto/sha256.hpp"

namespace xm::qhash {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

bool odd_angles(uint32_t ntime) { return ntime >= kOddAnglesTime; }

uint32_t zero_limit(uint32_t ntime) {
  if (ntime >= 1754220531) return 8;
  if (ntime >= 1753305380) return 24;
  if (ntime >= 1753105444) return 32;
  return 33;
}

// The reference applies, per layer and qubit i, exp(i th Y) then exp(i ph Z) with th, ph = -(2n + odd) pi / 32, then
// the CNOT chain. Layer-1 RY/RZ make a product state a_k = RZ RY |0>; the chain turns it into
// psi(y) = prod_k a_k(y_k ^ y_{k-1}) (y = prefix XOR), a chain with a 2-dimensional bond y_k. Layer 2: the RZ is
// diagonal and drops out of Z measurements; the second chain turns Z_j into the parity Z_0..Z_j of the layer-2 qubits.
// So <Z_j> = contraction of the bond density rho (2x2 Hermitian) through sites 0..j with Z inserted after each RY,
// closed by the identity on sites > j. With rho = (t I + z Z + x X + y Y) / 2 and per site (n1, n2, n3 = the site's
// layer-1 RY, layer-1 RZ and layer-2 RY nibbles, c(n) / s(n) = cos / sin((2n + odd) pi / 16)):
//   A = s(n1) c(n2), B = s(n1) s(n2), C = c(n1), C2 = c(n3), S2 = -s(n3)
//   t' = C2 (C z - B y), z' = C2 (t + A x), x' = S2 (x + A t), y' = S2 (C y + B z)
//   <Z_j> = t' + A_{j+1} x' (j < 15), t' + x' (j = 15); start t = z = 1, x = y = 0 (bond y_{-1} = 0).
// The identity closure contributes 1 on the bond diagonal and 2 Re(a0 conj a1) = A of the next site off it; after the
// last site the ket and bra bond indices are summed freely (off-diagonal weight 1).
std::array<double, kQubits> expectations(const uint8_t nib[64], bool odd) {
  double cs[16], sn[16];
  for (int n = 0; n < 16; ++n) {
    const double a = (2.0 * n + (odd ? 1 : 0)) * kPi / 16;
    cs[n] = std::cos(a);
    sn[n] = std::sin(a);
  }
  auto site_a = [&](int k) { return sn[nib[k]] * cs[nib[16 + k]]; };
  std::array<double, kQubits> e{};
  double t = 1, z = 1, x = 0, y = 0, A = site_a(0);
  for (int k = 0; k < kQubits; ++k) {
    const double B = sn[nib[k]] * sn[nib[16 + k]], C = cs[nib[k]];
    const double C2 = cs[nib[32 + k]], S2 = -sn[nib[32 + k]];
    const double t2 = C2 * (C * z - B * y), z2 = C2 * (t + A * x), x2 = S2 * (x + A * t), y2 = S2 * (C * y + B * z);
    t = t2, z = z2, x = x2, y = y2;
    const double next = k + 1 < kQubits ? site_a(k + 1) : 1.0;
    e[k] = t + x * next;
    A = next;
  }
  return e;
}

int16_t to_fixed(double x) {
  const double v = x >= 0 ? x * 32768 + 0.5 : x * 32768 - 0.5;
  return static_cast<int16_t>(static_cast<uint16_t>(static_cast<uint32_t>(static_cast<int32_t>(v))));
}

Hash hash(const uint8_t hdr[kHeaderSize]) {
  const uint32_t ntime = uint32_t(hdr[68]) | uint32_t(hdr[69]) << 8 | uint32_t(hdr[70]) << 16 | uint32_t(hdr[71]) << 24;
  const auto inner = crypto::sha256(hdr, kHeaderSize);
  uint8_t nib[64];
  for (int i = 0; i < 32; ++i) {
    nib[2 * i] = inner[i] >> 4;
    nib[2 * i + 1] = inner[i] & 15;
  }
  const auto e = expectations(nib, odd_angles(ntime));
  uint8_t buf[64];
  std::memcpy(buf, inner.data(), 32);
  uint32_t zeros = 0;
  for (int q = 0; q < kQubits; ++q) {
    const uint16_t f = static_cast<uint16_t>(to_fixed(e[q]));
    buf[32 + 2 * q] = uint8_t(f);
    buf[33 + 2 * q] = uint8_t(f >> 8);
    zeros += (buf[32 + 2 * q] == 0) + (buf[33 + 2 * q] == 0);
  }
  Hash out;
  if (zeros >= zero_limit(ntime)) {
    out.fill(0xff);
    return out;
  }
  const auto h = crypto::sha256(buf, sizeof buf);
  std::memcpy(out.data(), h.data(), 32);
  return out;
}

namespace {

// 256-bit big number as 8 little-endian u32 limbs.
using U256 = std::array<uint32_t, 8>;

Target to_target(const U256& v) {
  Target t{};
  for (int i = 0; i < 8; ++i)
    for (int b = 0; b < 4; ++b) t[4 * i + b] = uint8_t(v[i] >> (8 * b));
  return t;
}

}  // namespace

Target target_from_difficulty(double d) {
  // floor(0xFFFF * 2^240 / m) with m = round(d * 2^32): the 0xFFFF * 2^208 / d of the pool, exact for d with up to
  // 32 fraction bits. Bitwise long division (once per job).
  if (!(d > 0)) d = 1;
  if (d > 4294967295.0) d = 4294967295.0;
  const uint64_t m = static_cast<uint64_t>(std::llround(d * 4294967296.0));
  U256 num{};
  num[7] = 0xFFFF0000u;  // 0xFFFF * 2^240
  U256 q{};
  uint64_t rem = 0;
  for (int bit = 255; bit >= 0; --bit) {
    const bool carry = rem >> 63;
    rem = rem << 1 | ((num[bit / 32] >> (bit % 32)) & 1);
    if (carry || rem >= m) {
      rem -= m;
      q[bit / 32] |= 1u << (bit % 32);
    }
  }
  return to_target(q);
}

Target target_from_nbits(uint32_t nbits) {
  U256 v{};
  const int exp = int(nbits >> 24);
  const uint32_t mant = nbits & 0x007fffff;
  // value = mant * 256^(exp - 3)
  for (int i = 0; i < 3; ++i) {
    const int byte = exp - 3 + i;
    const uint32_t b = (mant >> (8 * i)) & 0xff;
    if (byte >= 0 && byte < 32) v[byte / 4] |= b << (8 * (byte % 4));
  }
  return to_target(v);
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

std::vector<uint8_t> extranonce2_bytes(const Template& t, uint64_t en2) {
  std::vector<uint8_t> v(static_cast<size_t>(t.extranonce2_size > 0 ? t.extranonce2_size : 0));
  for (size_t i = 0; i < v.size() && i < 8; ++i) v[i] = uint8_t(en2 >> (8 * i));
  return v;
}

std::string extranonce2_hex(const Template& t, uint64_t en2) {
  static const char* hx = "0123456789abcdef";
  std::string s;
  for (uint8_t b : extranonce2_bytes(t, en2)) {
    s += hx[b >> 4];
    s += hx[b & 15];
  }
  return s;
}

void header(const Template& t, uint64_t en2, uint8_t out[kHeaderSize]) {
  std::vector<uint8_t> cb = t.coinb1;
  cb.insert(cb.end(), t.extranonce1.begin(), t.extranonce1.end());
  const auto e2 = extranonce2_bytes(t, en2);
  cb.insert(cb.end(), e2.begin(), e2.end());
  cb.insert(cb.end(), t.coinb2.begin(), t.coinb2.end());
  auto root = crypto::sha256d(cb.data(), cb.size());
  for (const auto& br : t.branches) {
    uint8_t pair[64];
    std::memcpy(pair, root.data(), 32);
    std::memcpy(pair + 32, br.data(), 32);
    root = crypto::sha256d(pair, 64);
  }
  auto le32 = [&](int at, uint32_t v) {
    for (int i = 0; i < 4; ++i) out[at + i] = uint8_t(v >> (8 * i));
  };
  le32(0, t.version);
  std::memcpy(out + 4, t.prevhash.data(), 32);
  std::memcpy(out + 36, root.data(), 32);
  le32(68, t.ntime);
  le32(72, t.nbits);
  le32(76, 0);
}

bool prevhash_from_stratum(const std::string& hex, std::array<uint8_t, 32>& out) {
  if (hex.size() != 64) return false;
  uint8_t raw[32];
  for (int i = 0; i < 32; ++i) {
    unsigned v = 0;
    for (int k = 0; k < 2; ++k) {
      const char c = hex[2 * i + k];
      int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (d < 0) return false;
      v = v * 16 + unsigned(d);
    }
    raw[i] = uint8_t(v);
  }
  for (int w = 0; w < 8; ++w)
    for (int b = 0; b < 4; ++b) out[4 * w + b] = raw[4 * w + 3 - b];
  return true;
}

}  // namespace xm::qhash
