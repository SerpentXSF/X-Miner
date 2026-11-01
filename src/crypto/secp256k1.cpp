#include "crypto/secp256k1.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "crypto/sha256.hpp"

namespace xm::crypto::secp256k1 {

namespace {

constexpr U256 make(uint32_t w7, uint32_t w6, uint32_t w5, uint32_t w4, uint32_t w3, uint32_t w2, uint32_t w1,
                    uint32_t w0) {
  U256 r;
  r.w[0] = w0; r.w[1] = w1; r.w[2] = w2; r.w[3] = w3; r.w[4] = w4; r.w[5] = w5; r.w[6] = w6; r.w[7] = w7;
  return r;
}

int cmp(const U256& a, const U256& b) {
  for (int i = 7; i >= 0; --i) {
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
  }
  return 0;
}

bool is_zero(const U256& a) {
  for (uint32_t v : a.w) {
    if (v) return false;
  }
  return true;
}

// r = a + b, returns the carry out.
uint32_t add_raw(U256& r, const U256& a, const U256& b) {
  uint64_t c = 0;
  for (int i = 0; i < 8; ++i) {
    c += uint64_t(a.w[i]) + b.w[i];
    r.w[i] = uint32_t(c);
    c >>= 32;
  }
  return uint32_t(c);
}

// r = a - b, returns the borrow out.
uint32_t sub_raw(U256& r, const U256& a, const U256& b) {
  int64_t c = 0;
  for (int i = 0; i < 8; ++i) {
    c += int64_t(a.w[i]) - b.w[i];
    r.w[i] = uint32_t(c);
    c >>= 32;
  }
  return uint32_t(c & 1);
}

// x mod m for a little-endian number of any length, where 2^256 = c (mod m) and c < 2^256: x = lo + hi * c folded
// until the high part is zero, then m subtracted while x >= m.
// A product: up to 16 limbs, plus room for the folds below.
struct Wide {
  uint32_t w[18] = {};
};

// x mod m (x of 16 limbs), where 2^256 = c (mod m) and c < 2^256: x = lo + hi * c folded until the high part is
// zero, then m subtracted while x >= m.
U256 reduce(Wide x, const U256& m, const U256& c) {
  int cl = 8;
  while (cl > 1 && c.w[cl - 1] == 0) --cl;
  for (;;) {
    int n = 18;
    while (n > 8 && x.w[n - 1] == 0) --n;
    if (n <= 8) break;
    Wide r;
    std::memcpy(r.w, x.w, 8 * 4);
    for (int i = 8; i < n; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < cl; ++j) {
        const int k = i - 8 + j;
        carry += uint64_t(x.w[i]) * c.w[j] + r.w[k];
        r.w[k] = uint32_t(carry);
        carry >>= 32;
      }
      for (int k = i - 8 + cl; carry && k < 18; ++k) {
        carry += r.w[k];
        r.w[k] = uint32_t(carry);
        carry >>= 32;
      }
    }
    x = r;
  }
  U256 v;
  std::memcpy(v.w, x.w, sizeof v.w);
  while (cmp(v, m) >= 0) sub_raw(v, v, m);
  return v;
}

Wide mul_raw(const U256& a, const U256& b) {
  Wide r;
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 8; ++j) {
      carry += uint64_t(a.w[i]) * b.w[j] + r.w[i + j];
      r.w[i + j] = uint32_t(carry);
      carry >>= 32;
    }
    r.w[i + 8] = uint32_t(carry);
  }
  return r;
}

const U256 kPc = make(0, 0, 0, 0, 0, 0, 1, 0x3D1);  // 2^256 mod p = 2^32 + 977
const U256 kNc = make(0, 0, 0, 1, 0x45512319, 0x50B75FC4, 0x402DA173, 0x2FC9BEBF);  // 2^256 - n

struct Jac {
  U256 x, y, z;  // z = 0: infinity
};

Jac jdouble(const Jac& p) {
  if (is_zero(p.z) || is_zero(p.y)) return {U256{}, U256{}, U256{}};
  const U256 yy = fe_mul(p.y, p.y);
  const U256 s = fe_mul(fe_mul(p.x, yy), make(0, 0, 0, 0, 0, 0, 0, 4));
  const U256 xx = fe_mul(p.x, p.x);
  const U256 m = fe_add(fe_add(xx, xx), xx);
  const U256 x3 = fe_sub(fe_mul(m, m), fe_add(s, s));
  const U256 yyyy8 = fe_mul(fe_mul(yy, yy), make(0, 0, 0, 0, 0, 0, 0, 8));
  const U256 y3 = fe_sub(fe_mul(m, fe_sub(s, x3)), yyyy8);
  const U256 z3 = fe_mul(fe_add(p.y, p.y), p.z);
  return {x3, y3, z3};
}

Jac jadd(const Jac& p, const Jac& q) {
  if (is_zero(p.z)) return q;
  if (is_zero(q.z)) return p;
  const U256 z1z1 = fe_mul(p.z, p.z), z2z2 = fe_mul(q.z, q.z);
  const U256 u1 = fe_mul(p.x, z2z2), u2 = fe_mul(q.x, z1z1);
  const U256 s1 = fe_mul(fe_mul(p.y, q.z), z2z2), s2 = fe_mul(fe_mul(q.y, p.z), z1z1);
  if (u1 == u2) {
    if (s1 == s2) return jdouble(p);
    return {U256{}, U256{}, U256{}};
  }
  const U256 h = fe_sub(u2, u1), r = fe_sub(s2, s1);
  const U256 hh = fe_mul(h, h), hhh = fe_mul(hh, h), v = fe_mul(u1, hh);
  const U256 x3 = fe_sub(fe_sub(fe_mul(r, r), hhh), fe_add(v, v));
  const U256 y3 = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_mul(s1, hhh));
  const U256 z3 = fe_mul(fe_mul(p.z, q.z), h);
  return {x3, y3, z3};
}

}  // namespace

bool U256::operator==(const U256& o) const { return std::memcmp(w, o.w, sizeof w) == 0; }

const U256 kP = make(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F);
const U256 kN = make(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141);

U256 from_be(const uint8_t b[32]) {
  U256 r;
  for (int i = 0; i < 8; ++i) {
    const uint8_t* p = b + 28 - 4 * i;
    r.w[i] = uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
  }
  return r;
}

void to_be(const U256& x, uint8_t b[32]) {
  for (int i = 0; i < 8; ++i) {
    uint8_t* p = b + 28 - 4 * i;
    p[0] = uint8_t(x.w[i] >> 24); p[1] = uint8_t(x.w[i] >> 16); p[2] = uint8_t(x.w[i] >> 8); p[3] = uint8_t(x.w[i]);
  }
}

U256 fe_add(const U256& a, const U256& b) {
  U256 r;
  if (add_raw(r, a, b) || cmp(r, kP) >= 0) sub_raw(r, r, kP);
  return r;
}

U256 fe_sub(const U256& a, const U256& b) {
  U256 r;
  if (sub_raw(r, a, b)) add_raw(r, r, kP);
  return r;
}

U256 fe_mul(const U256& a, const U256& b) { return reduce(mul_raw(a, b), kP, kPc); }

U256 fe_pow(const U256& a, const U256& e) {
  U256 r = make(0, 0, 0, 0, 0, 0, 0, 1);
  for (int bit = 255; bit >= 0; --bit) {
    r = fe_mul(r, r);
    if (e.w[bit / 32] >> (bit % 32) & 1) r = fe_mul(r, a);
  }
  return r;
}

U256 fe_inv(const U256& a) {
  U256 e;
  sub_raw(e, kP, make(0, 0, 0, 0, 0, 0, 0, 2));
  return fe_pow(a, e);
}

bool fe_is_qr(const U256& a) {
  U256 e = kP;  // (p - 1) / 2
  for (int i = 0; i < 8; ++i) e.w[i] = e.w[i] >> 1 | (i < 7 ? e.w[i + 1] << 31 : 0);
  return fe_pow(a, e) == make(0, 0, 0, 0, 0, 0, 0, 1);
}

U256 sc_add(const U256& a, const U256& b) {
  U256 r;
  if (add_raw(r, a, b) || cmp(r, kN) >= 0) sub_raw(r, r, kN);
  return r;
}

U256 sc_mul(const U256& a, const U256& b) { return reduce(mul_raw(a, b), kN, kNc); }

U256 sc_reduce(const U256& a) {
  U256 r = a;
  if (cmp(r, kN) >= 0) sub_raw(r, r, kN);
  return r;
}

const Point& generator() {
  static const Point g = {
      make(0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798),
      make(0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8), false};
  return g;
}

Point mul_g(const U256& k) {
  const Point& g = generator();
  const Jac gj{g.x, g.y, make(0, 0, 0, 0, 0, 0, 0, 1)};
  Jac r{U256{}, U256{}, U256{}};
  for (int bit = 255; bit >= 0; --bit) {
    r = jdouble(r);
    if (k.w[bit / 32] >> (bit % 32) & 1) r = jadd(r, gj);
  }
  Point out;
  if (is_zero(r.z)) return out;
  const U256 zi = fe_inv(r.z), zi2 = fe_mul(zi, zi);
  out.x = fe_mul(r.x, zi2);
  out.y = fe_mul(r.y, fe_mul(zi2, zi));
  out.inf = false;
  return out;
}

void comb_table(int w, int windows, uint32_t* out) {
  const size_t half = size_t(1) << (w - 1);
  const U256 one = make(0, 0, 0, 0, 0, 0, 0, 1);
  // window bases 2^(w i) G
  std::vector<Jac> bases(windows);
  const Point& g = generator();
  Jac base{g.x, g.y, one};
  for (int i = 0; i < windows; ++i) {
    bases[i] = base;
    for (int d = 0; d < w; ++d) base = jdouble(base);
  }
  // chunks of consecutive multiples j B (j = j0 .. j0 + len - 1), each with its own batched inversion, on all cores
  const size_t chunk = std::min<size_t>(half, 8192);
  const size_t per_window = (half + chunk - 1) / chunk, tasks = per_window * size_t(windows);
  std::atomic<size_t> next{0};
  auto work = [&] {
    std::vector<Jac> jac(chunk);
    std::vector<U256> pre(chunk);
    for (size_t t; (t = next++) < tasks;) {
      const int wi = int(t / per_window);
      const size_t j0 = 1 + (t % per_window) * chunk, len = std::min(chunk, half + 1 - j0);
      const Jac& b = bases[wi];
      Jac acc{U256{}, U256{}, U256{}};  // j0 B by double-and-add
      for (int bit = 63; bit >= 0; --bit) {
        acc = jdouble(acc);
        if (j0 >> bit & 1) acc = jadd(acc, b);
      }
      for (size_t k = 0; k < len; ++k) {
        jac[k] = acc;
        if (k + 1 < len) acc = jadd(acc, b);
      }
      U256 run = one;  // Montgomery's trick: prefix products of z, one inversion, then back to front
      for (size_t k = 0; k < len; ++k) {
        pre[k] = run;
        run = fe_mul(run, jac[k].z);
      }
      U256 inv = fe_inv(run);
      for (size_t k = len; k-- > 0;) {
        const U256 zi = fe_mul(inv, pre[k]);
        inv = fe_mul(inv, jac[k].z);
        const U256 zi2 = fe_mul(zi, zi);
        const U256 x = fe_mul(jac[k].x, zi2), y = fe_mul(jac[k].y, fe_mul(zi2, zi));
        uint32_t* o = out + 16 * (size_t(wi) * half + j0 - 1 + k);
        std::memcpy(o, x.w, 32);
        std::memcpy(o + 8, y.w, 32);
      }
    }
  };
  const unsigned n = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned k = 1; k < n; ++k) pool.emplace_back(work);
  work();
  for (auto& th : pool) th.join();
}

std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* msg, size_t msg_len) {
  uint8_t k[64] = {};
  if (key_len > 64) {
    const Sha256Digest d = sha256(key, key_len);
    std::memcpy(k, d.data(), 32);
  } else {
    std::memcpy(k, key, key_len);
  }
  std::vector<uint8_t> inner(64 + msg_len);
  for (int i = 0; i < 64; ++i) inner[i] = k[i] ^ 0x36;
  if (msg_len) std::memcpy(inner.data() + 64, msg, msg_len);
  const Sha256Digest ih = sha256(inner.data(), inner.size());
  uint8_t outer[96];
  for (int i = 0; i < 64; ++i) outer[i] = k[i] ^ 0x5c;
  std::memcpy(outer + 64, ih.data(), 32);
  return sha256(outer, sizeof outer);
}

std::array<uint8_t, 32> rfc6979(const uint8_t key32[32], const uint8_t msg32[32], const uint8_t algo16[16],
                                unsigned counter) {
  uint8_t seed[80];
  std::memcpy(seed, key32, 32);
  std::memcpy(seed + 32, msg32, 32);
  std::memcpy(seed + 64, algo16, 16);
  std::array<uint8_t, 32> v, k;
  v.fill(1);
  k.fill(0);
  uint8_t buf[32 + 1 + 80];
  auto step = [&](uint8_t sep, bool with_seed) {  // K = HMAC(K, V | sep [| seed]); V = HMAC(K, V)
    std::memcpy(buf, v.data(), 32);
    buf[32] = sep;
    size_t n = 33;
    if (with_seed) {
      std::memcpy(buf + 33, seed, 80);
      n += 80;
    }
    k = hmac_sha256(k.data(), 32, buf, n);
    v = hmac_sha256(k.data(), 32, v.data(), 32);
  };
  step(0, true);
  step(1, true);
  for (unsigned i = 0;; ++i) {
    if (i) step(0, false);
    v = hmac_sha256(k.data(), 32, v.data(), 32);
    if (i == counter) return v;
  }
}

bool schnorr_sign_bch(const uint8_t msg32[32], const uint8_t seckey32[32], uint8_t sig64[64]) {
  static const uint8_t kAlgo[16] = {'S', 'c', 'h', 'n', 'o', 'r', 'r', '+', 'S', 'H', 'A', '2', '5', '6', ' ', ' '};
  const U256 x = from_be(seckey32);
  if (is_zero(x) || cmp(x, kN) >= 0) return false;
  const Point pub = mul_g(x);
  U256 k;
  for (unsigned c = 0;; ++c) {
    const auto kb = rfc6979(seckey32, msg32, kAlgo, c);
    k = from_be(kb.data());
    if (!is_zero(k) && cmp(k, kN) < 0) break;
  }
  const Point r = mul_g(k);
  if (!fe_is_qr(r.y)) sub_raw(k, kN, k);
  uint8_t buf[32 + 33 + 32];
  to_be(r.x, buf);
  buf[32] = uint8_t(2 + (pub.y.w[0] & 1));
  to_be(pub.x, buf + 33);
  std::memcpy(buf + 65, msg32, 32);
  const Sha256Digest eh = sha256(buf, sizeof buf);
  const U256 e = sc_reduce(from_be(eh.data()));
  const U256 s = sc_add(k, sc_mul(e, x));
  to_be(r.x, sig64);
  to_be(s, sig64 + 32);
  return true;
}

}  // namespace xm::crypto::secp256k1
