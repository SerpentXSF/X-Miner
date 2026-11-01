// secp256k1 for checking NexaPow on the CPU: field and scalar arithmetic on 8 x 32-bit limbs, Jacobian points,
// RFC6979 nonces (HMAC-SHA256 DRBG, as libsecp256k1's nonce_function_rfc6979) and the BCH-2019 Schnorr signature
// that Nexa signs with (libsecp256k1 schnorr module: R.y must be a quadratic residue, e = SHA256(R.x | compressed P | m)).
// Written for correctness, not speed (a signature takes well under a millisecond); not constant time.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm::crypto::secp256k1 {

// 256-bit number, little-endian 32-bit limbs.
struct U256 {
  uint32_t w[8] = {};
  bool operator==(const U256& o) const;
  bool operator!=(const U256& o) const { return !(*this == o); }
};

U256 from_be(const uint8_t b[32]);
void to_be(const U256& x, uint8_t b[32]);

extern const U256 kP;  // field prime 2^256 - 2^32 - 977
extern const U256 kN;  // group order

// Field (mod p) and scalar (mod n) arithmetic on reduced values.
U256 fe_add(const U256& a, const U256& b);
U256 fe_sub(const U256& a, const U256& b);
U256 fe_mul(const U256& a, const U256& b);
U256 fe_pow(const U256& a, const U256& e);
U256 fe_inv(const U256& a);
bool fe_is_qr(const U256& a);  // a is a nonzero quadratic residue (Legendre symbol 1)
U256 sc_add(const U256& a, const U256& b);
U256 sc_mul(const U256& a, const U256& b);
U256 sc_reduce(const U256& a);  // a mod n for any 256-bit a

struct Point {
  U256 x, y;  // affine
  bool inf = true;
};
// k * G (any k; 0 gives the point at infinity).
Point mul_g(const U256& k);
const Point& generator();
// Fixed-base comb table for GPU scalar multiplication: entry [i * half + j - 1] = j * 2^(w i) * G for window
// i < windows and 1 <= j <= half = 2^(w - 1), affine (one batched inversion).
std::vector<Point> comb_table(int w, int windows);

// RFC6979 nonce for (key32, msg32) with libsecp256k1's seed key32 | msg32 | algo16 (no extra data); counter > 0
// gives the retries.
std::array<uint8_t, 32> rfc6979(const uint8_t key32[32], const uint8_t msg32[32], const uint8_t algo16[16],
                                unsigned counter = 0);
std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* msg, size_t msg_len);

// BCH Schnorr signature R.x | s (32 + 32 bytes, big-endian). False if the key is 0 or >= n.
bool schnorr_sign_bch(const uint8_t msg32[32], const uint8_t seckey32[32], uint8_t sig64[64]);

}  // namespace xm::crypto::secp256k1
