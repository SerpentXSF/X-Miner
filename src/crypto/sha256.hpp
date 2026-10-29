// SHA-256 (FIPS 180-4): one-shot hash, double hash and the raw compression (for midstates).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace xm::crypto {

using Sha256Digest = std::array<uint8_t, 32>;

extern const uint32_t kSha256Init[8];
extern const uint32_t kSha256K[64];

// One 64-byte block into the state (big-endian message words).
void sha256_compress(uint32_t state[8], const uint8_t block[64]);
Sha256Digest sha256(const uint8_t* data, size_t len);
Sha256Digest sha256d(const uint8_t* data, size_t len);

}  // namespace xm::crypto
