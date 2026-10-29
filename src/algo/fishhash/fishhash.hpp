// FishHash (Iron Fish) and KarlsenHashV2 (Karlsen, "FishHashPlus") CPU references, written from iron-fish/fish-hash
// (rust/src/lib.rs) and rusty-karlsen (crypto/hashes/src/pow_hashers.rs, consensus/pow/src/lib.rs).
// One fixed dataset for both (no epochs): the Ethash light cache / item construction with a fixed seed, 1,179,641
// cache items and 37,748,717 items of 1024 bits (two 512-parent halves), 4.5 GB. A hash is 32 rounds of three
// 128-byte dataset reads mixed with 64-bit multiply-adds, between BLAKE3 hashes.
#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "algo/kawpow/kawpow.hpp"

namespace xm::fishhash {

using kawpow::Light;
using Bytes32 = std::array<uint8_t, 32>;

constexpr int kCacheItems = 1179641;
constexpr int kDatasetItems = 37748717;  // 1024-bit items
constexpr int kParents = 512;
constexpr int kRounds = 32;

std::shared_ptr<const Light> make_light();  // ~72 MB, a few seconds

enum class Variant { Iron, Karlsen };

// The 32-byte mix digest of a 64-byte seed (FishHash kernel or Karlsen's FishHashPlus index selection).
Bytes32 kernel(const Light& light, const uint8_t seed[64], Variant v);

// Iron Fish: BLAKE3(seed || kernel(seed)), seed = BLAKE3-XOF(header, 64); the share rule is hash <= target, both
// big-endian. The miner's nonce is header bytes 172..179, big-endian.
Bytes32 iron_hash(const Light& light, const uint8_t* header, size_t len);

// Karlsen: h1 = BLAKE3(pre_pow || time LE || 32 zero bytes || nonce LE), h3 = BLAKE3(kernel(h1 || 32 zero bytes)); the
// share rule is h3 <= target, both read as little-endian 256-bit numbers.
Bytes32 karlsen_h1(const Bytes32& pre_pow, uint64_t timestamp, uint64_t nonce);
Bytes32 karlsen_hash(const Light& light, const Bytes32& pre_pow, uint64_t timestamp, uint64_t nonce);

bool be_less_equal(const Bytes32& a, const Bytes32& b);  // big-endian a <= b
bool le_less_equal(const Bytes32& a, const Bytes32& b);  // little-endian a <= b

}  // namespace xm::fishhash
