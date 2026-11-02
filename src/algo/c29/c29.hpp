// Tari Cuckaroo29 (XTM "c29") CPU reference, written from the Tari node source (tari-project/tari, BSD-3:
// base_layer/core/src/proof_of_work/cuckaroo_pow.rs and siphash.rs):
//   keys   = blake2b-256(nonce as 8 big-endian bytes | mining hash (32)), read as four little-endian u64
//   edge i = siphash-2-4 "block" hash (blocks of 64 consecutive indices, each xored with the block's last hash,
//            rotation 21 = Grin's original Cuckaroo29), u = bits 0..28, v = bits 32..60
//   proof  = 42 ascending edge indices < 2^29 forming one cycle in the bipartite graph (u and v sides apart)
//   result = blake2b-256(the indices packed as 29-bit little-endian fields, 153 bytes); share difficulty
//            = (2^256 - 1) / result read big-endian.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm::c29 {

constexpr int kEdgeBits = 29;
constexpr uint64_t kEdges = 1ull << kEdgeBits;
constexpr uint64_t kNodeMask = kEdges - 1;
constexpr int kProofSize = 42;
constexpr size_t kPackedBytes = (kProofSize * kEdgeBits + 7) / 8;  // 153

using Keys = std::array<uint64_t, 4>;
using Hash = std::array<uint8_t, 32>;
using Proof = std::array<uint32_t, kProofSize>;

Keys sip_keys(const Hash& mining_hash, uint64_t nonce);
uint64_t siphash24(const Keys& k, uint64_t nonce, int rot = 21);
// The block hash of edge index `edge` (the whole 64-edge block is computed).
uint64_t siphash_block(const Keys& k, uint64_t edge, int rot = 21);
// Endpoints of an edge: u, v < 2^29.
void edge_nodes(const Keys& k, uint32_t edge, uint32_t& u, uint32_t& v);

// The proof forms a bipartite 42-cycle (indices ascending and < 2^29).
bool verify(const Keys& k, const Proof& p);
std::array<uint8_t, kPackedBytes> pack(const Proof& p);
Hash result(const Proof& p);
// (2^256 - 1) / result (big-endian), capped at 2^64 - 1.
uint64_t share_difficulty(const Hash& result);
// Pool target (16 hex = 8 bytes, little-endian as XMRig pools send it) -> difficulty (2^64 - 1) / target; 0 on bad
// input. Kryptex's "cccccccccccccc0c" is difficulty 20: read big-endian (difficulty 1) its shares are "Invalid share".
uint64_t target_difficulty(const char* hex16);

// Cycle finding on a trimmed edge list: (u, v) pairs on separate sides. Returns every 42-cycle as indices into the
// list (sorted by index into the list).
std::vector<std::array<uint32_t, kProofSize>> find_cycles(const std::vector<std::pair<uint32_t, uint32_t>>& edges);

}  // namespace xm::c29
