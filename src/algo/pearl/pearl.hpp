// Pearl (PRL) proof-of-useful-work: CPU reference implementation.
//
// Ported from the official Pearl sources (github.com/pearl-research-labs/pearl, ISC licence), commit 82e6eee:
//   zk-pow/src/api/proof_utils.rs   header/config/pattern encoding, nbits
//   zk-pow/src/api/seed.rs          V3 salted seed chain
//   zk-pow/src/circuit/pearl_noise.rs   noise generation
//   zk-pow/src/circuit/chip/jackpot/helper.rs   jackpot transcript
//   zk-pow/src/api/sanity_checks.rs   difficulty bound (incl. rank penalty)
//   pearl-blake3/src/merkle.rs, zk-pow/src/ffi/plain_proof.rs   Merkle multiproof + PlainProof wire format
// This is the correctness oracle for the GPU kernels; it favours clarity over speed.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "crypto/blake3.hpp"

namespace xm::pearl {

using Hash = blake3::Hash;
constexpr int kJackpotSize = 16;
constexpr int kRotPerTile = 13;

// A 3-D arithmetic progression of indices {a*s0 + b*s1 + c*s2 : a<l0, b<l1, c<l2}.
struct Pattern {
  std::array<std::pair<uint32_t, uint32_t>, 3> shape{};  // (stride, length)

  static Pattern contiguous(uint32_t n);                    // [0, 1, ..., n-1]
  static bool from_list(const std::vector<uint32_t>& idx, Pattern& out);
  static bool from_bytes(const uint8_t b[6], Pattern& out);
  std::array<uint8_t, 6> to_bytes() const;
  std::vector<uint32_t> to_list() const;
  uint32_t period() const { return shape[2].first * shape[2].second; }
  uint32_t size() const { return shape[0].second * shape[1].second * shape[2].second; }
  uint32_t max() const;
  bool offset_is_valid(uint32_t offset) const;
  bool operator==(const Pattern& o) const { return shape == o.shape; }
};

struct Config {
  uint32_t k = 4096;
  uint16_t rank = 128;
  Pattern rows = Pattern::contiguous(16);
  Pattern cols = Pattern::contiguous(16);
  std::array<uint8_t, 52> to_bytes() const;
  uint32_t dot_product_length() const { return k - k % rank; }
  uint32_t tile_size() const { return rows.size() * cols.size(); }
};

// 76-byte incomplete block header exactly as pools send it (wire order).
using Header = std::array<uint8_t, 76>;
uint32_t header_nbits(const Header& h);

Hash job_key(const Header& h, const Config& c);
// Keyed BLAKE3 of row-major int8 bytes, zero-padded to a multiple of 1024 (matrix commitment / Merkle root).
Hash matrix_root(const Hash& job_key, const int8_t* data, size_t rows, size_t cols);

struct Seeds {
  Hash b_noise_seed, a_noise_seed;
};
// salted = certificate V3 (current mainnet): roots are bound to m and n before the seed chain.
Seeds commitment_seeds(const Hash& job_key, const Hash& hash_a, const Hash& hash_b, uint32_t m, uint32_t n,
                       bool salted);
Hash seed_salt_a();  // blake3("pearl/cert-v3/noise-seed/A")
Hash seed_salt_b();

// Noise. "A" uses (a_noise_seed, label "A_tensor"), "B" uses (b_noise_seed, label "B_tensor").
enum class Side { A, B };
// One row of the uniform matrix E_AL (side A) or E_BR^T (side B): r values in [-32, 31].
std::vector<int8_t> uniform_row(Side side, const Hash& noise_seed, size_t row, uint32_t rank);
// The k sparse +1/-1 index pairs of E_AR^T (side A) or E_BL (side B).
std::vector<std::array<uint32_t, 2>> permutation(Side side, const Hash& noise_seed, uint32_t k, uint32_t rank);
// Full noise row for A (or column for B): E[l] = u[p0[l]] - u[p1[l]], in [-63, 63].
std::vector<int8_t> noise_line(const std::vector<int8_t>& uniform, const std::vector<std::array<uint32_t, 2>>& perm);

// Jackpot transcript of one tile from already-noised rows of A (h x k) and columns of B (w x k).
std::array<uint32_t, 16> jackpot(const std::vector<std::vector<int32_t>>& a_rows,
                                 const std::vector<std::vector<int32_t>>& b_cols, uint32_t k, uint32_t rank);
Hash jackpot_hash(const std::array<uint32_t, 16>& transcript, const Hash& a_noise_seed);

// 256-bit unsigned integers as little-endian 32-bit limbs (difficulty math).
struct U256 {
  std::array<uint32_t, 8> w{};
  static U256 from_le_bytes(const uint8_t b[32]);
  static U256 from_be_hex(const std::string& hex);  // pool targets are big-endian hex
  static U256 max();
  bool operator<=(const U256& o) const;
  bool operator==(const U256& o) const { return w == o.w; }
  bool is_zero() const;
  // this * f, saturating at 2^256-1; sets *overflow when it saturates.
  U256 mul_sat(uint64_t f, bool* overflow = nullptr) const;
  std::string hex_be() const;
};
U256 nbits_to_target(uint32_t nbits);
uint32_t target_to_nbits(const U256& target);  // compact encoding, rounding the target down
// The share/block bound: min(T*h*w*dpl, T*h*w*(dpl/r)*128), saturating (consensus path).
U256 difficulty_bound(uint32_t nbits, const Config& c);

// ---- Shares (PlainProof) ----
struct MerkleProof {
  std::vector<std::array<uint8_t, 1024>> leaf_data;
  std::vector<uint64_t> leaf_indices;
  uint64_t total_leaves = 0;
  Hash root{};
  std::vector<Hash> siblings;
};
// BLAKE3 tree of a padded matrix, kept so shares (multiproofs) and incremental updates are cheap.
// layers[0] = chunk chaining values, ..., layers.back() = {root}. Pairs are combined, an odd node is carried up.
struct MerkleTree {
  Hash key{};
  std::vector<std::vector<Hash>> layers;
  const Hash& root() const { return layers.back()[0]; }
  size_t leaves() const { return layers[0].size(); }
};
MerkleTree tree_from_chunk_cvs(const Hash& key, std::vector<Hash> chunk_cvs);   // >= 2 chunks
// Replace chunk `index` (1024 bytes) and update the path to the root.
void tree_update_chunk(MerkleTree& t, uint64_t index, const uint8_t* chunk1024);
// Multiproof for `row_indices` of a row-major int8 matrix with `cols` columns, from its tree and data.
MerkleProof tree_proof(const MerkleTree& t, const int8_t* data, size_t len, size_t cols,
                       const std::vector<uint64_t>& row_indices);

// Builds the multiproof for the given rows of a row-major int8 matrix (rows x cols), keyed with job_key.
MerkleProof matrix_proof(const Hash& job_key, const int8_t* data, size_t rows, size_t cols,
                         const std::vector<uint64_t>& row_indices);

struct PlainProof {
  uint64_t m = 0, n = 0, k = 0, noise_rank = 0;
  MerkleProof a, bt;
  std::vector<uint64_t> a_rows, b_cols;
};
std::vector<uint8_t> serialize(const PlainProof& p);  // bincode 1.x fixint little-endian
std::string base64(const std::vector<uint8_t>& bytes);

// ---- End-to-end CPU miner (slow; tests and --selftest only) ----
struct Work {
  Header header{};
  Config config;
  uint32_t m = 0, n = 0;
  std::vector<int8_t> a;   // m x k row-major (signal, values in [-64, 64])
  std::vector<int8_t> bt;  // n x k row-major, i.e. B transposed
  bool salted = true;
};
struct Found {
  uint32_t t_rows = 0, t_cols = 0;
  std::array<uint32_t, 16> transcript{};
  Hash hash{};
};
// Scans every tile of the pattern partition; returns the tiles whose jackpot hash is <= the bound for nbits.
std::vector<Found> cpu_mine(const Work& w, uint32_t nbits, size_t max_found = 1);
PlainProof build_proof(const Work& w, const Found& f);

}  // namespace xm::pearl
