// KawPow (Ravencoin's ProgPoW 0.9.4 variant) CPU reference: Keccak, the Ethash light cache and dataset items,
// the ProgPoW program and hash. Used to re-check every GPU share, by --selftest and by the unit tests.
//
// Ported from cpp-kawpow (https://github.com/RavenCommunity/cpp-kawpow @061d341, Copyright 2018-2019 Pawel Bylica,
// Apache-2.0); see THIRD_PARTY.md and NOTICE. Hashes are little-endian 32-bit words as in the reference.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xm::kawpow {

using Hash256 = std::array<uint32_t, 8>;

// ProgPoW / KawPow parameters.
constexpr int kEpochLength = 7500;        // blocks per DAG epoch (Ethash uses 30000)
constexpr int kPeriodLength = 3;          // blocks per random program
constexpr int kRegs = 32;                 // mix registers per lane
constexpr int kLanes = 16;                // lanes per hash
constexpr int kDagLoads = 4;              // 32-bit words per lane per DAG access
constexpr int kCacheBytes = 16 * 1024;    // the DAG's first 16 KB, held in fast memory
constexpr int kCacheWords = kCacheBytes / 4;
constexpr int kCntDag = 64;               // DAG accesses (rounds) per hash
constexpr int kCntCache = 11;             // cache reads per round
constexpr int kCntMath = 18;              // random math ops per round
constexpr int kDatasetParents = 512;      // light-cache parents per 512-bit dataset item (KawPow; Ethash 256)

constexpr uint32_t kFnvPrime = 0x01000193;
constexpr uint32_t kFnvOffset = 0x811c9dc5;
inline uint32_t fnv1(uint32_t u, uint32_t v) { return (u * kFnvPrime) ^ v; }
inline uint32_t fnv1a(uint32_t u, uint32_t v) { return (u ^ v) * kFnvPrime; }

// "RAVENCOINKAWPOW", the padding KawPow puts in the Keccak-f800 state.
extern const uint32_t kRavencoinKawpow[15];

void keccakf1600(uint64_t st[25]);
void keccakf800(uint32_t st[25]);
std::array<uint8_t, 32> keccak256(const uint8_t* data, size_t n);  // original Keccak padding (0x01)
std::array<uint8_t, 64> keccak512(const uint8_t* data, size_t n);

class Kiss99 {
 public:
  Kiss99(uint32_t z, uint32_t w, uint32_t jsr, uint32_t jcong) : z_(z), w_(w), jsr_(jsr), jcong_(jcong) {}
  uint32_t operator()() {
    z_ = 36969 * (z_ & 0xffff) + (z_ >> 16);
    w_ = 18000 * (w_ & 0xffff) + (w_ >> 16);
    jcong_ = 69069 * jcong_ + 1234567;
    jsr_ ^= jsr_ << 17;
    jsr_ ^= jsr_ >> 13;
    jsr_ ^= jsr_ << 5;
    return (((z_ << 16) + w_) ^ jcong_) + jsr_;
  }

 private:
  uint32_t z_, w_, jsr_, jcong_;
};

int light_cache_items(int epoch);    // 512-bit items
int dataset_items_1024(int epoch);   // 1024-bit items (the DAG is this many x 128 bytes)
Hash256 epoch_seed(int epoch);       // keccak256 applied `epoch` times to zero
int epoch_from_seed(const Hash256& seed);  // -1 if not one of the first 4096 epochs

// One epoch's light cache plus the 16 KB L1 cache (the DAG's first 16 KB).
struct Light {
  int epoch = 0;
  int cache_items = 0;
  int dataset_items = 0;           // 1024-bit items
  std::vector<uint32_t> cache;     // cache_items x 16 words
  std::vector<uint32_t> l1;        // kCacheWords words
};
std::shared_ptr<const Light> make_light(int epoch);

// Dataset (DAG) item `index` of 512 bits from the light cache (16 words).
void dataset_item_512(const Light& light, uint64_t index, uint32_t out[16]);

// The random program of one period, as the GPU kernel generator and the CPU hash both consume it.
struct Program {
  struct CacheOp { int src, dst; uint32_t sel; };
  struct MathOp { int src1, src2, dst; uint32_t sel1, sel2; };
  struct Op { bool cache; CacheOp c; MathOp m; };
  std::vector<Op> ops;           // per round, in order (interleaved cache / math as the reference)
  int dag_dst[kDagLoads];        // destination register per DAG word (word 0 always register 0)
  uint32_t dag_sel[kDagLoads];
};
Program make_program(uint64_t period);

uint32_t random_math(uint32_t a, uint32_t b, uint32_t sel);
void random_merge(uint32_t& a, uint32_t b, uint32_t sel);

struct Result {
  Hash256 final_hash{};
  Hash256 mix_hash{};
};
// The KawPow hash of (header_hash, nonce) at `height` with DAG items computed from the light cache (slow: a few ms).
Result hash(const Light& light, int height, const Hash256& header_hash, uint64_t nonce);
// The first Keccak-f800 (hash seed and the 8 words carried into the final one).
void initial_keccak(const Hash256& header_hash, uint64_t nonce, uint32_t state2[8]);
// The final hash from the carried words and a mix hash (cheap; checks a share against a target before the mix).
Hash256 final_keccak(const uint32_t state2[8], const Hash256& mix_hash);
// a <= b with both read as 256-bit big-endian numbers (hash bytes in order).
bool less_or_equal(const Hash256& a, const Hash256& b);

Hash256 hash_from_hex(const std::string& hex);  // 64 hex digits, bytes in order
std::string hash_to_hex(const Hash256& h);

}  // namespace xm::kawpow
