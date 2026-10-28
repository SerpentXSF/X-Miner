// KawPow CPU reference. Ported from cpp-kawpow (Copyright 2018-2019 Pawel Bylica, Apache-2.0): lib/ethash/ethash.cpp,
// progpow.cpp, primes.c and lib/keccak; Keccak-f written in loop form.
#include "algo/kawpow/kawpow.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace xm::kawpow {

namespace {
// KawPow: "RAVENCOINKAWPOW" in the first hash, its first 9 letters in the final one. FiroPow: Keccak-style padding
// (a 1 after the data, 0x80008081 at the end; firoorg/firo progpow.cpp).
const uint32_t kRavenInit[15] = {0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57};
const uint32_t kRavenFinal[9] = {0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E};
const uint32_t kFiroInit[15] = {0x00000001, 0, 0, 0, 0, 0, 0, 0, 0x80008081, 0, 0, 0, 0, 0, 0};
const uint32_t kFiroFinal[9] = {0, 0x00000001, 0, 0, 0, 0, 0, 0, 0x80008081};
}  // namespace

//                               name         epoch  period cache math parents init        max  terminal v092   pads
const Variant kKawpow = {"kawpow", 7500, 3, 11, 18, 512, 1ull << 30, -1, -1, false, kRavenInit, kRavenFinal, false};
const Variant kFiropow = {"firopow", 1300, 1, 11, 18, 512, (1ull << 30) + (1ull << 29), 926, 650, false, kFiroInit,
                          kFiroFinal, false};
const Variant kProgpow092 = {"progpow", 30000, 50, 12, 20, 256, 1ull << 30, -1, -1, true, nullptr, nullptr, false};
const Variant kProgpowZ = {"progpowz", 30000, 50, 12, 20, 256, 1ull << 30, -1, -1, true, nullptr, nullptr, true};

namespace {

constexpr uint64_t kRc[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull,
    0x000000000000808Bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008Aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000Aull,
    0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};
constexpr int kRotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
constexpr int kPiln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

template <class W, int kBits, int kRounds>
void keccakf(W st[25]) {
  auto rol = [](W x, int s) -> W {
    s %= kBits;
    return s ? static_cast<W>((x << s) | (x >> (kBits - s))) : x;
  };
  for (int r = 0; r < kRounds; ++r) {
    W bc[5];
    for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    for (int i = 0; i < 5; ++i) {
      W t = bc[(i + 4) % 5] ^ rol(bc[(i + 1) % 5], 1);
      for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
    }
    W t = st[1];
    for (int i = 0; i < 24; ++i) {
      int j = kPiln[i];
      W b = st[j];
      st[j] = rol(t, kRotc[i]);
      t = b;
    }
    for (int j = 0; j < 25; j += 5) {
      for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
      for (int i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    st[0] ^= static_cast<W>(kRc[r]);
  }
}

template <size_t kOut>
std::array<uint8_t, kOut> keccak(const uint8_t* data, size_t n) {
  constexpr size_t rate = 200 - 2 * kOut;
  uint64_t st[25] = {};
  uint8_t* sb = reinterpret_cast<uint8_t*>(st);
  while (n >= rate) {
    for (size_t i = 0; i < rate; ++i) sb[i] ^= data[i];
    keccakf1600(st);
    data += rate;
    n -= rate;
  }
  for (size_t i = 0; i < n; ++i) sb[i] ^= data[i];
  sb[n] ^= 0x01;
  sb[rate - 1] ^= 0x80;
  keccakf1600(st);
  std::array<uint8_t, kOut> out;
  std::memcpy(out.data(), sb, kOut);
  return out;
}

int largest_prime(int n) {
  if (n < 2) return 0;
  if (n == 2) return 2;
  if (n % 2 == 0) --n;
  auto odd_prime = [](int x) {
    for (int d = 3; int64_t(d) * d <= x; d += 2)
      if (x % d == 0) return false;
    return true;
  };
  while (!odd_prime(n)) n -= 2;
  return n;
}

uint32_t rotl32(uint32_t x, uint32_t n) { return (x << (n % 32)) | (x >> ((32 - n % 32) % 32)); }
uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> (n % 32)) | (x << ((32 - n % 32) % 32)); }
uint32_t clz32(uint32_t x) {
  uint32_t n = 0;
  if (!x) return 32;
  while (!(x & 0x80000000u)) {
    x <<= 1;
    ++n;
  }
  return n;
}
uint32_t popcount32(uint32_t x) {
  uint32_t n = 0;
  for (; x; x &= x - 1) ++n;
  return n;
}

// ProgPoW RNG state of one period: KISS99 plus Fisher-Yates permutations of the destination and source registers.
struct MixRng {
  Kiss99 rng{0, 0, 0, 0};
  int dst_seq[kRegs], src_seq[kRegs];
  int dst_n = 0, src_n = 0;
  explicit MixRng(uint64_t period) {
    const uint32_t lo = static_cast<uint32_t>(period), hi = static_cast<uint32_t>(period >> 32);
    const uint32_t z = fnv1a(kFnvOffset, lo), w = fnv1a(z, hi), jsr = fnv1a(w, lo), jcong = fnv1a(jsr, hi);
    rng = Kiss99{z, w, jsr, jcong};
    for (int i = 0; i < kRegs; ++i) dst_seq[i] = src_seq[i] = i;
    for (int i = kRegs; i > 1; --i) {
      std::swap(dst_seq[i - 1], dst_seq[rng() % i]);
      std::swap(src_seq[i - 1], src_seq[rng() % i]);
    }
  }
  int next_dst() { return dst_seq[(dst_n++) % kRegs]; }
  int next_src() { return src_seq[(src_n++) % kRegs]; }
};

// Item `index` of the 2048-bit view of the dataset: four consecutive 512-bit items.
void dataset_item_2048(const Light& light, uint32_t index, uint32_t out[64]) {
  for (int k = 0; k < 4; ++k) dataset_item_512(light, uint64_t(index) * 4 + k, out + 16 * k);
}

}  // namespace

void keccakf1600(uint64_t st[25]) { keccakf<uint64_t, 64, 24>(st); }
void keccakf800(uint32_t st[25]) { keccakf<uint32_t, 32, 22>(st); }
std::array<uint8_t, 32> keccak256(const uint8_t* data, size_t n) { return keccak<32>(data, n); }
std::array<uint8_t, 64> keccak512(const uint8_t* data, size_t n) { return keccak<64>(data, n); }

// In items, as the reference: (2^30 + epoch * 2^23) bytes overflows an int from epoch 256 (Ravencoin is past 600).
int light_cache_items(int epoch) { return largest_prime((1 << 24) / 64 + epoch * ((1 << 17) / 64)); }
int dataset_items_1024(int epoch, uint64_t init) {
  return largest_prime(int(init / 128) + epoch * ((1 << 23) / 128));
}

int epoch_for_height(const Variant& v, int height) {
  const int e = height / v.epoch_length;
  return v.max_epoch >= 0 && e > v.max_epoch ? v.terminal_epoch : e;
}

Hash256 epoch_seed(int epoch) {
  std::array<uint8_t, 32> s{};
  for (int i = 0; i < epoch; ++i) s = keccak256(s.data(), 32);
  Hash256 h;
  std::memcpy(h.data(), s.data(), 32);
  return h;
}

int epoch_from_seed(const Hash256& seed) {
  std::array<uint8_t, 32> s{};
  for (int e = 0; e < 4096; ++e) {
    if (std::memcmp(s.data(), seed.data(), 32) == 0) return e;
    s = keccak256(s.data(), 32);
  }
  return -1;
}

std::shared_ptr<const Light> make_light(int epoch, int parents, uint64_t dataset_init, int seed_epoch) {
  auto L = std::make_shared<Light>();
  L->epoch = epoch;
  L->parents = parents;
  L->cache_items = light_cache_items(epoch);
  L->dataset_items = dataset_items_1024(epoch, dataset_init);
  const int n = L->cache_items;
  L->cache.resize(size_t(n) * 16);
  auto item = [&](int i) { return reinterpret_cast<uint8_t*>(&L->cache[size_t(i) * 16]); };
  const Hash256 seed = epoch_seed(seed_epoch < 0 ? epoch : seed_epoch);
  auto h = keccak512(reinterpret_cast<const uint8_t*>(seed.data()), 32);
  std::memcpy(item(0), h.data(), 64);
  for (int i = 1; i < n; ++i) {
    h = keccak512(item(i - 1), 64);
    std::memcpy(item(i), h.data(), 64);
  }
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < n; ++i) {
      const uint32_t v = L->cache[size_t(i) * 16] % uint32_t(n);
      const uint32_t w = uint32_t(n + (i - 1)) % uint32_t(n);
      uint8_t x[64];
      for (int k = 0; k < 64; ++k) x[k] = item(int(v))[k] ^ item(int(w))[k];
      h = keccak512(x, 64);
      std::memcpy(item(i), h.data(), 64);
    }
  }
  L->l1.resize(kCacheWords);
  for (int i = 0; i < kCacheWords / 64; ++i) dataset_item_2048(*L, uint32_t(i), &L->l1[size_t(i) * 64]);
  return L;
}

void dataset_item_512(const Light& light, uint64_t index, uint32_t out[16]) {
  const uint32_t n = uint32_t(light.cache_items);
  uint32_t mix[16];
  std::memcpy(mix, &light.cache[size_t(index % n) * 16], 64);
  mix[0] ^= static_cast<uint32_t>(index);
  auto h = keccak512(reinterpret_cast<const uint8_t*>(mix), 64);
  std::memcpy(mix, h.data(), 64);
  const uint32_t seed = static_cast<uint32_t>(index);
  for (uint32_t j = 0; j < uint32_t(light.parents); ++j) {
    const uint32_t parent = fnv1(seed ^ j, mix[j % 16]) % n;
    const uint32_t* p = &light.cache[size_t(parent) * 16];
    for (int k = 0; k < 16; ++k) mix[k] = fnv1(mix[k], p[k]);
  }
  h = keccak512(reinterpret_cast<const uint8_t*>(mix), 64);
  std::memcpy(out, h.data(), 64);
}

Program make_program(uint64_t period, const Variant& v) {
  MixRng st(period);
  Program p;
  const int max_ops = v.cnt_cache > v.cnt_math ? v.cnt_cache : v.cnt_math;
  for (int i = 0; i < max_ops; ++i) {
    if (i < v.cnt_cache) {
      Program::Op op{};
      op.cache = true;
      op.c.src = st.next_src();
      op.c.dst = st.next_dst();
      op.c.sel = st.rng();
      p.ops.push_back(op);
    }
    if (i < v.cnt_math) {
      Program::Op op{};
      op.cache = false;
      const uint32_t src_rnd = st.rng() % (kRegs * (kRegs - 1));
      op.m.src1 = int(src_rnd % kRegs);
      op.m.src2 = int(src_rnd / kRegs);
      if (op.m.src2 >= op.m.src1) ++op.m.src2;
      op.m.sel1 = st.rng();
      op.m.dst = st.next_dst();
      op.m.sel2 = st.rng();
      p.ops.push_back(op);
    }
  }
  for (int i = 0; i < kDagLoads; ++i) {
    p.dag_dst[i] = i == 0 ? 0 : st.next_dst();
    p.dag_sel[i] = st.rng();
  }
  return p;
}

int math_op(uint32_t sel, const Variant& v) {
  const int s = int(sel % 11);
  return v.zano_math ? (s + 9) % 11 : s;  // ProgPowZ: 0 clz, 1 popc, 2 add, 3 mul, ..., 10 xor
}

uint32_t random_math(uint32_t a, uint32_t b, int op) {
  switch (op) {
    default:
    case 0: return a + b;
    case 1: return a * b;
    case 2: return static_cast<uint32_t>((uint64_t(a) * b) >> 32);
    case 3: return std::min(a, b);
    case 4: return rotl32(a, b);
    case 5: return rotr32(a, b);
    case 6: return a & b;
    case 7: return a | b;
    case 8: return a ^ b;
    case 9: return clz32(a) + clz32(b);
    case 10: return popcount32(a) + popcount32(b);
  }
}

void random_merge(uint32_t& a, uint32_t b, uint32_t sel) {
  const uint32_t x = (sel >> 16) % 31 + 1;
  switch (sel % 4) {
    case 0: a = a * 33 + b; break;
    case 1: a = (a ^ b) * 33; break;
    case 2: a = rotl32(a, x) ^ b; break;
    case 3: a = rotr32(a, x) ^ b; break;
  }
}

namespace {
uint32_t bswap32(uint32_t x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
}  // namespace

Result hash(const Light& light, int height, const Hash256& header_hash, uint64_t nonce, const Variant& v) {
  // First Keccak-f800: header, nonce, then the variant's padding (0.9.2: zeros).
  uint32_t st[25] = {};
  for (int i = 0; i < 8; ++i) st[i] = header_hash[i];
  st[8] = static_cast<uint32_t>(nonce);
  st[9] = static_cast<uint32_t>(nonce >> 32);
  if (v.pad_init)
    for (int i = 10; i < 25; ++i) st[i] = v.pad_init[i - 10];
  keccakf800(st);
  uint32_t state2[8];
  for (int i = 0; i < 8; ++i) state2[i] = st[i];
  // The two words that seed the per-lane KISS99: 0.9.4 the first two state words, 0.9.2 the 64-bit seed
  // bswap(st0) << 32 | bswap(st1), low word first.
  const uint32_t s_lo = v.v092 ? bswap32(state2[1]) : state2[0], s_hi = v.v092 ? bswap32(state2[0]) : state2[1];
  uint32_t mix[kLanes][kRegs];
  const uint32_t z = fnv1a(kFnvOffset, s_lo), w = fnv1a(z, s_hi);
  for (uint32_t l = 0; l < uint32_t(kLanes); ++l) {
    const uint32_t jsr = fnv1a(w, l), jcong = fnv1a(jsr, l);
    Kiss99 rng{z, w, jsr, jcong};
    for (int r = 0; r < kRegs; ++r) mix[l][r] = rng();
  }
  const Program prog = make_program(uint64_t(height / v.period), v);
  const uint32_t num_items = uint32_t(light.dataset_items / 2);  // 2048-bit items
  uint32_t item[64];
  for (int r = 0; r < kCntDag; ++r) {
    dataset_item_2048(light, mix[r % kLanes][0] % num_items, item);
    for (const auto& op : prog.ops) {
      for (int l = 0; l < kLanes; ++l) {
        if (op.cache) {
          random_merge(mix[l][op.c.dst], light.l1[mix[l][op.c.src] % kCacheWords], op.c.sel);
        } else {
          random_merge(mix[l][op.m.dst], random_math(mix[l][op.m.src1], mix[l][op.m.src2], math_op(op.m.sel1, v)),
                       op.m.sel2);
        }
      }
    }
    for (int l = 0; l < kLanes; ++l) {
      const int off = ((l ^ r) % kLanes) * kDagLoads;
      for (int i = 0; i < kDagLoads; ++i) random_merge(mix[l][prog.dag_dst[i]], item[off + i], prog.dag_sel[i]);
    }
  }
  uint32_t lane_hash[kLanes];
  for (int l = 0; l < kLanes; ++l) {
    lane_hash[l] = kFnvOffset;
    for (int r = 0; r < kRegs; ++r) lane_hash[l] = fnv1a(lane_hash[l], mix[l][r]);
  }
  Result res;
  res.mix_hash.fill(kFnvOffset);
  for (int l = 0; l < kLanes; ++l) res.mix_hash[l % 8] = fnv1a(res.mix_hash[l % 8], lane_hash[l]);
  // Final Keccak-f800. 0.9.4: the first hash's 8 words, the mix, the padding. 0.9.2: header, seed, mix, zeros.
  uint32_t fs[25] = {};
  if (v.v092) {
    for (int i = 0; i < 8; ++i) fs[i] = header_hash[i];
    fs[8] = s_lo;
    fs[9] = s_hi;
    for (int i = 0; i < 8; ++i) fs[10 + i] = res.mix_hash[i];
  } else {
    for (int i = 0; i < 8; ++i) fs[i] = state2[i];
    for (int i = 0; i < 8; ++i) fs[8 + i] = res.mix_hash[i];
    for (int i = 16; i < 25; ++i) fs[i] = v.pad_final[i - 16];
  }
  keccakf800(fs);
  for (int i = 0; i < 8; ++i) res.final_hash[i] = fs[i];
  return res;
}

bool less_or_equal(const Hash256& a, const Hash256& b) {
  const uint8_t* pa = reinterpret_cast<const uint8_t*>(a.data());
  const uint8_t* pb = reinterpret_cast<const uint8_t*>(b.data());
  for (int i = 0; i < 32; ++i)
    if (pa[i] != pb[i]) return pa[i] < pb[i];
  return true;
}

Hash256 hash_from_hex(const std::string& hex) {
  Hash256 h{};
  uint8_t* p = reinterpret_cast<uint8_t*>(h.data());
  for (size_t i = 0; i < 32 && 2 * i + 1 < hex.size(); ++i) p[i] = uint8_t(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
  return h;
}

std::string hash_to_hex(const Hash256& h) {
  static const char* d = "0123456789abcdef";
  const uint8_t* p = reinterpret_cast<const uint8_t*>(h.data());
  std::string s;
  for (int i = 0; i < 32; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

}  // namespace xm::kawpow
