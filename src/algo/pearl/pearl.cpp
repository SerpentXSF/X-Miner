#include "algo/pearl/pearl.hpp"

#include <algorithm>
#include <cstring>
#include <set>

namespace xm::pearl {

// ------------------------------------------------------------------------------------------------ patterns

Pattern Pattern::contiguous(uint32_t n) {
  std::vector<uint32_t> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = i;
  Pattern p;
  from_list(v, p);
  return p;
}

bool Pattern::from_bytes(const uint8_t b[6], Pattern& out) {
  // proof_utils.rs PeriodicPattern::from_bytes
  uint32_t min_stride = 1;
  bool done = false;
  Pattern p;
  for (int i = 0; i < 3; ++i) {
    uint32_t factor = 1u + b[2 * i], length = 1u + b[2 * i + 1];
    if (length == 1 || done) {
      if (factor != 1 || length != 1) return false;  // non-canonical
      done = true;
    } else if (factor <= 1 && min_stride != 1) {
      return false;  // a single stride must not be broken
    }
    if (min_stride > (1u << 24) / (factor * length)) return false;  // period <= 2^24
    uint32_t stride = factor * min_stride;
    p.shape[i] = {stride, length};
    min_stride = stride * length;
  }
  if (p.to_bytes() != std::array<uint8_t, 6>{b[0], b[1], b[2], b[3], b[4], b[5]}) return false;
  out = p;
  return true;
}

std::array<uint8_t, 6> Pattern::to_bytes() const {
  std::array<uint8_t, 6> d{};
  uint32_t min_stride = 1;
  for (int i = 0; i < 3; ++i) {
    uint32_t factor = shape[i].first / min_stride;
    d[2 * i] = static_cast<uint8_t>(factor - 1);
    d[2 * i + 1] = static_cast<uint8_t>(shape[i].second - 1);
    min_stride = shape[i].first * shape[i].second;
  }
  return d;
}

std::vector<uint32_t> Pattern::to_list() const {
  std::vector<uint32_t> res{0};
  for (const auto& [stride, length] : shape) {
    std::vector<uint32_t> next;
    next.reserve(res.size() * length);
    for (uint32_t i = 0; i < length; ++i)
      for (uint32_t r : res) next.push_back(r + i * stride);
    res.swap(next);
  }
  return res;
}

uint32_t Pattern::max() const {
  auto l = to_list();
  return *std::max_element(l.begin(), l.end());
}

bool Pattern::from_list(const std::vector<uint32_t>& idx, Pattern& out) {
  // proof_utils.rs PeriodicPattern::from_list (greedy period detection)
  if (idx.empty() || idx[0] != 0) return false;
  for (size_t i = 1; i < idx.size(); ++i)
    if (idx[i] <= idx[i - 1]) return false;
  std::vector<uint32_t> p = idx;
  std::vector<std::pair<uint32_t, uint32_t>> sv;
  while (p.size() > 1) {
    bool found = false;
    for (size_t period = 1; period < p.size(); ++period) {
      if (p.size() % period) continue;
      uint32_t s = p[period];
      bool periodic = true;
      for (size_t i = 0; i + period < p.size(); ++i)
        if (p[i] + s != p[i + period]) {
          periodic = false;
          break;
        }
      if (periodic) {
        sv.push_back({s, static_cast<uint32_t>(p.size() / period)});
        p.resize(period);
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  std::reverse(sv.begin(), sv.end());
  uint32_t period = sv.empty() ? 1 : sv.back().first * sv.back().second;
  while (sv.size() < 3) sv.push_back({period, 1});
  if (sv.size() > 3) return false;
  Pattern r;
  for (int i = 0; i < 3; ++i) r.shape[i] = sv[i];
  Pattern check;
  auto bytes = r.to_bytes();
  if (!from_bytes(bytes.data(), check) || !(check == r)) return false;
  out = r;
  return true;
}

bool Pattern::offset_is_valid(uint32_t offset) const {
  for (int i = 2; i >= 0; --i) {
    offset %= shape[i].first * shape[i].second;
    if (offset >= shape[i].first) return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------ header/config

std::array<uint8_t, 52> Config::to_bytes() const {
  std::array<uint8_t, 52> b{};
  for (int i = 0; i < 4; ++i) b[i] = static_cast<uint8_t>(k >> (8 * i));
  b[4] = static_cast<uint8_t>(rank);
  b[5] = static_cast<uint8_t>(rank >> 8);
  // b[6..8] = mma_type 0 (Int7xInt7ToInt32); b[20..52] = dense trailer (e = 0, top_k = 0, zero padding)
  auto r = rows.to_bytes(), c = cols.to_bytes();
  std::copy(r.begin(), r.end(), b.begin() + 8);
  std::copy(c.begin(), c.end(), b.begin() + 14);
  return b;
}

uint32_t header_nbits(const Header& h) {
  return uint32_t(h[72]) | uint32_t(h[73]) << 8 | uint32_t(h[74]) << 16 | uint32_t(h[75]) << 24;
}

Hash job_key(const Header& h, const Config& c) {
  uint8_t buf[128];
  std::memcpy(buf, h.data(), 76);
  auto cb = c.to_bytes();
  std::memcpy(buf + 76, cb.data(), 52);
  return blake3::hash(buf, sizeof buf);
}

Hash matrix_root(const Hash& key, const int8_t* data, size_t rows, size_t cols) {
  size_t len = rows * cols;
  size_t padded = (len + 1023) / 1024 * 1024;
  std::vector<uint8_t> buf(padded, 0);
  std::memcpy(buf.data(), data, len);
  return blake3::keyed_hash(key.data(), buf.data(), buf.size());
}

Hash seed_salt_a() {
  const char* s = "pearl/cert-v3/noise-seed/A";
  return blake3::hash(reinterpret_cast<const uint8_t*>(s), std::strlen(s));
}
Hash seed_salt_b() {
  const char* s = "pearl/cert-v3/noise-seed/B";
  return blake3::hash(reinterpret_cast<const uint8_t*>(s), std::strlen(s));
}

namespace {

Hash bind_root(const Hash& salt, const Hash& root, uint32_t dim) {
  uint8_t msg[64] = {};
  std::memcpy(msg, root.data(), 32);
  for (int i = 0; i < 4; ++i) msg[32 + i] = static_cast<uint8_t>(dim >> (8 * i));
  return blake3::keyed_hash(salt.data(), msg, 64);
}

Hash hash_concat(const Hash& a, const Hash& b) {
  uint8_t msg[64];
  std::memcpy(msg, a.data(), 32);
  std::memcpy(msg + 32, b.data(), 32);
  return blake3::hash(msg, 64);
}

}  // namespace

Seeds commitment_seeds(const Hash& jk, const Hash& hash_a, const Hash& hash_b, uint32_t m, uint32_t n, bool salted) {
  Hash a = salted ? bind_root(seed_salt_a(), hash_a, m) : hash_a;
  Hash b = salted ? bind_root(seed_salt_b(), hash_b, n) : hash_b;
  Seeds s;
  s.b_noise_seed = hash_concat(jk, b);
  s.a_noise_seed = hash_concat(s.b_noise_seed, a);
  return s;
}

// ------------------------------------------------------------------------------------------------ noise

namespace {

Hash random_hash(size_t index, Side side, const Hash& key, int slot) {
  uint8_t msg[64] = {};
  int32_t v = static_cast<int32_t>(1 + index);
  std::memcpy(msg + slot * 4, &v, 4);  // little-endian hosts only (x86-64, ARM64)
  std::memcpy(msg + 32, side == Side::A ? "A_tensor" : "B_tensor", 8);
  return blake3::keyed_hash(key.data(), msg, 64);
}

}  // namespace

std::vector<int8_t> uniform_row(Side side, const Hash& seed, size_t row, uint32_t rank) {
  std::vector<int8_t> out(rank);
  size_t start = row * rank;
  size_t cur_block = SIZE_MAX;
  Hash h{};
  for (uint32_t j = 0; j < rank; ++j) {
    size_t idx = start + j;
    if (idx / 32 != cur_block) {
      cur_block = idx / 32;
      h = random_hash(cur_block, side, seed, 0);
    }
    out[j] = static_cast<int8_t>(static_cast<int>(h[idx % 32] & 63) - 32);
  }
  return out;
}

std::vector<std::array<uint32_t, 2>> permutation(Side side, const Hash& seed, uint32_t k, uint32_t rank) {
  std::vector<std::array<uint32_t, 2>> res(k);
  uint32_t mask = rank - 1;
  for (uint32_t l = 0; l < k; l += 8) {
    Hash h = random_hash(l / 8, side, seed, 1);
    for (uint32_t j = 0; j < 8 && l + j < k; ++j) {
      uint32_t u = uint32_t(h[4 * j]) | uint32_t(h[4 * j + 1]) << 8 | uint32_t(h[4 * j + 2]) << 16 |
                   uint32_t(h[4 * j + 3]) << 24;
      uint32_t first = u & mask;
      uint32_t mulhi = static_cast<uint32_t>((uint64_t(rank - 1) * u) >> 32);
      res[l + j] = {first, first ^ (1 + mulhi)};
    }
  }
  return res;
}

std::vector<int8_t> noise_line(const std::vector<int8_t>& u, const std::vector<std::array<uint32_t, 2>>& perm) {
  std::vector<int8_t> out(perm.size());
  for (size_t l = 0; l < perm.size(); ++l)
    out[l] = static_cast<int8_t>(int(u[perm[l][0]]) - int(u[perm[l][1]]));
  return out;
}

// ------------------------------------------------------------------------------------------------ jackpot

std::array<uint32_t, 16> jackpot(const std::vector<std::vector<int32_t>>& a_rows,
                                 const std::vector<std::vector<int32_t>>& b_cols, uint32_t k, uint32_t rank) {
  size_t h = a_rows.size(), w = b_cols.size();
  std::vector<int32_t> acc(h * w, 0);
  std::array<uint32_t, 16> t{};
  for (uint32_t ll = rank; ll <= k; ll += rank) {
    for (size_t u = 0; u < h; ++u)
      for (size_t v = 0; v < w; ++v) {
        int32_t s = acc[u * w + v];
        for (uint32_t l = ll - rank; l < ll; ++l)
          s = static_cast<int32_t>(static_cast<uint32_t>(s) +
                                   static_cast<uint32_t>(a_rows[u][l] * b_cols[v][l]));  // wrapping i32
        acc[u * w + v] = s;
      }
    uint32_t x = 0;
    for (int32_t s : acc) x ^= static_cast<uint32_t>(s);
    uint32_t tid = (ll / rank - 1) % kJackpotSize;
    t[tid] = ((t[tid] << kRotPerTile) | (t[tid] >> (32 - kRotPerTile))) ^ x;
  }
  return t;
}

Hash jackpot_hash(const std::array<uint32_t, 16>& t, const Hash& a_noise_seed) {
  uint8_t msg[64];
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 4; ++j) msg[4 * i + j] = static_cast<uint8_t>(t[i] >> (8 * j));
  return blake3::keyed_hash(a_noise_seed.data(), msg, 64);
}

// ------------------------------------------------------------------------------------------------ U256

U256 U256::from_le_bytes(const uint8_t b[32]) {
  U256 r;
  for (int i = 0; i < 8; ++i)
    r.w[i] = uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16 |
             uint32_t(b[4 * i + 3]) << 24;
  return r;
}

U256 U256::from_be_hex(const std::string& hex) {
  U256 r;
  std::string h = hex;
  if (h.rfind("0x", 0) == 0) h = h.substr(2);
  if (h.size() > 64) h = h.substr(h.size() - 64);
  h = std::string(64 - h.size(), '0') + h;
  for (int i = 0; i < 8; ++i) r.w[7 - i] = static_cast<uint32_t>(std::stoul(h.substr(i * 8, 8), nullptr, 16));
  return r;
}

U256 U256::max() {
  U256 r;
  r.w.fill(0xffffffffu);
  return r;
}

bool U256::operator<=(const U256& o) const {
  for (int i = 7; i >= 0; --i)
    if (w[i] != o.w[i]) return w[i] < o.w[i];
  return true;
}

bool U256::is_zero() const {
  for (auto x : w)
    if (x) return false;
  return true;
}

U256 U256::mul_sat(uint64_t f, bool* overflow) const {
  uint32_t fl = static_cast<uint32_t>(f), fh = static_cast<uint32_t>(f >> 32);
  uint64_t r[10] = {};
  for (int i = 0; i < 8; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 2; ++j) {
      uint64_t cur = r[i + j] + uint64_t(w[i]) * (j ? fh : fl) + carry;
      r[i + j] = cur & 0xffffffffu;
      carry = cur >> 32;
    }
    for (int j = i + 2; carry && j < 10; ++j) {
      uint64_t cur = r[j] + carry;
      r[j] = cur & 0xffffffffu;
      carry = cur >> 32;
    }
  }
  if (r[8] || r[9]) {
    if (overflow) *overflow = true;
    return max();
  }
  if (overflow) *overflow = false;
  U256 out;
  for (int i = 0; i < 8; ++i) out.w[i] = static_cast<uint32_t>(r[i]);
  return out;
}

std::string U256::hex_be() const {
  char buf[65];
  for (int i = 0; i < 8; ++i) std::snprintf(buf + 8 * i, 9, "%08x", w[7 - i]);
  return buf;
}

U256 nbits_to_target(uint32_t nbits) {
  uint32_t exponent = nbits >> 24, mantissa = nbits & 0x00ffffff;
  U256 t;
  if (mantissa == 0 || exponent == 0 || (mantissa & 0x00800000)) return t;
  // target = mantissa * 256^(exponent-3)
  int shift_bytes = static_cast<int>(exponent) - 3;
  uint8_t bytes[64] = {};  // little-endian scratch, wide enough for any exponent
  uint8_t m[3] = {static_cast<uint8_t>(mantissa), static_cast<uint8_t>(mantissa >> 8),
                  static_cast<uint8_t>(mantissa >> 16)};
  for (int i = 0; i < 3; ++i) {
    int pos = i + shift_bytes;
    if (pos >= 0 && pos < 64) bytes[pos] = m[i];
  }
  for (int i = 32; i < 64; ++i)
    if (bytes[i]) return U256::max();  // does not fit in 256 bits
  return U256::from_le_bytes(bytes);
}

uint32_t target_to_nbits(const U256& t) {
  int size = 32;
  auto byte_at = [&](int i) { return static_cast<uint8_t>(t.w[i / 4] >> (8 * (i % 4))); };  // LE byte i
  while (size > 0 && byte_at(size - 1) == 0) --size;
  if (size == 0) return 0;
  uint32_t mantissa = 0;
  for (int i = 0; i < 3; ++i) {
    int pos = size - 1 - i;
    mantissa = (mantissa << 8) | (pos >= 0 ? byte_at(pos) : 0);
  }
  if (mantissa & 0x00800000) {
    mantissa >>= 8;
    ++size;
  }
  return static_cast<uint32_t>(size) << 24 | mantissa;
}

U256 difficulty_bound(uint32_t nbits, const Config& c) {
  U256 t = nbits_to_target(nbits);
  uint64_t tile = c.tile_size(), dpl = c.dot_product_length();
  U256 plain = t.mul_sat(tile * dpl);
  U256 penal = t.mul_sat(tile * (dpl / c.rank) * 128);
  return plain <= penal ? plain : penal;
}

// ------------------------------------------------------------------------------------------------ Merkle / proof

namespace {

struct Tree {
  std::vector<std::vector<Hash>> layers;  // layers[0] = chunk CVs; last = {root}
};

Tree build_tree(const Hash& key, const std::vector<uint8_t>& data) {
  Tree t;
  blake3::Cv kw = blake3::key_words(key.data());
  size_t chunks = data.size() / 1024;
  if (chunks <= 1) {
    t.layers.push_back({blake3::keyed_hash(key.data(), data.data(), data.size())});
    return t;
  }
  std::vector<Hash> l0(chunks);
  for (size_t i = 0; i < chunks; ++i)
    l0[i] = blake3::cv_bytes(blake3::chunk_cv(kw, data.data() + i * 1024, 1024, i, blake3::KEYED_HASH));
  t.layers.push_back(l0);
  auto cvw = [](const Hash& h) { return blake3::key_words(h.data()); };  // bytes -> words
  while (t.layers.back().size() > 2) {
    const auto& prev = t.layers.back();
    std::vector<Hash> next;
    for (size_t i = 0; i + 1 < prev.size(); i += 2)
      next.push_back(blake3::cv_bytes(blake3::parent_cv(kw, cvw(prev[i]), cvw(prev[i + 1]), blake3::KEYED_HASH)));
    if (prev.size() % 2) next.push_back(prev.back());
    t.layers.push_back(next);
  }
  const auto& last = t.layers.back();
  t.layers.push_back({blake3::parent_root(kw, cvw(last[0]), cvw(last[1]), blake3::KEYED_HASH)});
  return t;
}

void put64(std::vector<uint8_t>& o, uint64_t v) {
  for (int i = 0; i < 8; ++i) o.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put_proof(std::vector<uint8_t>& o, const MerkleProof& p, const std::vector<uint64_t>& rows) {
  put64(o, p.leaf_data.size());
  for (const auto& leaf : p.leaf_data) {
    put64(o, 1024);
    o.insert(o.end(), leaf.begin(), leaf.end());
  }
  put64(o, p.leaf_indices.size());
  for (auto i : p.leaf_indices) put64(o, i);
  put64(o, p.total_leaves);
  o.insert(o.end(), p.root.begin(), p.root.end());
  put64(o, p.siblings.size());
  for (const auto& s : p.siblings) o.insert(o.end(), s.begin(), s.end());
  put64(o, rows.size());
  for (auto r : rows) put64(o, r);
}

}  // namespace

MerkleProof matrix_proof(const Hash& key, const int8_t* data, size_t rows, size_t cols,
                         const std::vector<uint64_t>& row_indices) {
  size_t len = rows * cols;
  std::vector<uint8_t> buf((len + 1023) / 1024 * 1024, 0);
  std::memcpy(buf.data(), data, len);
  Tree t = build_tree(key, buf);

  std::set<uint64_t> leaves;
  for (uint64_t r : row_indices) {
    uint64_t first = r * cols / 1024, last = ((r + 1) * cols - 1) / 1024;
    for (uint64_t i = first; i <= last; ++i) leaves.insert(i);
  }
  MerkleProof p;
  p.total_leaves = t.layers[0].size();
  p.root = t.layers.back()[0];
  for (uint64_t i : leaves) {
    std::array<uint8_t, 1024> chunk{};
    std::memcpy(chunk.data(), buf.data() + i * 1024, 1024);
    p.leaf_data.push_back(chunk);
    p.leaf_indices.push_back(i);
  }
  // Siblings bottom-up, ascending within each level (merkle.rs get_multileaf_proof).
  std::set<uint64_t> cur = leaves;
  uint64_t level_len = p.total_leaves;
  size_t level = 0;
  while (level_len > 1 && !cur.empty()) {
    const auto& nodes = t.layers[level];
    for (uint64_t i : cur) {
      if (i % 2 == 1) {
        if (!cur.count(i - 1)) p.siblings.push_back(nodes[i - 1]);
      } else if (!cur.count(i + 1) && i + 1 < level_len) {
        p.siblings.push_back(nodes[i + 1]);
      }
    }
    std::set<uint64_t> up;
    for (uint64_t i : cur) up.insert(i / 2);
    cur.swap(up);
    level_len = (level_len + 1) / 2;
    ++level;
  }
  return p;
}

std::vector<uint8_t> serialize(const PlainProof& p) {
  std::vector<uint8_t> o;
  put64(o, p.m);
  put64(o, p.n);
  put64(o, p.k);
  put64(o, p.noise_rank);
  put_proof(o, p.a, p.a_rows);
  put_proof(o, p.bt, p.b_cols);
  o.push_back(0);  // moe: None (dense)
  return o;
}

std::string base64(const std::vector<uint8_t>& b) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((b.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < b.size(); i += 3) {
    uint32_t v = uint32_t(b[i]) << 16 | uint32_t(b[i + 1]) << 8 | b[i + 2];
    out += t[v >> 18];
    out += t[(v >> 12) & 63];
    out += t[(v >> 6) & 63];
    out += t[v & 63];
  }
  if (i < b.size()) {
    uint32_t v = uint32_t(b[i]) << 16 | (i + 1 < b.size() ? uint32_t(b[i + 1]) << 8 : 0);
    out += t[v >> 18];
    out += t[(v >> 12) & 63];
    out += i + 1 < b.size() ? t[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ CPU miner

std::vector<Found> cpu_mine(const Work& w, uint32_t nbits, size_t max_found) {
  const Config& c = w.config;
  Hash jk = job_key(w.header, c);
  Hash ha = matrix_root(jk, w.a.data(), w.m, c.k);
  Hash hb = matrix_root(jk, w.bt.data(), w.n, c.k);
  Seeds s = commitment_seeds(jk, ha, hb, w.m, w.n, w.salted);
  auto perm_a = permutation(Side::A, s.a_noise_seed, c.k, c.rank);
  auto perm_b = permutation(Side::B, s.b_noise_seed, c.k, c.rank);
  std::vector<std::vector<int32_t>> an(w.m, std::vector<int32_t>(c.k)), bn(w.n, std::vector<int32_t>(c.k));
  for (uint32_t i = 0; i < w.m; ++i) {
    auto e = noise_line(uniform_row(Side::A, s.a_noise_seed, i, c.rank), perm_a);
    for (uint32_t l = 0; l < c.k; ++l) an[i][l] = w.a[size_t(i) * c.k + l] + e[l];
  }
  for (uint32_t j = 0; j < w.n; ++j) {
    auto e = noise_line(uniform_row(Side::B, s.b_noise_seed, j, c.rank), perm_b);
    for (uint32_t l = 0; l < c.k; ++l) bn[j][l] = w.bt[size_t(j) * c.k + l] + e[l];
  }
  U256 bound = difficulty_bound(nbits, c);
  auto rl = c.rows.to_list(), cl = c.cols.to_list();
  std::vector<Found> out;
  for (uint32_t tr = 0; tr + c.rows.max() < w.m; ++tr) {
    if (!c.rows.offset_is_valid(tr)) continue;
    std::vector<std::vector<int32_t>> ar;
    for (uint32_t d : rl) ar.push_back(an[tr + d]);
    for (uint32_t tc = 0; tc + c.cols.max() < w.n; ++tc) {
      if (!c.cols.offset_is_valid(tc)) continue;
      std::vector<std::vector<int32_t>> bc;
      for (uint32_t d : cl) bc.push_back(bn[tc + d]);
      Found f;
      f.t_rows = tr;
      f.t_cols = tc;
      f.transcript = jackpot(ar, bc, c.k, c.rank);
      f.hash = jackpot_hash(f.transcript, s.a_noise_seed);
      if (U256::from_le_bytes(f.hash.data()) <= bound) {
        out.push_back(f);
        if (out.size() >= max_found) return out;
      }
    }
  }
  return out;
}

PlainProof build_proof(const Work& w, const Found& f) {
  const Config& c = w.config;
  Hash jk = job_key(w.header, c);
  PlainProof p;
  p.m = w.m;
  p.n = w.n;
  p.k = c.k;
  p.noise_rank = c.rank;
  for (uint32_t d : c.rows.to_list()) p.a_rows.push_back(f.t_rows + d);
  for (uint32_t d : c.cols.to_list()) p.b_cols.push_back(f.t_cols + d);
  std::sort(p.a_rows.begin(), p.a_rows.end());
  std::sort(p.b_cols.begin(), p.b_cols.end());
  p.a = matrix_proof(jk, w.a.data(), w.m, c.k, p.a_rows);
  p.bt = matrix_proof(jk, w.bt.data(), w.n, c.k, p.b_cols);
  return p;
}

}  // namespace xm::pearl
