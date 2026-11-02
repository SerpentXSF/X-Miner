#include "algo/c29/c29.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "algo/blake2b/blake2b.hpp"

namespace xm::c29 {

namespace {

inline uint64_t rotl(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

struct Sip {
  uint64_t v0, v1, v2, v3;
  void round(int rot) {
    v0 += v1; v2 += v3;
    v1 = rotl(v1, 13); v3 = rotl(v3, 16);
    v1 ^= v0; v3 ^= v2;
    v0 = rotl(v0, 32);
    v2 += v1; v0 += v3;
    v1 = rotl(v1, 17); v3 = rotl(v3, rot);
    v1 ^= v2; v3 ^= v0;
    v2 = rotl(v2, 32);
  }
  // one siphash-2-4 of nonce on the running state (Tari's SipHash24::hash; the state carries over in a block)
  void hash(uint64_t nonce, int rot) {
    v3 ^= nonce;
    round(rot); round(rot);
    v0 ^= nonce;
    v2 ^= 0xff;
    for (int i = 0; i < 4; ++i) round(rot);
  }
  uint64_t digest() const { return (v0 ^ v1) ^ (v2 ^ v3); }
};

}  // namespace

Keys sip_keys(const Hash& mining_hash, uint64_t nonce) {
  uint8_t buf[40];
  for (int i = 0; i < 8; ++i) buf[i] = uint8_t(nonce >> (56 - 8 * i));
  std::memcpy(buf + 8, mining_hash.data(), 32);
  const auto d = b2b::blake2b256(buf, sizeof buf);
  Keys k{};
  for (int w = 0; w < 4; ++w)
    for (int i = 0; i < 8; ++i) k[w] |= uint64_t(d[8 * w + i]) << (8 * i);
  return k;
}

uint64_t siphash24(const Keys& k, uint64_t nonce, int rot) {
  Sip s{k[0], k[1], k[2], k[3]};
  s.hash(nonce, rot);
  return s.digest();
}

uint64_t siphash_block(const Keys& k, uint64_t edge, int rot) {
  const uint64_t edge0 = edge & ~63ull;
  uint64_t h[64];
  Sip s{k[0], k[1], k[2], k[3]};
  for (int i = 0; i < 64; ++i) {
    s.hash(edge0 + i, rot);
    h[i] = s.digest();
  }
  const uint64_t last = h[63];
  const int i = int(edge & 63);
  return i == 63 ? last : h[i] ^ last;
}

void edge_nodes(const Keys& k, uint32_t edge, uint32_t& u, uint32_t& v) {
  const uint64_t e = siphash_block(k, edge);
  u = uint32_t(e & kNodeMask);
  v = uint32_t((e >> 32) & kNodeMask);
}

bool verify(const Keys& k, const Proof& p) {
  uint64_t uv[2 * kProofSize];
  uint64_t x0 = 0, x1 = 0;
  for (int i = 0; i < kProofSize; ++i) {
    if (p[i] > kNodeMask || (i && p[i] <= p[i - 1])) return false;
    uint32_t u, v;
    edge_nodes(k, p[i], u, v);
    uv[2 * i] = u;
    uv[2 * i + 1] = v;
    x0 ^= u;
    x1 ^= v;
  }
  if (x0 || x1) return false;
  // the cycle walk of Grin's cuckaroo verifier: even entries are u, odd are v; k steps by 2 stay on one side
  int n = 0, i = 0;
  do {
    int j = i;
    for (int kk = (i + 2) % (2 * kProofSize); kk != i; kk = (kk + 2) % (2 * kProofSize)) {
      if (uv[kk] == uv[i]) {
        if (j != i) return false;  // a node with more than two edges
        j = kk;
      }
    }
    if (j == i) return false;  // a dead end
    i = j ^ 1;
    ++n;
  } while (i != 0);
  return n == kProofSize;
}

std::array<uint8_t, kPackedBytes> pack(const Proof& p) {
  std::array<uint8_t, kPackedBytes> out{};
  for (int i = 0; i < kProofSize; ++i)
    for (int b = 0; b < kEdgeBits; ++b)
      if (p[i] >> b & 1) {
        const int bit = i * kEdgeBits + b;
        out[bit >> 3] |= uint8_t(1u << (bit & 7));
      }
  return out;
}

Hash result(const Proof& p) {
  const auto packed = pack(p);
  return b2b::blake2b256_any(packed.data(), packed.size());
}

uint64_t share_difficulty(const Hash& r) {
  // (2^256 - 1) / x for x read big-endian: when x >= 2^192 the quotient fits in 64 bits and equals
  // floor((2^256 - 1) / x); compute with the top 128 bits of x (accurate to within 1 for the share check).
  uint64_t hi = 0, lo = 0;
  for (int i = 0; i < 8; ++i) hi = hi << 8 | r[i];
  for (int i = 8; i < 16; ++i) lo = lo << 8 | r[i];
  if (hi == 0) return ~0ull;  // x < 2^192: the difficulty exceeds 2^64
  // 2^128 / (hi:lo as a fraction of 2^128) ~ 2^64 / (hi + lo / 2^64)
  const long double x = (long double)hi + (long double)lo / 18446744073709551616.0L;
  const long double d = 18446744073709551616.0L / x;
  return d >= 18446744073709551615.0L ? ~0ull : (uint64_t)d;
}

uint64_t target_difficulty(const char* hex16) {
  uint64_t t = 0;
  for (int i = 0; i < 8; ++i) {  // 8 bytes, little-endian (XMRig convention)
    int b = 0;
    for (int h = 0; h < 2; ++h) {
      const char c = hex16[2 * i + h];
      const int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (v < 0) return 0;
      b = b << 4 | v;
    }
    t |= uint64_t(b) << (8 * i);
  }
  return t ? ~0ull / t : 0;
}

std::vector<std::array<uint32_t, kProofSize>> find_cycles(const std::vector<std::pair<uint32_t, uint32_t>>& edges) {
  // union-find over a spanning forest; an edge inside one tree closes the cycle tree path + edge
  std::unordered_map<uint64_t, uint32_t> id;
  std::vector<uint32_t> parent;
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> adj;  // (neighbor, edge index)
  auto node = [&](uint64_t key) {
    auto it = id.find(key);
    if (it != id.end()) return it->second;
    const uint32_t n = uint32_t(parent.size());
    id.emplace(key, n);
    parent.push_back(n);
    adj.emplace_back();
    return n;
  };
  auto find = [&](uint32_t x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
  };
  std::vector<std::array<uint32_t, kProofSize>> out;
  std::vector<int32_t> prev_node, prev_edge;
  for (uint32_t e = 0; e < edges.size(); ++e) {
    const uint32_t a = node(edges[e].first), b = node(uint64_t(1) << 32 | edges[e].second);
    const uint32_t ra = find(a), rb = find(b);
    if (ra != rb) {
      parent[ra] = rb;
      adj[a].push_back({b, e});
      adj[b].push_back({a, e});
      continue;
    }
    // tree path a -> b by BFS (the trees here are small)
    prev_node.assign(parent.size(), -1);
    prev_edge.assign(parent.size(), -1);
    std::vector<uint32_t> queue{a};
    prev_node[a] = int32_t(a);
    for (size_t q = 0; q < queue.size() && prev_node[b] < 0; ++q)
      for (auto [nb, ei] : adj[queue[q]])
        if (prev_node[nb] < 0) {
          prev_node[nb] = int32_t(queue[q]);
          prev_edge[nb] = int32_t(ei);
          queue.push_back(nb);
        }
    std::vector<uint32_t> cyc{e};
    for (uint32_t x = b; x != a && cyc.size() <= kProofSize; x = uint32_t(prev_node[x])) cyc.push_back(uint32_t(prev_edge[x]));
    if (cyc.size() == kProofSize) {
      std::sort(cyc.begin(), cyc.end());
      std::array<uint32_t, kProofSize> c{};
      std::copy(cyc.begin(), cyc.end(), c.begin());
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace xm::c29
