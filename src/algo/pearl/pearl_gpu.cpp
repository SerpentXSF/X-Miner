#include "algo/pearl/pearl_gpu.hpp"

#include <cstring>

#include "gpu/kernels.hpp"

namespace xm::pearl {

namespace {

struct Key8 {
  uint32_t w[8];
};
struct U256w {
  uint32_t w[8];
};

Key8 key8(const Hash& h) {
  Key8 k;
  auto w = blake3::key_words(h.data());
  for (int i = 0; i < 8; ++i) k.w[i] = w[i];
  return k;
}

unsigned blocks(size_t threads, unsigned bs) { return static_cast<unsigned>((threads + bs - 1) / bs); }

}  // namespace

Engine::Engine(gpu::Device& dev, uint32_t m, uint32_t n, uint64_t seed) : dev_(dev), m_(m), n_(n), rng_(seed) {
  cfg_.k = 4096;
  cfg_.rank = 128;
  cfg_.rows = Pattern::contiguous(16);
  cfg_.cols = Pattern::contiguous(16);
}

Engine::~Engine() {
  for (gpu::DevPtr p : {d_a_, d_bt_, d_ap_, d_bp_, d_uni_, d_pairs_, d_cvs_, d_out_, d_found_}) dev_.free(p);
}

void Engine::fill_random(int8_t* p, size_t n) {
  // Signal values in [-64, 63]; 8 per 64-bit draw.
  size_t i = 0;
  while (i < n) {
    uint64_t r = rng_();
    for (int j = 0; j < 8 && i < n; ++j, ++i) p[i] = static_cast<int8_t>(static_cast<int>((r >> (8 * j)) & 127) - 64);
  }
}

bool Engine::init(std::string& err) {
  if (m_ % 128 || n_ % 128) {
    err = "matrix dimensions must be multiples of 128";
    return false;
  }
  dev_.make_current();
  if (!gpu::load_kernels(dev_, "pearl", err)) return false;
  const uint32_t k = cfg_.k, r = cfg_.rank;
  size_t sa = size_t(m_) * k, sb = size_t(n_) * k, tiles = size_t(m_ / 16) * (n_ / 16);
  a_.resize(sa);
  bt_.resize(sb);
  fill_random(a_.data(), sa);
  fill_random(bt_.data(), sb);
  d_a_ = dev_.alloc(sa);
  d_bt_ = dev_.alloc(sb);
  d_ap_ = dev_.alloc(sa);
  d_bp_ = dev_.alloc(sb);
  d_uni_ = dev_.alloc(size_t(std::max(m_, n_)) * r);
  d_pairs_ = dev_.alloc(size_t(k) * 2 * 4);
  d_cvs_ = dev_.alloc(std::max(sa, sb) / 1024 * 32);
  d_out_ = dev_.alloc(tiles * 32 * 4);
  d_found_ = dev_.alloc((1 + kMaxFound * 17) * 4);
  if (!d_a_ || !d_bt_ || !d_ap_ || !d_bp_ || !d_uni_ || !d_pairs_ || !d_cvs_ || !d_out_ || !d_found_) {
    err = "out of GPU memory (" + dev_.last_error() + ")";
    return false;
  }
  if (!dev_.upload(d_a_, a_.data(), sa) || !dev_.upload(d_bt_, bt_.data(), sb)) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::chunk_tree(gpu::DevPtr data, size_t len, MerkleTree& out, std::string& err) {
  uint32_t chunks = static_cast<uint32_t>(len / 1024);
  Key8 key = key8(job_key_);
  void* args[] = {&data, &chunks, &key, &d_cvs_};
  if (!dev_.launch("xm_b3_chunk_cvs", blocks(chunks, 128), 1, 1, 128, 1, 1, 0, args) || !dev_.sync()) {
    err = dev_.last_error();
    return false;
  }
  std::vector<uint32_t> w(size_t(chunks) * 8);
  if (!dev_.download(w.data(), d_cvs_, w.size() * 4)) {
    err = dev_.last_error();
    return false;
  }
  std::vector<Hash> cvs(chunks);
  for (uint32_t c = 0; c < chunks; ++c) {
    blake3::Cv cv;
    for (int i = 0; i < 8; ++i) cv[i] = w[size_t(c) * 8 + i];
    cvs[c] = blake3::cv_bytes(cv);
  }
  out = tree_from_chunk_cvs(job_key_, std::move(cvs));
  return true;
}

bool Engine::launch_noise(bool side_b, const Hash& seed, gpu::DevPtr sig, gpu::DevPtr packed, uint32_t rows,
                          std::string& err) {
  Key8 key = key8(seed);
  uint32_t sb = side_b ? 1 : 0, k = cfg_.k, r = cfg_.rank;
  uint32_t nhash = rows * r / 32;
  void* a1[] = {&key, &sb, &nhash, &d_uni_};
  void* a2[] = {&key, &sb, &k, &r, &d_pairs_};
  void* a3[] = {&sig, &d_uni_, &d_pairs_, &packed, &rows, &k, &r};
  bool ok = dev_.launch("xm_noise_uniform", blocks(nhash, 256), 1, 1, 256, 1, 1, 0, a1) &&
            dev_.launch("xm_noise_perm", blocks(k / 8, 128), 1, 1, 128, 1, 1, 0, a2) &&
            dev_.launch("xm_apply_noise", blocks(size_t(rows) * (k / 8), 256), 1, 1, 256, 1, 1, 0, a3);
  if (!ok) err = dev_.last_error();
  return ok;
}

bool Engine::set_job(const Job& job, std::string& err) {
  job_ = job;
  job_key_ = job_key(job.header, cfg_);
  const size_t sa = size_t(m_) * cfg_.k, sb = size_t(n_) * cfg_.k;
  if (!chunk_tree(d_a_, sa, tree_a_, err) || !chunk_tree(d_bt_, sb, tree_b_, err)) return false;
  // B's seed depends only on the job key and B's commitment, so B' is built once per job.
  seeds_ = commitment_seeds(job_key_, tree_a_.root(), tree_b_.root(), m_, n_, job.salted);
  if (!launch_noise(true, seeds_.b_noise_seed, d_bt_, d_bp_, n_, err) || !dev_.sync()) {
    if (err.empty()) err = dev_.last_error();
    return false;
  }
  have_job_ = true;
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  const uint32_t k = cfg_.k;
  // New nonce chunk: changes A's commitment and therefore both noise seeds' A side.
  nonce_chunk_ = (nonce_chunk_ + 1) % tree_a_.leaves();
  int8_t* chunk = a_.data() + nonce_chunk_ * 1024;
  fill_random(chunk, 1024);
  if (!dev_.upload(d_a_ + nonce_chunk_ * 1024, chunk, 1024)) {
    err = dev_.last_error();
    return 0;
  }
  tree_update_chunk(tree_a_, nonce_chunk_, reinterpret_cast<const uint8_t*>(chunk));
  seeds_ = commitment_seeds(job_key_, tree_a_.root(), tree_b_.root(), m_, n_, job_.salted);

  if (!launch_noise(false, seeds_.a_noise_seed, d_a_, d_ap_, m_, err)) return 0;
  int M = static_cast<int>(m_), N = static_cast<int>(n_), K = static_cast<int>(k);
  void* g[] = {&d_ap_, &d_bp_, &d_out_, &M, &N, &K};
  if (!dev_.launch("xm_pearl_gemm", n_ / 128, m_ / 128, 1, 128, 1, 1, 0, g)) {
    err = dev_.last_error();
    return 0;
  }
  uint32_t tiles = (m_ / 16) * (n_ / 16), maxf = kMaxFound;
  Key8 seed = key8(seeds_.a_noise_seed);
  U256 bound = difficulty_bound(job_.share_nbits, cfg_);
  U256w bw;
  for (int i = 0; i < 8; ++i) bw.w[i] = bound.w[i];
  void* j[] = {&d_out_, &tiles, &seed, &bw, &d_found_, &maxf};
  std::vector<uint32_t> found(1 + kMaxFound * 17);
  if (!dev_.memset(d_found_, 0, 4) || !dev_.launch("xm_jackpot_check", blocks(tiles, 256), 1, 1, 256, 1, 1, 0, j) ||
      !dev_.download(found.data(), d_found_, found.size() * 4)) {  // the download waits for the stream
    err = dev_.last_error();
    return 0;
  }
  uint32_t hits = std::min(found[0], kMaxFound);
  if (hits) {
    U256 block_bound = difficulty_bound(header_nbits(job_.header), cfg_);
    auto perm_a = permutation(Side::A, seeds_.a_noise_seed, k, cfg_.rank);
    auto perm_b = permutation(Side::B, seeds_.b_noise_seed, k, cfg_.rank);
    for (uint32_t h = 0; h < hits; ++h) {
      const uint32_t* rec = &found[1 + h * 17];
      Found f;
      f.t_rows = rec[0] / (n_ / 16) * 16;
      f.t_cols = rec[0] % (n_ / 16) * 16;
      for (int w = 0; w < 16; ++w) f.transcript[w] = rec[1 + w];
      // Re-verify on the CPU reference: the transcript of this tile from the host matrices and noise.
      std::vector<std::vector<int32_t>> ar, bc;
      for (uint32_t d = 0; d < 16; ++d) {
        auto ea = noise_line(uniform_row(Side::A, seeds_.a_noise_seed, f.t_rows + d, cfg_.rank), perm_a);
        auto eb = noise_line(uniform_row(Side::B, seeds_.b_noise_seed, f.t_cols + d, cfg_.rank), perm_b);
        std::vector<int32_t> ra(k), rb(k);
        for (uint32_t l = 0; l < k; ++l) {
          ra[l] = a_[size_t(f.t_rows + d) * k + l] + ea[l];
          rb[l] = bt_[size_t(f.t_cols + d) * k + l] + eb[l];
        }
        ar.push_back(std::move(ra));
        bc.push_back(std::move(rb));
      }
      auto cpu_t = jackpot(ar, bc, k, cfg_.rank);
      f.hash = jackpot_hash(cpu_t, seeds_.a_noise_seed);
      if (cpu_t != f.transcript || !(U256::from_le_bytes(f.hash.data()) <= bound)) {
        ++cpu_rejects_;
        continue;
      }
      PlainProof p;
      p.m = m_;
      p.n = n_;
      p.k = k;
      p.noise_rank = cfg_.rank;
      for (uint32_t d = 0; d < 16; ++d) {
        p.a_rows.push_back(f.t_rows + d);
        p.b_cols.push_back(f.t_cols + d);
      }
      p.a = tree_proof(tree_a_, a_.data(), a_.size(), k, p.a_rows);
      p.bt = tree_proof(tree_b_, bt_.data(), bt_.size(), k, p.b_cols);
      Share s;
      s.job_id = job_.id;
      s.found = f;
      s.proof_b64 = base64(serialize(p));
      s.is_block = U256::from_le_bytes(f.hash.data()) <= block_bound;
      shares.push_back(std::move(s));
    }
  }
  return double(m_) * double(n_) * double(k);
}

}  // namespace xm::pearl
