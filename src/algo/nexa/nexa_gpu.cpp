#include "algo/nexa/nexa_gpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "crypto/secp256k1.hpp"
#include "crypto/sha256.hpp"
#include "gpu/kernels.hpp"
#include "nexa_cu.hpp"

namespace xm::nexa {

namespace {

inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

// The SHA-256 working state after rounds 0..n-1 of a block from st (message words w).
void rounds(const uint32_t st[8], const uint32_t* w, int n, uint32_t out[8]) {
  uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
  for (int i = 0; i < n; ++i) {
    const uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + crypto::kSha256K[i] + w[i];
    const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = e; out[5] = f; out[6] = g; out[7] = h;
}

// HMAC key state after the (all-zero 32-byte key) ^ pad block.
void zero_key_state(uint8_t pad, uint32_t out[8]) {
  uint8_t blk[64];
  std::memset(blk, pad, sizeof blk);
  std::memcpy(out, crypto::kSha256Init, 32);
  crypto::sha256_compress(out, blk);
}

struct Params {
  uint32_t cw[8], r8[8], is0[8], os0[8];
  uint32_t w8, en3;
  uint64_t base, target_hi;
  uint32_t dump, pad;
};
static_assert(sizeof(Params) == 160, "kernel parameter layout");

std::string hex(const uint8_t* p, size_t n) {
  static const char* hx = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += hx[p[i] >> 4];
    s += hx[p[i] & 15];
  }
  return s;
}

}  // namespace

std::array<uint8_t, 12> nonce_bytes(const Job& job, uint64_t nonce) {
  std::array<uint8_t, 12> b{};
  for (size_t i = 0; i < 4 && i < job.extranonce1.size(); ++i) b[i] = job.extranonce1[i];
  for (int i = 0; i < 8; ++i) b[4 + i] = uint8_t(nonce >> (8 * i));
  return b;
}

Engine::Engine(gpu::Device& dev, uint64_t seed, uint32_t batch) : dev_(dev), rng_(seed), batch_(batch) {}

Engine::~Engine() {
  dev_.free(d_found_);
  dev_.free(d_table_);
  dev_.free(d_dbg_);
  dev_.free(d_st_);
  dev_.free(d_pr_);
}

bool Engine::init(std::string& err) {
  dev_.make_current();
  if (const char* e = std::getenv("XM_NEXA_TPB")) tpb_ = std::max(32u, uint32_t(std::atoi(e)));  // developer tuning
  // Comb window: 24 bits (11 point additions per scalar, a 5.9 GB table) on NVIDIA cards with 10 GB or more (RTX
  // 3060: compiled 25.7 -> 28.3 MH/s, generated 30.5 -> 33.3), else 22 (12 additions, 1.6 GB; best on the RX 9070
  // XT: 53.0 at 20, 55.6 at 22, 55.0 at 24), 20 below 4 GB (13, 436 MB). Developer: XM_NEXA_W.
  const size_t mem = dev_.info().memory;
  w_ = dev_.info().vendor == gpu::Vendor::Nvidia && mem >= (size_t(10) << 30) ? 24 : mem >= (size_t(4) << 30) ? 22 : 20;
  if (const char* e = std::getenv("XM_NEXA_W")) w_ = std::clamp(std::atoi(e), 9, 24);
  if (batch_ == 0) {
    batch_ = 1u << 20;
    if (const char* e = std::getenv("XM_NEXA_BATCH")) batch_ = uint32_t(std::strtoul(e, nullptr, 10));
  }
  batch_ = std::max(tpb_, batch_ / tpb_ * tpb_);
  const int nw = (256 + w_ - 1) / w_ + (256 % w_ == 0 ? 1 : 0);
  d_found_ = dev_.alloc((1 + kMaxFound) * 4);
  if (!d_found_) {
    err = dev_.last_error();
    return false;
  }
  // The comb table, built on the GPU from the window bases 2^(W i) G (xm_nexa_table, prebuilt for every target:
  // kernels/nexatab; 256 threads per block): entry i 2^(W-1) + m - 1 = m 2^(W i) G, affine x then y, 8 little-endian
  // words each. Its module is loaded first; the run-time compiled one below replaces it.
  if (!gpu::load_kernels(dev_, "nexatab", err)) return false;
  {
    std::vector<uint32_t> bases(size_t(nw) * 16);
    for (int i = 0; i < nw; ++i) {
      crypto::secp256k1::U256 k{};
      k.w[(w_ * i) / 32] = 1u << ((w_ * i) % 32);
      const auto pt = crypto::secp256k1::mul_g(k);
      std::memcpy(&bases[16 * i], pt.x.w, 32);
      std::memcpy(&bases[16 * i + 8], pt.y.w, 32);
    }
    const size_t entries = size_t(nw) << (w_ - 1);
    d_table_ = dev_.alloc(entries * 64);
    gpu::DevPtr d_bases = dev_.alloc(bases.size() * 4);
    uint32_t w = uint32_t(w_);
    void* args[] = {&d_bases, &d_table_, &w};
    const bool ok = d_table_ && d_bases && dev_.upload(d_bases, bases.data(), bases.size() * 4) &&
                    dev_.launch("xm_nexa_table", unsigned(entries / 256), 1, 1, 256, 1, 1, 0, args) && dev_.wait();
    dev_.free(d_bases);
    if (!ok) {
      err = "NexaPow comb table (" + std::to_string(entries * 64 >> 20) + " MB): " + dev_.last_error();
      return false;
    }
  }
  // Blocks per SM the register budget is fitted to: 2 on NVIDIA (RTX 3060: 15.0 vs 10.6 MH/s with 1; 3 spills).
  // HIP reads the second launch bound differently; AMD keeps 1. Developer: XM_NEXA_MINB.
  int minb = dev_.info().vendor == gpu::Vendor::Nvidia ? 2 : 1;
  if (const char* e = std::getenv("XM_NEXA_MINB")) minb = std::clamp(std::atoi(e), 1, 8);
  // RTX 30 with the defaults the prebuilt cubin is made for (W 20, 256 threads, 2 blocks): the three-kernel form,
  // with the prebuilt (generated) xm_nexa_b added next to the run-time compiled kernels
  const char* gen_env = std::getenv("XM_NEXA_GEN");
  const auto archs = gpu::kernel_archs("nexa");
  const bool want_gen = dev_.info().vendor == gpu::Vendor::Nvidia && w_ == kGenW && tpb_ == 256 && minb == 2 &&
                        std::find(archs.begin(), archs.end(), dev_.info().arch) != archs.end() &&
                        !(gen_env && std::string(gen_env) == "0");
  {
    std::string kernel(reinterpret_cast<const char*>(nexa_cu_bytes), nexa_cu_size);
    if (const char* f = std::getenv("XM_NEXA_KERNEL")) {  // developer: kernel source from a file
      std::ifstream in(f, std::ios::binary);
      std::stringstream ss;
      ss << in.rdbuf();
      if (!in) {
        err = std::string("XM_NEXA_KERNEL ") + f + " not readable";
        return false;
      }
      kernel = ss.str();
    }
    // The three-kernel form: with the generated kernel, and on AMD (RX 9070 XT: 53.1 vs 52.0 MH/s, 325 vs 339 W).
    // Other NVIDIA GPUs keep the single kernel (RTX 3060 without the generated kernel: a tie). Developer:
    // XM_NEXA_SPLIT=0/1.
    split_ = want_gen || dev_.info().vendor == gpu::Vendor::Amd;
    if (const char* e = std::getenv("XM_NEXA_SPLIT")) split_ = std::string(e) != "0";
    // Kernel c of that form keeps few values live across its inversion and Legendre loops: fitted to 32 warps per SM
    // (RTX 3060: its time -20% against 2 blocks of 256). Developer: XM_NEXA_MINB_C.
    int minb_c = std::clamp(std::max(minb, 1024 / int(tpb_)), 1, 8);
    if (const char* e = std::getenv("XM_NEXA_MINB_C")) minb_c = std::clamp(std::atoi(e), 1, 8);
    std::string src = "#define TPB " + std::to_string(tpb_) + "\n#define W " + std::to_string(w_) + "\n#define NW " +
                      std::to_string(nw) + "\n#define MINB " + std::to_string(minb) + "\n#define MINB_C " +
                      std::to_string(minb_c) + "\n" + (split_ ? "#define XM_NEXA_SPLIT\n" : "") + kernel;
    std::vector<std::string> opts;
    if (dev_.info().vendor == gpu::Vendor::Amd) opts.push_back("-O3");
    if (!gpu::rtc_compile(dev_.info(), src, "nexa.cu", opts, module_)) {
      err = "NexaPow kernel did not compile: " + module_.log.substr(0, 3000);
      return false;
    }
    if (!dev_.load_module(gpu::rtc_blob(module_, dev_.info().vendor, "nexa"))) {
      err = dev_.last_error();
      return false;
    }
  }
  std::string why;
  gen_ = want_gen && split_ && gpu::add_kernels(dev_, "nexa", why);
  return true;
}

bool Engine::set_job(const Job& job, std::string&) {
  const bool same_work = have_job_ && job.commitment == job_.commitment && job.extranonce1 == job_.extranonce1;
  if (!same_work) {
    // A random start per GPU and job keeps several GPUs (and restarts) off each other's nonces.
    uint64_t r = rng_ += 0x9e3779b97f4a7c15ull;
    r = (r ^ (r >> 31)) * 0xbf58476d1ce4e5b9ull;
    counter_ = r ^ (r >> 29);
  }
  job_ = job;
  target_ = target_from_difficulty(job.difficulty);
  block_target_ = target_from_nbits(job.nbits);
  // first block: commitment (words 0..7), 0x0c | en1 | nonce; rounds 0..7 depend on the commitment only
  Params p{};
  for (int i = 0; i < 8; ++i) p.cw[i] = be32(job.commitment.data() + 4 * i);
  rounds(crypto::kSha256Init, p.cw, 8, p.r8);
  zero_key_state(0x36, p.is0);
  zero_key_state(0x5c, p.os0);
  uint8_t en[4] = {};
  for (size_t i = 0; i < 4 && i < job.extranonce1.size(); ++i) en[i] = job.extranonce1[i];
  p.w8 = 0x0cu << 24 | uint32_t(en[0]) << 16 | uint32_t(en[1]) << 8 | en[2];
  p.en3 = en[3];
  p.target_hi = target_top64(target_);
  params_.assign(sizeof(Params) / 4, 0);
  std::memcpy(params_.data(), &p, sizeof p);
  have_job_ = true;
  return true;
}

bool Engine::launch(uint64_t base, uint32_t n, uint32_t dump, std::string& err) {
  Params p;
  std::memcpy(&p, params_.data(), sizeof p);
  p.base = base;
  p.dump = dump;
  uint32_t maxf = kMaxFound;
  if (!dev_.memset(d_found_, 0, 4)) {
    err = dev_.last_error();
    return false;
  }
  bool ok;
  if (split_) {
    if (split_n_ < n) {  // per nonce: 27 words of stage-1 results, 68 for P and R (kernels/nexa/nexa.cu PR_ROW)
      dev_.free(d_st_);
      dev_.free(d_pr_);
      d_st_ = dev_.alloc(size_t(n) * 27 * 4);
      d_pr_ = dev_.alloc(size_t(n) * 68 * 4);
      split_n_ = d_st_ && d_pr_ ? n : 0;
      if (!split_n_) {
        err = dev_.last_error();
        return false;
      }
    }
    void* aa[] = {&p, &d_st_, &n};
    void* ba[] = {&d_table_, &d_st_, &d_pr_, &n};
    void* ca[] = {&p, &d_st_, &d_pr_, &n, &d_found_, &maxf, &d_dbg_};
    ok = dev_.launch("xm_nexa_a", n / tpb_, 1, 1, tpb_, 1, 1, 0, aa) &&
         dev_.launch("xm_nexa_b", n / tpb_, 1, 1, tpb_, 1, 1, 0, ba) &&
         dev_.launch("xm_nexa_c", n / tpb_, 1, 1, tpb_, 1, 1, 0, ca);
  } else {
    void* args[] = {&p, &d_table_, &d_found_, &maxf, &d_dbg_};
    ok = dev_.launch("xm_nexa", n / tpb_, 1, 1, tpb_, 1, 1, 0, args);
  }
  if (!ok || !dev_.wait()) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

bool Engine::dump(uint64_t base, uint32_t n, std::vector<uint32_t>& out, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return false;
  }
  n = std::max(tpb_, n / tpb_ * tpb_);
  const size_t bytes = size_t(n) * 24 * 4;
  if (dbg_bytes_ < bytes) {
    dev_.free(d_dbg_);
    d_dbg_ = dev_.alloc(bytes);
    dbg_bytes_ = d_dbg_ ? bytes : 0;
    if (!d_dbg_) {
      err = dev_.last_error();
      return false;
    }
  }
  if (!launch(base, n, 1, err)) return false;
  if (const char* f = std::getenv("XM_NEXA_DUMP_PR"); f && split_) {  // developer: stage-1 words and P / R rows
    std::vector<uint32_t> st(size_t(n) * 27), pr(size_t(n) * 68);
    if (dev_.download(st.data(), d_st_, st.size() * 4) && dev_.download(pr.data(), d_pr_, pr.size() * 4)) {
      if (FILE* fp = std::fopen(f, "wb")) {
        std::fwrite(&n, 4, 1, fp);
        std::fwrite(st.data(), 4, st.size(), fp);
        std::fwrite(pr.data(), 4, pr.size(), fp);
        std::fclose(fp);
      }
    }
  }
  out.resize(size_t(n) * 24);
  if (!dev_.download(out.data(), d_dbg_, bytes)) {
    err = dev_.last_error();
    return false;
  }
  return true;
}

double Engine::iterate(std::vector<Share>& shares, std::string& err) {
  if (!have_job_) {
    err = "no job";
    return 0;
  }
  const uint32_t n = batch_;
  const uint64_t base = counter_;
  if (!launch(base, n, 0, err)) return 0;
  std::vector<uint32_t> found(1 + kMaxFound);
  if (!dev_.download(found.data(), d_found_, found.size() * 4)) {
    err = dev_.last_error();
    return 0;
  }
  counter_ += n;
  const uint32_t hits = std::min(found[0], kMaxFound);
  for (uint32_t k = 0; k < hits; ++k) {
    Share s;
    s.nonce = base + found[1 + k];
    const auto nb = nonce_bytes(job_, s.nonce);
    if (!pow_hash(job_.commitment, nb.data(), nb.size(), s.hash) || !meets(s.hash, target_)) {
      ++cpu_rejects_;
      continue;
    }
    s.job_id = job_.id;
    s.extranonce1_hex = hex(nb.data(), 4);
    s.nonce_hex = hex(nb.data() + 4, 8);
    s.work = job_.difficulty * 4294967296.0;
    s.is_block = meets(s.hash, block_target_);
    shares.push_back(s);
  }
  return n;
}

}  // namespace xm::nexa
