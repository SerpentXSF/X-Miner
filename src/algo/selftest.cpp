#include "algo/selftest.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "algo/autolykos/autolykos_gpu.hpp"
#include "algo/blake2b/blake2b_gpu.hpp"
#include "algo/etchash/etchash_gpu.hpp"
#include "algo/fishhash/fishhash_gpu.hpp"
#include "algo/kawpow/kawpow_gpu.hpp"
#include "algo/quantus/quantus_gpu.hpp"
#include "algo/xelis/xelis_gpu.hpp"
#include "algo/noid/noid_gpu.hpp"
#include "algo/octopus/octopus_gpu.hpp"
#include "algo/pearl/pearl_gpu.hpp"
#include "algo/registry.hpp"
#include "core/console.hpp"
#include "gpu/kernels.hpp"

namespace xm {

namespace {

// Pipeline check: the xm_smoke kernel against the same expression on the CPU.
bool smoke(gpu::Device& dev, std::string& detail) {
  if (!gpu::load_kernels(dev, "smoke", detail)) return false;
  const unsigned n = 1u << 20;
  std::vector<unsigned> a(n), b(n), out(n, 0), want(n);
  uint32_t x = 12345;
  for (unsigned i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    a[i] = x;
    b[i] = x >> 7;
    want[i] = a[i] * b[i] + (i ^ 0x5a5a5a5au);
  }
  size_t bytes = n * sizeof(unsigned);
  gpu::DevPtr da = dev.alloc(bytes), db = dev.alloc(bytes), dout = dev.alloc(bytes);
  bool ok = da && db && dout && dev.upload(da, a.data(), bytes) && dev.upload(db, b.data(), bytes) &&
            dev.memset(dout, 0, bytes);
  unsigned nn = n;
  void* args[] = {&da, &db, &dout, &nn};
  auto t0 = std::chrono::steady_clock::now();
  ok = ok && dev.launch("xm_smoke", (n + 255) / 256, 1, 1, 256, 1, 1, 0, args) && dev.sync() &&
       dev.download(out.data(), dout, bytes);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  dev.free(da);
  dev.free(db);
  dev.free(dout);
  if (!ok) {
    detail = dev.last_error();
    return false;
  }
  unsigned bad = 0;
  for (unsigned i = 0; i < n; ++i) bad += out[i] != want[i];
  char buf[96];
  std::snprintf(buf, sizeof buf, "%u/%u words match (%.1f ms)", n - bad, n, ms);
  detail = buf;
  return bad == 0;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Pearl: one 1024 x 1024 x 8192 pass with a target every tile meets. The engine re-computes each reported tile's
// transcript on the CPU (host matrices + CPU noise), so this checks noise, GEMM, fold and hash against the reference.
bool pearl_check(gpu::Device& dev, std::string& detail) {
  pearl::Engine eng(dev, 1024, 1024, 0x5e1f7e57);
  pearl::Job job;
  job.id = "selftest";
  for (int i = 0; i < 76; ++i) job.header[i] = static_cast<uint8_t>(i * 37 + 11);
  job.share_nbits = 0x2000ffff;  // 2^248: the bound saturates, so every tile is a share
  auto t0 = std::chrono::steady_clock::now();
  std::vector<pearl::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail) || eng.iterate(shares, detail) == 0) return false;
  char buf[128];
  std::snprintf(buf, sizeof buf, "%zu tiles match the CPU reference, %llu mismatches (%.0f ms)", shares.size(),
                static_cast<unsigned long long>(eng.cpu_rejects()), ms_since(t0));
  detail = buf;
  return !shares.empty() && eng.cpu_rejects() == 0;
}

// Parano1d: 2^18 nonces of a known-answer header against target 2^246 (about 256 hits expected). Every GPU hit is
// re-hashed with the CPU reference digest.
bool noid_check(gpu::Device& dev, std::string& detail) {
  noid::Engine eng(dev, 0x5e1f7e57);
  noid::Job job;
  job.id = "selftest";
  for (int i = 0; i < 16; ++i) job.fields[i] = {0x9e3779b97f4a7c15ull * (i + 1), 0xbf58476d1ce4e5b9ull ^ uint64_t(i)};
  job.share_target[30] = 0x40;
  job.nonce_prefix = 0x0123456789abcdefull;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<noid::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail) || eng.iterate(1u << 18, shares, detail) == 0) return false;
  char buf[128];
  std::snprintf(buf, sizeof buf, "%zu hits verified by the CPU reference, %llu rejected (%.0f ms)", shares.size(),
                static_cast<unsigned long long>(eng.cpu_rejects()), ms_since(t0));
  detail = buf;
  return shares.size() >= 128 && shares.size() <= 512 && eng.cpu_rejects() == 0;
}

// BLAKE2b: 2^24 nonces of a random header against target 2^240 (about 256 hits expected). Every GPU hit is re-hashed
// with the CPU reference digest.
bool blake2b_check(gpu::Device& dev, std::string& detail) {
  b2b::Engine eng(dev, 0xb2b5e1f7);
  b2b::Job job;
  job.id = "selftest";
  for (size_t i = 0; i < job.prevhash.size(); ++i) job.prevhash[i] = static_cast<uint8_t>(i * 37 + 11);
  for (size_t i = 0; i < job.coinb1.size(); ++i) job.coinb1[i] = static_cast<uint8_t>(i * 91 + 5);
  job.ntime[0] = 0x5a;
  job.share_target[30] = 0x01;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<b2b::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail) || eng.iterate(1u << 24, shares, detail) == 0) return false;
  char buf[128];
  std::snprintf(buf, sizeof buf, "%zu hits verified by the CPU reference, %llu rejected (%.0f ms)", shares.size(),
                static_cast<unsigned long long>(eng.cpu_rejects()), ms_since(t0));
  detail = buf;
  return shares.size() >= 128 && shares.size() <= 512;
}

// Quantus: 2^20 nonces of a fixed header against target 2^500 (about 256 hits expected). Every GPU hit is re-hashed
// with the CPU reference.
bool quantus_check(gpu::Device& dev, std::string& detail) {
  qtc::Engine eng(dev, 0x97a5e1f7);
  qtc::Job job;
  job.id = "selftest";
  for (size_t i = 0; i < job.mining_hash.size(); ++i) job.mining_hash[i] = static_cast<uint8_t>(i * 29 + 7);
  job.extranonce = {0xa1, 0xb2, 0xc3, 0xd4};
  job.share_target[1] = 0x10;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<qtc::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail) || eng.iterate(1u << 20, shares, detail) == 0) return false;
  char buf[128];
  std::snprintf(buf, sizeof buf, "%zu hits verified by the CPU reference, %llu rejected (%.0f ms)", shares.size(),
                static_cast<unsigned long long>(eng.cpu_rejects()), ms_since(t0));
  detail = buf;
  return shares.size() >= 128 && shares.size() <= 512;
}

// ProgPoW family (KawPow, FiroPow, Epic ProgPoW 0.9.2, ProgPowZ): epoch 0 (a 1-1.5 GB DAG generated on the GPU), one period's program compiled at run time, 2^20 nonces against
// a 2^-12 target (about 256 hits). Every GPU hit is re-hashed with the CPU reference (light cache).
bool progpow_check(gpu::Device& dev, std::string& detail, const kawpow::Variant& v, int height) {
  kawpow::Engine eng(dev, 0x6a77e1f7, v);
  kawpow::Job job;
  job.id = "selftest";
  job.height = height;
  uint8_t* h = reinterpret_cast<uint8_t*>(job.header.data());
  for (int i = 0; i < 32; ++i) h[i] = static_cast<uint8_t>(i * 53 + 3);
  reinterpret_cast<uint8_t*>(job.target.data())[1] = 0x10;
  job.nonce_prefix = 0xabcdefull << 40;
  job.prefix_bits = 24;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<kawpow::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  if (eng.iterate(1u << 20, shares, detail) == 0) return false;
  char buf[160];
  std::snprintf(buf, sizeof buf, "%zu hits verified by the CPU reference, %llu rejected (DAG + compile %.0f ms, %.0f ms)",
                shares.size(), static_cast<unsigned long long>(eng.cpu_rejects()), setup, ms_since(t0));
  detail = buf;
  return shares.size() >= 128 && shares.size() <= 256 && eng.cpu_rejects() == 0;
}
bool kawpow_check(gpu::Device& dev, std::string& d) { return progpow_check(dev, d, kawpow::kKawpow, 1234); }
bool firopow_check(gpu::Device& dev, std::string& d) { return progpow_check(dev, d, kawpow::kFiropow, 1234); }
bool epic_check(gpu::Device& dev, std::string& d) { return progpow_check(dev, d, kawpow::kProgpow092, 1234); }
bool zano_check(gpu::Device& dev, std::string& d) { return progpow_check(dev, d, kawpow::kProgpowZ, 1234); }

// Etchash: epoch 0 (1 GB DAG), 2^20 nonces against a 2^-12 target (about 256 hits), every hit re-hashed on the CPU.
bool etchash_check(gpu::Device& dev, std::string& detail) {
  etchash::Engine eng(dev, 0xe7c5e1f7);
  etchash::Job job;
  job.id = "selftest";
  job.height = 1234;
  uint8_t* h = reinterpret_cast<uint8_t*>(job.header.data());
  for (int i = 0; i < 32; ++i) h[i] = static_cast<uint8_t>(i * 41 + 9);
  reinterpret_cast<uint8_t*>(job.target.data())[1] = 0x10;
  job.nonce_prefix = 0x0f57ull << 48;
  job.prefix_bits = 16;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<etchash::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  if (eng.iterate(1u << 20, shares, detail) == 0) return false;
  char buf[160];
  std::snprintf(buf, sizeof buf, "%zu hits verified by the CPU reference, %llu rejected (DAG + compile %.0f ms, %.0f ms)",
                shares.size(), static_cast<unsigned long long>(eng.cpu_rejects()), setup, ms_since(t0));
  detail = buf;
  return shares.size() >= 128 && shares.size() <= 256 && eng.cpu_rejects() == 0;
}

// Autolykos v2: a small height (N = 2^26, a 2 GB table built on the GPU) and 4096 consecutive nonces against
// b = 2^251 (about 128 hits). Every nonce is also hashed on the CPU: the GPU must report exactly the CPU's hits.
bool autolykos_check(gpu::Device& dev, std::string& detail) {
  autolykos::Engine eng(dev, 0xa1c5e1f7);
  autolykos::Job job;
  job.id = "selftest";
  job.height = 500000;
  for (int i = 0; i < 32; ++i) job.msg[i] = static_cast<uint8_t>(i * 37 + 5);
  job.b[0] = 0x08;  // about 128 hits: below the 256-candidate buffer
  job.nonce_prefix = 0xdf26ull << 48;
  job.prefix_bits = 16;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<autolykos::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  const uint64_t first = eng.next_nonce();
  if (eng.iterate(4096, shares, detail) == 0) return false;
  size_t want = 0, missing = 0;
  for (uint64_t k = 0; k < 4096; ++k) {
    if (!autolykos::less_than(autolykos::hit(job.msg, first + k, job.height), job.b)) continue;
    ++want;
    bool got = false;
    for (const auto& s : shares) got = got || s.nonce == first + k;
    missing += !got;
  }
  char buf[200];
  std::snprintf(buf, sizeof buf, "%zu of %zu CPU hits found in 4096 nonces, %llu wrong (table + compile %.0f ms, %.0f ms)",
                shares.size(), want, static_cast<unsigned long long>(eng.cpu_rejects()), setup, ms_since(t0));
  detail = buf;
  return want >= 64 && missing == 0 && shares.size() == want && eng.cpu_rejects() == 0;
}

// Octopus: stage 0 (a 4 GB DAG) and 1024 consecutive nonces (32 nonce groups) against a 2^-4 boundary (about 64
// hits). Every nonce is also hashed on the CPU: the GPU must report exactly the CPU's hits.
bool octopus_check(gpu::Device& dev, std::string& detail) {
  octopus::Engine eng(dev, 0x0c75e1f7);
  octopus::Job job;
  job.id = "selftest";
  job.height = 1234;
  uint8_t* h = reinterpret_cast<uint8_t*>(job.header.data());
  for (int i = 0; i < 32; ++i) h[i] = static_cast<uint8_t>(i * 29 + 3);
  reinterpret_cast<uint8_t*>(job.boundary.data())[0] = 0x10;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<octopus::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  const uint64_t first = eng.next_nonce();
  if (eng.iterate(1024, shares, detail) == 0) return false;
  const double gpu_ms = ms_since(t0);
  const auto light = octopus::make_light(0);
  octopus::Points pts;
  octopus::make_points(job.header, pts);
  size_t want = 0, missing = 0;
  for (uint64_t k = 0; k < 1024; ++k) {
    if (!octopus::below(octopus::hash(*light, pts, job.header, first + k), job.boundary)) continue;
    ++want;
    bool got = false;
    for (const auto& s : shares) got = got || s.nonce == first + k;
    missing += !got;
  }
  char buf[200];
  std::snprintf(buf, sizeof buf, "%zu of %zu CPU hits found in 1024 nonces, %llu wrong (DAG + compile %.0f ms, %.0f ms)",
                shares.size(), want, static_cast<unsigned long long>(eng.cpu_rejects()), setup, gpu_ms);
  detail = buf;
  return want >= 32 && missing == 0 && shares.size() == want && eng.cpu_rejects() == 0;
}

// FishHash / KarlsenHashV2: the fixed 4.5 GB dataset and 256 consecutive nonces against a 2^-3 target (about 32
// hits). Every nonce is also hashed on the CPU: the GPU must report exactly the CPU's hits.
bool fish_check(gpu::Device& dev, std::string& detail, fishhash::Variant v) {
  const bool iron = v == fishhash::Variant::Iron;
  fishhash::Engine eng(dev, 0xf15e1f7, v);
  fishhash::Job job;
  job.id = "selftest";
  for (int i = 0; i < 180; ++i) job.header[i] = static_cast<uint8_t>(i * 29 + 3);
  for (int i = 0; i < 32; ++i) job.pre_pow[i] = static_cast<uint8_t>(i * 41 + 9);
  job.timestamp = 5435345234ull;
  job.target[iron ? 0 : 31] = 0x20;
  job.nonce_prefix = 0x0766ull << 48;
  job.prefix_bits = 16;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<fishhash::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  const uint64_t first = eng.next_nonce();
  if (eng.iterate(256, shares, detail) == 0) return false;
  const double gpu_ms = ms_since(t0);
  const auto light = fishhash::make_light();
  size_t want = 0, missing = 0;
  for (uint64_t k = 0; k < 256; ++k) {
    const uint64_t n = first + k;
    fishhash::Bytes32 h;
    if (iron) {
      uint8_t header[180];
      std::memcpy(header, job.header, 172);
      for (int b = 0; b < 8; ++b) header[172 + b] = static_cast<uint8_t>(n >> (56 - 8 * b));
      h = fishhash::iron_hash(*light, header, 180);
    } else {
      h = fishhash::karlsen_hash(*light, job.pre_pow, job.timestamp, n);
    }
    if (!(iron ? fishhash::be_less_equal(h, job.target) : fishhash::le_less_equal(h, job.target))) continue;
    ++want;
    bool got = false;
    for (const auto& s : shares) got = got || s.nonce == n;
    missing += !got;
  }
  char buf[200];
  std::snprintf(buf, sizeof buf, "%zu of %zu CPU hits found in 256 nonces, %llu wrong (dataset + compile %.0f ms, %.0f ms)",
                shares.size(), want, static_cast<unsigned long long>(eng.cpu_rejects()), setup, gpu_ms);
  detail = buf;
  return want >= 12 && missing == 0 && shares.size() == want && eng.cpu_rejects() == 0;
}
bool fishhash_check(gpu::Device& dev, std::string& d) { return fish_check(dev, d, fishhash::Variant::Iron); }
bool karlsen_check(gpu::Device& dev, std::string& d) { return fish_check(dev, d, fishhash::Variant::Karlsen); }

// XelisHash v3: 256 consecutive nonces (256 scratchpads) against difficulty 8 (about 32 hits). Every nonce is also
// hashed on the CPU: the GPU must report exactly the CPU's hits.
bool xelis_check(gpu::Device& dev, std::string& detail) {
  xelis::Engine eng(dev, 0x8e15e1f7, 256);
  xelis::Job job;
  job.id = "selftest";
  for (int i = 0; i < xelis::kInputSize; ++i) job.work.bytes[i] = static_cast<uint8_t>(i * 29 + 3);
  job.difficulty = 8;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<xelis::Share> shares;
  if (!eng.init(detail) || !eng.set_job(job, detail)) return false;
  const double setup = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  const uint64_t first = eng.next_nonce();
  if (eng.iterate(shares, detail) == 0) return false;
  const double gpu_ms = ms_since(t0);
  std::vector<uint64_t> scratch;
  size_t want = 0, missing = 0;
  for (uint64_t k = 0; k < eng.threads(); ++k) {
    xelis::Work w = job.work;
    w.set_nonce(first + k);
    if (!xelis::meets_difficulty(xelis::hash(w.bytes, xelis::kInputSize, scratch), job.difficulty)) continue;
    ++want;
    bool got = false;
    for (const auto& s : shares) got = got || s.nonce == first + k;
    missing += !got;
  }
  char buf[200];
  std::snprintf(buf, sizeof buf, "%zu of %zu CPU hits found in %u nonces, %llu wrong (compile %.0f ms, %.0f ms)",
                shares.size(), want, eng.threads(), static_cast<unsigned long long>(eng.cpu_rejects()), setup, gpu_ms);
  detail = buf;
  return want >= 12 && missing == 0 && shares.size() == want && eng.cpu_rejects() == 0;
}

void report(size_t i, const gpu::DeviceInfo& d, const std::string& what, int state, const std::string& detail) {
  static const char* kState[] = {"FAIL  ", "PASS  ", "SKIP  "};
  Console::get().line({{Tone::Radar, "  [GPU " + std::to_string(i) + " ] "},
                       {Tone::Plain, d.name + "  "},
                       {Tone::Dim, what + " (" + d.arch + ")  "},
                       {state == 1 ? Tone::Ok : state == 0 ? Tone::Red : Tone::Dim, kState[state]},
                       {Tone::Dim, detail}});
}

}  // namespace

int run_selftest(const std::vector<gpu::DeviceInfo>& devices) {
  int failures = 0;
  struct Check {
    const char* algo;
    bool (*fn)(gpu::Device&, std::string&);
  };
  const Check checks[] = {{"pearlhash", pearl_check}, {"noid", noid_check}, {"blake2b", blake2b_check}, {"quantus", quantus_check}, {"kawpow", kawpow_check}, {"firopow", firopow_check}, {"progpow-epic", epic_check}, {"progpowz", zano_check}, {"etchash", etchash_check}, {"autolykos2", autolykos_check}, {"octopus", octopus_check}, {"fishhash", fishhash_check}, {"karlsenhashv2", karlsen_check}, {"xelishashv3", xelis_check}};
  for (size_t i = 0; i < devices.size(); ++i) {
    auto dev = gpu::open(devices[i]);
    std::string detail;
    bool ok = smoke(*dev, detail);
    failures += !ok;
    report(i, devices[i], "pipeline", ok, detail);
    for (const auto& c : checks) {
      const AlgorithmInfo* a = find_algorithm(c.algo);
      if (!a || !supports(*a, devices[i])) {
        report(i, devices[i], c.algo, 2, "no kernel for this GPU");
        continue;
      }
      detail.clear();
      ok = c.fn(*dev, detail);
      failures += !ok;
      report(i, devices[i], c.algo, ok, detail);
    }
  }
  return failures;
}

}  // namespace xm
