#include "algo/selftest.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

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

}  // namespace

int run_selftest(const std::vector<gpu::DeviceInfo>& devices) {
  int failures = 0;
  for (size_t i = 0; i < devices.size(); ++i) {
    auto dev = gpu::open(devices[i]);
    std::string detail;
    bool ok = smoke(*dev, detail);
    failures += !ok;
    Console::get().line({{Tone::Radar, "  [GPU " + std::to_string(i) + " ] "},
                         {Tone::Plain, devices[i].name + "  "},
                         {Tone::Dim, "pipeline (" + devices[i].arch + ")  "},
                         {ok ? Tone::Ok : Tone::Red, ok ? "PASS  " : "FAIL  "},
                         {Tone::Dim, detail}});
  }
  return failures;
}

}  // namespace xm
