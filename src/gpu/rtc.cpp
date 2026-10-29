#include "gpu/rtc.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#ifndef _WIN32
#include <dirent.h>
#include <unistd.h>
#endif
#include <string>

#include "gpu/dynlib.hpp"

namespace xm::gpu {

namespace {

#ifndef _WIN32
std::string exe_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  std::string dir(buf, size_t(n));
  return dir.substr(0, dir.rfind('/') + 1);
}
#endif

// Linux dlopen() does not look next to the executable; Windows LoadLibrary does (the libraries ship there).
bool open_first(DynLib& lib, const std::vector<std::string>& names) {
#ifndef _WIN32
  const std::string dir = exe_dir();
  if (!dir.empty())
    for (const auto& name : names)
      if (name.find('/') == std::string::npos && lib.open((dir + name).c_str())) return true;
#endif
  for (const auto& name : names)
    if (lib.open(name.c_str())) return true;
  return false;
}

// ---------------------------------------------------------------- NVRTC
struct Nvrtc {
  DynLib builtins;  // Linux: the shipped libnvrtc-builtins, preloaded
  DynLib lib;
  using P = void*;
  int (*create)(P*, const char*, const char*, int, const char* const*, const char* const*) = nullptr;
  int (*compile)(P, int, const char* const*) = nullptr;
  int (*log_size)(P, size_t*) = nullptr;
  int (*log)(P, char*) = nullptr;
  int (*cubin_size)(P, size_t*) = nullptr;
  int (*cubin)(P, char*) = nullptr;
  int (*ptx_size)(P, size_t*) = nullptr;
  int (*ptx)(P, char*) = nullptr;
  int (*destroy)(P*) = nullptr;
  int (*num_archs)(int*) = nullptr;
  int (*archs)(int*) = nullptr;
  std::vector<int> supported;  // e.g. 50 ... 90 (compute capability x 10)
  bool ready = false;
  Nvrtc() {
#ifdef _WIN32
    if (!open_first(lib, {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"})) return;
#else
    // NVRTC dlopen()s libnvrtc-builtins.so.<major>.<minor> by name at compile time, which misses the copy shipped next
    // to x-miner: load it from there first, so the by-name lookup finds the already loaded library.
    if (DIR* d = opendir(exe_dir().c_str())) {
      while (dirent* e = readdir(d)) {
        const std::string f = e->d_name;
        if (f.rfind("libnvrtc-builtins.so.", 0) == 0 && std::count(f.begin(), f.end(), '.') == 3) builtins.open((exe_dir() + f).c_str());
      }
      closedir(d);
    }
    if (!open_first(lib, {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"})) return;
#endif
    ready = lib.sym("nvrtcCreateProgram", create) && lib.sym("nvrtcCompileProgram", compile) &&
            lib.sym("nvrtcGetProgramLogSize", log_size) && lib.sym("nvrtcGetProgramLog", log) &&
            lib.sym("nvrtcGetPTXSize", ptx_size) && lib.sym("nvrtcGetPTX", ptx) &&
            lib.sym("nvrtcDestroyProgram", destroy);
    lib.sym("nvrtcGetCUBINSize", cubin_size);
    lib.sym("nvrtcGetCUBIN", cubin);
    int n = 0;
    if (lib.sym("nvrtcGetNumSupportedArchs", num_archs) && lib.sym("nvrtcGetSupportedArchs", archs) &&
        num_archs(&n) == 0 && n > 0) {
      supported.resize(size_t(n));
      if (archs(supported.data()) != 0) supported.clear();
    }
  }
};
Nvrtc& nvrtc() {
  static Nvrtc n;
  return n;
}

// ---------------------------------------------------------------- hipRTC
struct Hiprtc {
  DynLib lib;
  using P = void*;
  int (*create)(P*, const char*, const char*, int, const char**, const char**) = nullptr;
  int (*compile)(P, int, const char**) = nullptr;
  int (*log_size)(P, size_t*) = nullptr;
  int (*log)(P, char*) = nullptr;
  int (*code_size)(P, size_t*) = nullptr;
  int (*code)(P, char*) = nullptr;
  int (*destroy)(P*) = nullptr;
  bool ready = false;
  Hiprtc() {
#ifdef _WIN32
    if (!open_first(lib, {"hiprtc0702.dll", "hiprtc0701.dll", "hiprtc0700.dll", "hiprtc0605.dll", "hiprtc0604.dll",
                          "hiprtc0603.dll", "hiprtc0602.dll", "hiprtc0601.dll", "hiprtc0600.dll"}))
      return;
#else
    if (!open_first(lib, {"libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so", "/opt/rocm/lib/libhiprtc.so.7",
                          "/opt/rocm/lib/libhiprtc.so.6", "/opt/rocm/lib/libhiprtc.so"}))
      return;
#endif
    ready = lib.sym("hiprtcCreateProgram", create) && lib.sym("hiprtcCompileProgram", compile) &&
            lib.sym("hiprtcGetProgramLogSize", log_size) && lib.sym("hiprtcGetProgramLog", log) &&
            lib.sym("hiprtcGetCodeSize", code_size) && lib.sym("hiprtcGetCode", code) &&
            lib.sym("hiprtcDestroyProgram", destroy);
  }
};
Hiprtc& hiprtc() {
  static Hiprtc h;
  return h;
}

std::mutex& rtc_mutex() {  // both compilers are documented thread-safe, but one compile at a time keeps memory down
  static std::mutex m;
  return m;
}

// Developer: XM_RTC_DUMP=<dir> writes each compiled source and its code object / cubin / PTX there.
void dump(const std::string& source, const std::string& name, const RtcOutput& out) {
  const char* dir = std::getenv("XM_RTC_DUMP");
  if (!dir) return;
  const std::string base = std::string(dir) + "/" + name + "." + out.arch;
  std::ofstream(base + ".src", std::ios::binary) << source;
  std::ofstream(base + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(out.code.data()),
                                                       std::streamsize(out.code.size()));
}

}  // namespace

std::string rtc_library(Vendor v) {
  if (v == Vendor::Nvidia) return nvrtc().ready ? nvrtc().lib.name() : "";
  return hiprtc().ready ? hiprtc().lib.name() : "";
}

bool rtc_compile(const DeviceInfo& dev, const std::string& source, const std::string& name,
                 const std::vector<std::string>& options, RtcOutput& out) {
  std::lock_guard<std::mutex> lk(rtc_mutex());
  out = RtcOutput{};
  std::vector<std::string> opts = options;
  if (const char* e = std::getenv("XM_RTC_OPTS")) {  // developer: extra compiler options, space separated
    std::string s = e;
    for (size_t p = 0; p < s.size();) {
      const size_t q = s.find(' ', p);
      if (q != p) opts.push_back(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
      if (q == std::string::npos) break;
      p = q + 1;
    }
  }
  if (dev.vendor == Vendor::Nvidia) {
    Nvrtc& N = nvrtc();
    if (!N.ready) {
      out.log = "NVRTC (nvrtc64_*.dll / libnvrtc.so) not found next to x-miner";
      return false;
    }
    // A cubin for the exact GPU when this NVRTC knows it; otherwise PTX for the newest architecture it knows that
    // the GPU can run (the driver JIT-compiles it).
    const int cc = std::stoi(dev.arch.substr(3));
    int best = 0;
    bool exact = false;
    for (int a : N.supported) {
      if (a == cc) exact = true;
      if (a <= cc && a > best) best = a;
    }
    if (N.supported.empty()) best = cc;  // old NVRTC without the query: try the GPU's own architecture
    exact = exact && N.cubin && N.cubin_size;
    out.arch = exact ? dev.arch : "compute_" + std::to_string(best);
    opts.push_back("--gpu-architecture=" + (exact ? dev.arch : out.arch));
    opts.push_back("-std=c++17");
    void* prog = nullptr;
    if (N.create(&prog, source.c_str(), name.c_str(), 0, nullptr, nullptr) != 0) {
      out.log = "nvrtcCreateProgram failed";
      return false;
    }
    std::vector<const char*> argv;
    for (auto& o : opts) argv.push_back(o.c_str());
    const int rc = N.compile(prog, int(argv.size()), argv.data());
    size_t n = 0;
    if (N.log_size(prog, &n) == 0 && n > 1) {
      out.log.resize(n);
      N.log(prog, &out.log[0]);
      out.log.resize(n - 1);
    }
    bool ok = rc == 0;
    if (ok) {
      size_t sz = 0;
      if (exact) {
        ok = N.cubin_size(prog, &sz) == 0 && sz > 0;
        if (ok) {
          out.code.resize(sz);
          ok = N.cubin(prog, reinterpret_cast<char*>(out.code.data())) == 0;
        }
      } else {
        ok = N.ptx_size(prog, &sz) == 0 && sz > 0;
        if (ok) {
          out.code.resize(sz);  // includes the terminating NUL
          ok = N.ptx(prog, reinterpret_cast<char*>(out.code.data())) == 0;
        }
      }
    }
    N.destroy(&prog);
    if (ok) dump(source, name, out);
    return ok;
  }
  Hiprtc& H = hiprtc();
  if (!H.ready) {
    out.log = "hipRTC (hiprtc*.dll / libhiprtc.so) not found: install the AMD HIP runtime / ROCm";
    return false;
  }
  out.arch = dev.arch;
  opts.push_back("--offload-arch=" + dev.arch);
  opts.push_back("-std=c++17");
  void* prog = nullptr;
  if (H.create(&prog, source.c_str(), name.c_str(), 0, nullptr, nullptr) != 0) {
    out.log = "hiprtcCreateProgram failed";
    return false;
  }
  std::vector<const char*> argv;
  for (auto& o : opts) argv.push_back(o.c_str());
  const int rc = H.compile(prog, int(argv.size()), argv.data());
  size_t n = 0;
  if (H.log_size(prog, &n) == 0 && n > 1) {
    out.log.resize(n);
    H.log(prog, &out.log[0]);
    out.log.resize(n - 1);
  }
  bool ok = rc == 0;
  if (ok) {
    size_t sz = 0;
    ok = H.code_size(prog, &sz) == 0 && sz > 0;
    if (ok) {
      out.code.resize(sz);
      ok = H.code(prog, reinterpret_cast<char*>(out.code.data())) == 0;
    }
  }
  H.destroy(&prog);
  if (ok) dump(source, name, out);
  return ok;
}

KernelBlob rtc_blob(const RtcOutput& out, Vendor v, const char* algorithm) {
  return KernelBlob{algorithm, v, out.arch.c_str(), out.code.data(), out.code.size()};
}

}  // namespace xm::gpu
