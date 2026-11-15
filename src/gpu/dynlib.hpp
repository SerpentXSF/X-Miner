// Minimal cross-platform shared-library loading.
#pragma once

#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <vector>
#endif

namespace xm::gpu {

class DynLib {
 public:
  bool open(const char* name) {
#ifdef _WIN32
    h_ = LoadLibraryA(name);
#else
    h_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
    if (h_) name_ = name;
    return h_ != nullptr;
  }
  template <class F>
  bool sym(const char* s, F& out) {
#ifdef _WIN32
    out = reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h_), s)));
#else
    out = reinterpret_cast<F>(dlsym(h_, s));
#endif
    return out != nullptr;
  }
  bool ok() const { return h_ != nullptr; }
  const std::string& name() const { return name_; }

 private:
  void* h_ = nullptr;  // never closed: GPU runtimes must outlive every worker thread
  std::string name_;
};

#ifndef _WIN32
// The ROCm HIP runtime shipped in <executable dir>/hip/ by the Linux and HiveOS packages (HiveOS installs only ROCm's
// OpenCL, no HIP). Its dependencies are opened by full path first, so the loader reuses them for libamdhip64 /
// libhiprtc instead of an older ROCm on the loader path (HiveOS: ROCm 5.2 in /opt/rocm/lib). Used only when no
// system HIP is found.
inline std::string bundle_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  const std::string exe(buf, size_t(n));
  return exe.substr(0, exe.rfind('/') + 1) + "hip/";
}

inline bool open_bundled_rocm(DynLib& lib, const char* soname) {
  static const std::vector<DynLib> deps = [] {
    std::vector<DynLib> v;
    for (const char* d : {"librocprofiler-register.so.0", "libhsa-runtime64.so.1", "libamd_comgr.so.3",
                          "libhiprtc-builtins.so.6"}) {
      DynLib l;
      if (l.open((bundle_dir() + d).c_str())) v.push_back(l);
    }
    return v;
  }();
  const std::string path = bundle_dir() + soname;
  if (lib.open(path.c_str())) return true;
  // Shipped but not loadable (usually a missing system library: libnuma, libelf, libdrm_amdgpu): say why.
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    if (!dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL))
      std::fprintf(stderr, "  bundled HIP runtime %s did not load: %s\n", soname, dlerror());
  }
  return false;
}
#endif

}  // namespace xm::gpu
