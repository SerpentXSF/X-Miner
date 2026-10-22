// Minimal cross-platform shared-library loading.
#pragma once

#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
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

}  // namespace xm::gpu
