// AMD backend: the HIP runtime shipped with the GPU driver, loaded at run time.
// Windows: amdhip64_7.dll / amdhip64_6.dll (installed by Adrenalin). Linux: libamdhip64.so (ROCm / amdgpu-pro).
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

#include "gpu/dynlib.hpp"
#include "gpu/gpu.hpp"
#include "hip/hip_device_prop.h"

namespace xm::gpu {

namespace {

using hipError_t = int;
using hipModule_t = void*;
using hipFunction_t = void*;
using hipStream_t = void*;
using hipEvent_t = void*;

struct Hip {
  DynLib lib;
  hipError_t (*hipInit)(unsigned) = nullptr;
  hipError_t (*hipGetDeviceCount)(int*) = nullptr;
  hipError_t (*hipGetDevicePropertiesR0600)(hipDeviceProp_t*, int) = nullptr;
  hipError_t (*hipSetDevice)(int) = nullptr;
  hipError_t (*hipSetDeviceFlags)(unsigned) = nullptr;  // optional
  hipError_t (*hipMalloc)(void**, size_t) = nullptr;
  hipError_t (*hipFree)(void*) = nullptr;
  hipError_t (*hipMemcpyHtoD)(void*, const void*, size_t) = nullptr;
  hipError_t (*hipMemcpyDtoH)(void*, void*, size_t) = nullptr;
  hipError_t (*hipMemsetD8)(void*, unsigned char, size_t) = nullptr;
  hipError_t (*hipModuleLoadData)(hipModule_t*, const void*) = nullptr;
  hipError_t (*hipModuleGetFunction)(hipFunction_t*, hipModule_t, const char*) = nullptr;
  hipError_t (*hipModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                      unsigned, hipStream_t, void**, void**) = nullptr;
  hipError_t (*hipStreamCreate)(hipStream_t*) = nullptr;
  hipError_t (*hipStreamSynchronize)(hipStream_t) = nullptr;
  hipError_t (*hipEventCreateWithFlags)(hipEvent_t*, unsigned) = nullptr;  // optional: blocking waits
  hipError_t (*hipEventRecord)(hipEvent_t, hipStream_t) = nullptr;
  hipError_t (*hipEventSynchronize)(hipEvent_t) = nullptr;
  const char* (*hipGetErrorString)(hipError_t) = nullptr;
  bool ready = false;

  Hip() {
#ifdef _WIN32
    const char* names[] = {"amdhip64_7.dll", "amdhip64_6.dll"};
#else
    // Also the default ROCm prefix: HiveOS and most distros do not put /opt/rocm/lib on the loader path.
    const char* names[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so.7",
                           "/opt/rocm/lib/libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so"};
#endif
    for (const char* n : names)
      if (lib.open(n)) break;
    if (!lib.ok()) return;
    bool all = lib.sym("hipInit", hipInit) && lib.sym("hipGetDeviceCount", hipGetDeviceCount) &&
               lib.sym("hipGetDevicePropertiesR0600", hipGetDevicePropertiesR0600) &&
               lib.sym("hipSetDevice", hipSetDevice) && lib.sym("hipMalloc", hipMalloc) &&
               lib.sym("hipFree", hipFree) && lib.sym("hipMemcpyHtoD", hipMemcpyHtoD) &&
               lib.sym("hipMemcpyDtoH", hipMemcpyDtoH) && lib.sym("hipMemsetD8", hipMemsetD8) &&
               lib.sym("hipModuleLoadData", hipModuleLoadData) &&
               lib.sym("hipModuleGetFunction", hipModuleGetFunction) &&
               lib.sym("hipModuleLaunchKernel", hipModuleLaunchKernel) &&
               lib.sym("hipStreamCreate", hipStreamCreate) &&
               lib.sym("hipStreamSynchronize", hipStreamSynchronize) &&
               lib.sym("hipGetErrorString", hipGetErrorString);
    lib.sym("hipSetDeviceFlags", hipSetDeviceFlags);
    ready = all && hipInit(0) == 0;
  }
};

Hip& hip() {
  static Hip h;
  return h;
}

std::string base_arch(const char* gcn) {
  // "gfx1201:sramecc-:xnack-" -> "gfx1201"
  std::string s = gcn;
  return s.substr(0, s.find(':'));
}

class HipDevice : public Device {
 public:
  explicit HipDevice(DeviceInfo info) : info_(std::move(info)) {
    make_current();
    // Blocking sync: waits park the thread instead of spinning a CPU core (must precede the first stream).
    if (hip().hipSetDeviceFlags) hip().hipSetDeviceFlags(0x4);  // hipDeviceScheduleBlockingSync
    if (hip().hipStreamCreate(&stream_) != 0) stream_ = nullptr;
  }
  const DeviceInfo& info() const override { return info_; }
  bool make_current() override { return check(hip().hipSetDevice(info_.ordinal)); }
  DevPtr alloc(size_t bytes) override {
    void* p = nullptr;
    return check(hip().hipMalloc(&p, bytes)) ? reinterpret_cast<DevPtr>(p) : 0;
  }
  void free(DevPtr p) override {
    if (p) hip().hipFree(reinterpret_cast<void*>(p));
  }
  bool upload(DevPtr dst, const void* src, size_t n) override {
    return check(hip().hipMemcpyHtoD(reinterpret_cast<void*>(dst), src, n));
  }
  bool download(void* dst, DevPtr src, size_t n) override {
    return check(hip().hipMemcpyDtoH(dst, reinterpret_cast<void*>(src), n));
  }
  bool memset(DevPtr dst, int v, size_t n) override {
    return check(hip().hipMemsetD8(reinterpret_cast<void*>(dst), static_cast<unsigned char>(v), n));
  }
  bool load_module(const KernelBlob& blob) override {
    if (blob.vendor != Vendor::Amd || info_.arch != blob.arch) return false;
    funcs_.clear();
    return check(hip().hipModuleLoadData(&module_, blob.data));
  }
  bool launch(const char* kernel, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
              unsigned shared, void** args) override {
    hipFunction_t f = nullptr;
    auto it = funcs_.find(kernel);
    if (it != funcs_.end()) {
      f = it->second;
    } else {
      if (!check(hip().hipModuleGetFunction(&f, module_, kernel))) return false;
      funcs_[kernel] = f;
    }
    return check(hip().hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream_, args, nullptr));
  }
  bool sync() override { return check(hip().hipStreamSynchronize(stream_)); }
  bool wait() override {
    auto& H = hip();
    if (!event_ && H.hipEventCreateWithFlags && H.hipEventRecord && H.hipEventSynchronize)
      H.hipEventCreateWithFlags(&event_, 0x1 | 0x2);  // hipEventBlockingSync | hipEventDisableTiming
    if (!event_) return sync();
    return check(H.hipEventRecord(event_, stream_)) && check(H.hipEventSynchronize(event_));
  }
  std::string last_error() const override { return err_; }

 private:
  bool check(hipError_t e) {
    if (e == 0) return true;
    err_ = std::string("HIP error ") + std::to_string(e) + ": " + hip().hipGetErrorString(e);
    return false;
  }
  DeviceInfo info_;
  hipStream_t stream_ = nullptr;
  hipEvent_t event_ = nullptr;
  hipModule_t module_ = nullptr;
  std::map<std::string, hipFunction_t> funcs_;
  std::string err_;
};

}  // namespace

std::string amd_runtime_name() { return hip().ready ? hip().lib.name() : std::string(); }

std::vector<DeviceInfo> amd_devices() {
  std::vector<DeviceInfo> out;
  if (!hip().ready) return out;
  int n = 0;
  if (hip().hipGetDeviceCount(&n) != 0) return out;
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p;
    std::memset(&p, 0, sizeof p);
    if (hip().hipGetDevicePropertiesR0600(&p, i) != 0) continue;
    DeviceInfo d;
    d.vendor = Vendor::Amd;
    d.ordinal = i;
    d.name = p.name;
    d.arch = base_arch(p.gcnArchName);
    d.memory = p.totalGlobalMem;
    d.pci_bus = p.pciBusID;
    d.compute_units = p.multiProcessorCount;
    if (p.integrated) continue;  // skip APU/iGPU
    out.push_back(d);
  }
  std::sort(out.begin(), out.end(), [](const DeviceInfo& a, const DeviceInfo& b) { return a.pci_bus < b.pci_bus; });
  return out;
}

std::unique_ptr<Device> open_amd(const DeviceInfo& info) { return std::make_unique<HipDevice>(info); }

}  // namespace xm::gpu
