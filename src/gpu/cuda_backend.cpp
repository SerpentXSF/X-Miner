// NVIDIA backend: the CUDA driver API (nvcuda.dll / libcuda.so.1), loaded at run time.
// Only the stable driver API is used, so no CUDA toolkit is needed on the mining rig.
#include <algorithm>
#include <cstdlib>
#include <map>

#include "gpu/dynlib.hpp"
#include "gpu/gpu.hpp"

namespace xm::gpu {

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUevent = void*;
using CUdeviceptr = unsigned long long;

enum : int {
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
};

struct Cuda {
  DynLib lib;
  CUresult (*cuInit)(unsigned) = nullptr;
  CUresult (*cuDriverGetVersion)(int*) = nullptr;
  CUresult (*cuDeviceGetCount)(int*) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
  CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
  CUresult (*cuDeviceTotalMem_v2)(size_t*, CUdevice) = nullptr;
  CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
  CUresult (*cuDevicePrimaryCtxSetFlags)(CUdevice, unsigned) = nullptr;  // optional
  CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
  CUresult (*cuMemAlloc_v2)(CUdeviceptr*, size_t) = nullptr;
  CUresult (*cuMemFree_v2)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t) = nullptr;
  CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t) = nullptr;
  CUresult (*cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t) = nullptr;
  CUresult (*cuModuleLoadData)(CUmodule*, const void*) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
  CUresult (*cuStreamCreate)(CUstream*, unsigned) = nullptr;
  CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
  CUresult (*cuEventCreate)(CUevent*, unsigned) = nullptr;  // optional: blocking waits
  CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*cuEventSynchronize)(CUevent) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
  CUresult (*cuFuncSetAttribute)(CUfunction, int, int) = nullptr;  // optional: > 48 KB dynamic shared memory
  bool ready = false;
  int driver_version = 0;

  Cuda() {
#ifdef _WIN32
    lib.open("nvcuda.dll");
#else
    if (!lib.open("libcuda.so.1")) lib.open("libcuda.so");
#endif
    if (!lib.ok()) return;
    bool all = lib.sym("cuInit", cuInit) && lib.sym("cuDriverGetVersion", cuDriverGetVersion) &&
               lib.sym("cuDeviceGetCount", cuDeviceGetCount) && lib.sym("cuDeviceGet", cuDeviceGet) &&
               lib.sym("cuDeviceGetName", cuDeviceGetName) && lib.sym("cuDeviceTotalMem_v2", cuDeviceTotalMem_v2) &&
               lib.sym("cuDeviceGetAttribute", cuDeviceGetAttribute) &&
               lib.sym("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain) &&
               lib.sym("cuCtxSetCurrent", cuCtxSetCurrent) && lib.sym("cuMemAlloc_v2", cuMemAlloc_v2) &&
               lib.sym("cuMemFree_v2", cuMemFree_v2) && lib.sym("cuMemcpyHtoD_v2", cuMemcpyHtoD_v2) &&
               lib.sym("cuMemcpyDtoH_v2", cuMemcpyDtoH_v2) && lib.sym("cuMemsetD8_v2", cuMemsetD8_v2) &&
               lib.sym("cuModuleLoadData", cuModuleLoadData) &&
               lib.sym("cuModuleGetFunction", cuModuleGetFunction) && lib.sym("cuLaunchKernel", cuLaunchKernel) &&
               lib.sym("cuStreamCreate", cuStreamCreate) && lib.sym("cuStreamSynchronize", cuStreamSynchronize) &&
               lib.sym("cuGetErrorString", cuGetErrorString);
    lib.sym("cuFuncSetAttribute", cuFuncSetAttribute);
    if (!lib.sym("cuDevicePrimaryCtxSetFlags_v2", cuDevicePrimaryCtxSetFlags))
      lib.sym("cuDevicePrimaryCtxSetFlags", cuDevicePrimaryCtxSetFlags);
    lib.sym("cuEventCreate", cuEventCreate);
    lib.sym("cuEventRecord", cuEventRecord);
    lib.sym("cuEventSynchronize", cuEventSynchronize);
    ready = all && cuInit(0) == 0;
    if (ready) cuDriverGetVersion(&driver_version);
  }
};

Cuda& cuda() {
  static Cuda c;
  return c;
}

class CudaDevice : public Device {
 public:
  explicit CudaDevice(DeviceInfo info) : info_(std::move(info)) {
    // Blocking sync: waits park the thread instead of spinning a CPU core (before the context is first used).
    if (cuda().cuDevicePrimaryCtxSetFlags && !std::getenv("XM_SPIN_WAIT") && cuda().cuDeviceGet(&dev_, info_.ordinal) == 0)
      cuda().cuDevicePrimaryCtxSetFlags(dev_, 0x4);  // CU_CTX_SCHED_BLOCKING_SYNC
    if (check(cuda().cuDeviceGet(&dev_, info_.ordinal)) && check(cuda().cuDevicePrimaryCtxRetain(&ctx_, dev_)) &&
        make_current())
      check(cuda().cuStreamCreate(&stream_, 0));
  }
  const DeviceInfo& info() const override { return info_; }
  bool make_current() override { return check(cuda().cuCtxSetCurrent(ctx_)); }
  DevPtr alloc(size_t bytes) override {
    CUdeviceptr p = 0;
    return check(cuda().cuMemAlloc_v2(&p, bytes)) ? p : 0;
  }
  void free(DevPtr p) override {
    if (p) cuda().cuMemFree_v2(p);
  }
  bool upload(DevPtr dst, const void* src, size_t n) override { return check(cuda().cuMemcpyHtoD_v2(dst, src, n)); }
  bool download(void* dst, DevPtr src, size_t n) override { return check(cuda().cuMemcpyDtoH_v2(dst, src, n)); }
  bool memset(DevPtr dst, int v, size_t n) override {
    return check(cuda().cuMemsetD8_v2(dst, static_cast<unsigned char>(v), n));
  }
  bool load_module(const KernelBlob& blob) override {
    if (blob.vendor != Vendor::Nvidia) return false;
    // A cubin must match the architecture exactly; PTX ("compute_XY") is JIT-compiled for any newer GPU.
    std::string a = blob.arch;
    if (a.rfind("sm_", 0) == 0 && a != info_.arch) return false;
    if (a.rfind("compute_", 0) == 0 && std::stoi(a.substr(8)) > std::stoi(info_.arch.substr(3))) return false;
    funcs_.clear();
    big_shared_.clear();
    return check(cuda().cuModuleLoadData(&module_, blob.data));
  }
  bool launch(const char* kernel, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
              unsigned shared, void** args) override {
    CUfunction f = nullptr;
    auto it = funcs_.find(kernel);
    if (it != funcs_.end()) {
      f = it->second;
    } else {
      if (!check(cuda().cuModuleGetFunction(&f, module_, kernel))) return false;
      funcs_[kernel] = f;
    }
    if (shared > 48 * 1024) {  // opt in to more than the default 48 KB of dynamic shared memory (once per size)
      unsigned& set = big_shared_[f];
      if (set < shared) {
        const int kMaxDynamicSharedSizeBytes = 8;  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        if (!cuda().cuFuncSetAttribute ||
            !check(cuda().cuFuncSetAttribute(f, kMaxDynamicSharedSizeBytes, static_cast<int>(shared))))
          return false;
        set = shared;
      }
    }
    return check(cuda().cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream_, args, nullptr));
  }
  bool sync() override { return check(cuda().cuStreamSynchronize(stream_)); }
  bool wait() override {
    auto& C = cuda();
    if (!event_ && C.cuEventCreate && C.cuEventRecord && C.cuEventSynchronize)
      C.cuEventCreate(&event_, 0x1 | 0x2);  // CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING
    static const bool spin = std::getenv("XM_SPIN_WAIT") != nullptr;  // developer A/B switch
    if (!event_ || spin) return sync();
    return check(C.cuEventRecord(event_, stream_)) && check(C.cuEventSynchronize(event_));
  }
  std::string last_error() const override { return err_; }

 private:
  bool check(CUresult e) {
    if (e == 0) return true;
    const char* s = nullptr;
    cuda().cuGetErrorString(e, &s);
    err_ = std::string("CUDA error ") + std::to_string(e) + ": " + (s ? s : "?");
    return false;
  }
  DeviceInfo info_;
  CUdevice dev_ = 0;
  CUcontext ctx_ = nullptr;
  CUstream stream_ = nullptr;
  CUevent event_ = nullptr;
  CUmodule module_ = nullptr;
  std::map<std::string, CUfunction> funcs_;
  std::map<CUfunction, unsigned> big_shared_;  // dynamic shared memory opted in per function
  std::string err_;
};

}  // namespace

std::string nvidia_runtime_name() {
  if (!cuda().ready) return {};
  int v = cuda().driver_version;
  return cuda().lib.name() + " (CUDA " + std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10) + ")";
}

std::vector<DeviceInfo> nvidia_devices() {
  std::vector<DeviceInfo> out;
  if (!cuda().ready) return out;
  int n = 0;
  if (cuda().cuDeviceGetCount(&n) != 0) return out;
  for (int i = 0; i < n; ++i) {
    CUdevice dev = 0;
    if (cuda().cuDeviceGet(&dev, i) != 0) continue;
    char name[256] = {};
    cuda().cuDeviceGetName(name, sizeof name, dev);
    int major = 0, minor = 0, bus = -1, sms = 0;
    cuda().cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuda().cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuda().cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    cuda().cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    DeviceInfo d;
    d.vendor = Vendor::Nvidia;
    d.ordinal = i;
    d.name = name;
    d.arch = "sm_" + std::to_string(major * 10 + minor);
    cuda().cuDeviceTotalMem_v2(&d.memory, dev);
    d.pci_bus = bus;
    d.compute_units = sms;
    out.push_back(d);
  }
  std::sort(out.begin(), out.end(), [](const DeviceInfo& a, const DeviceInfo& b) { return a.pci_bus < b.pci_bus; });
  return out;
}

std::unique_ptr<Device> open_nvidia(const DeviceInfo& info) { return std::make_unique<CudaDevice>(info); }

std::unique_ptr<Device> open_amd(const DeviceInfo& info);

std::vector<DeviceInfo> all_devices() {
  auto a = amd_devices();
  auto n = nvidia_devices();
  a.insert(a.end(), n.begin(), n.end());
  return a;
}

std::unique_ptr<Device> open(const DeviceInfo& info) {
  return info.vendor == Vendor::Amd ? open_amd(info) : open_nvidia(info);
}

}  // namespace xm::gpu
