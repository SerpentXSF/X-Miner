// Vendor-neutral GPU access. AMD (HIP) and NVIDIA (CUDA driver API) runtimes are loaded at run time, so the
// executable needs no GPU SDK and runs on AMD-only, NVIDIA-only and mixed rigs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xm::gpu {

enum class Vendor { Amd, Nvidia };

struct DeviceInfo {
  Vendor vendor;
  int ordinal = 0;          // index within its vendor runtime
  std::string name;         // "AMD Radeon RX 9070 XT"
  std::string arch;         // "gfx1201", "sm_86"
  size_t memory = 0;        // bytes
  int pci_bus = -1;
  int compute_units = 0;
};

using DevPtr = uint64_t;  // opaque device pointer (hipDeviceptr / CUdeviceptr)

// A compiled kernel module for one device architecture, embedded in the executable.
struct KernelBlob {
  const char* algorithm;   // "pearlhash"
  Vendor vendor;
  const char* arch;        // "gfx1201", "sm_86" (cubin) or "compute_80" (PTX, JIT-compiled)
  const unsigned char* data;
  size_t size;
};

// One opened device with its own context/stream. Not thread-safe: one worker thread per device.
class Device {
 public:
  virtual ~Device() = default;
  virtual const DeviceInfo& info() const = 0;
  virtual bool make_current() = 0;
  virtual DevPtr alloc(size_t bytes) = 0;          // 0 on failure
  virtual void free(DevPtr p) = 0;
  virtual bool upload(DevPtr dst, const void* src, size_t bytes) = 0;
  virtual bool download(void* dst, DevPtr src, size_t bytes) = 0;
  virtual bool memset(DevPtr dst, int value, size_t bytes) = 0;
  // Loads a module from a blob; returns false if the blob does not fit this device.
  virtual bool load_module(const KernelBlob& blob) = 0;
  // Launch a kernel from the loaded module. args are pointers to each argument value, like cuLaunchKernel.
  virtual bool launch(const char* kernel, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                      unsigned bz, unsigned shared_bytes, void** args) = 0;
  virtual bool sync() = 0;
  virtual std::string last_error() const = 0;
};

// Discovery. Each list is empty when that vendor's runtime is not installed. GPU ids shown to users are the
// position in all_devices(): AMD first, then NVIDIA, each in PCI bus order.
std::vector<DeviceInfo> amd_devices();
std::vector<DeviceInfo> nvidia_devices();
std::vector<DeviceInfo> all_devices();
std::string amd_runtime_name();     // "amdhip64_7.dll" or "" if missing
std::string nvidia_runtime_name();  // "nvcuda.dll (driver 12.8)" or ""

std::unique_ptr<Device> open(const DeviceInfo& info);

}  // namespace xm::gpu
