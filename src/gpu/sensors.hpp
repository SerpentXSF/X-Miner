// GPU power, temperature, fan and clock readings. Every source is optional: missing values are -1.
//   AMD Windows: ADL PMLog (atiadlxx.dll, installed with Adrenalin), matched to devices by PCI bus.
//   AMD Linux:   amdgpu sysfs hwmon.
//   NVIDIA:      NVML (nvml.dll / libnvidia-ml.so.1, installed with the driver).
#pragma once

#include <memory>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm::gpu {

struct Reading {
  double power_w = -1, temp_c = -1, fan_pct = -1, core_mhz = -1;
};

class Sensors {
 public:
  explicit Sensors(const std::vector<DeviceInfo>& devices);
  ~Sensors();
  Reading read(size_t index);  // index into the devices given at construction

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xm::gpu
