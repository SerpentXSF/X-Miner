// Run-time compilation of generated kernels (ProgPoW programs change every few blocks): NVRTC on NVIDIA, hipRTC on
// AMD. Both libraries are loaded at run time like the GPU runtimes; the NVIDIA one ships next to x-miner (the
// driver does not include it), the AMD one comes with the HIP runtime / ROCm.
#pragma once

#include <string>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm::gpu {

struct RtcOutput {
  std::vector<unsigned char> code;  // cubin / PTX (NUL-terminated) / AMD code object
  std::string arch;                 // "sm_86", "compute_90" or "gfx1201": what load_module() expects
  std::string log;                  // compiler messages (errors and warnings)
};

// Compiles `source` (CUDA / HIP C++, extern "C" kernels) for `dev`. `options` are passed to the compiler as given.
bool rtc_compile(const DeviceInfo& dev, const std::string& source, const std::string& name,
                 const std::vector<std::string>& options, RtcOutput& out);
std::string rtc_library(Vendor v);  // loaded library name, "" if none was found
// A KernelBlob view of a compiled output for Device::load_module (valid while `out` lives).
KernelBlob rtc_blob(const RtcOutput& out, Vendor v, const char* algorithm);

}  // namespace xm::gpu
