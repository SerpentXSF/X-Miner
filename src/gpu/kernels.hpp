// Kernel blobs embedded from kernels/prebuilt/ at build time.
#pragma once

#include <string>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm::gpu {

const std::vector<KernelBlob>& kernel_blobs();

// Loads the best blob of `algorithm` into the device: exact AMD arch, else the AMD family-generic one; exact NVIDIA
// cubin, else NVIDIA PTX.
// Returns false (with the reason in `why`) if no blob fits.
bool load_kernels(Device& dev, const std::string& algorithm, std::string& why);

// The same, added next to the device's loaded module (Device::add_module): exact architecture matches (or the AMD
// family-generic target) only.
bool add_kernels(Device& dev, const std::string& algorithm, std::string& why);

// AMD family-generic code object target that runs on `arch` (LLVM generic targets, code object v6), or "" if none:
// gfx1010-1013 -> gfx10-1-generic, gfx1030-1036 -> gfx10-3-generic, gfx1100-1103 / gfx1150-1153 -> gfx11-generic,
// gfx1200-1201 -> gfx12-generic. Lets the embedded kernels run on chips without their own code object (APUs, BC-250).
std::string amd_generic_arch(const std::string& arch);

// Architectures that have a blob for `algorithm`, e.g. {"gfx1201", "sm_86", "compute_80"}.
std::vector<std::string> kernel_archs(const std::string& algorithm);

}  // namespace xm::gpu
