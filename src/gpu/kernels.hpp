// Kernel blobs embedded from kernels/prebuilt/ at build time.
#pragma once

#include <string>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm::gpu {

const std::vector<KernelBlob>& kernel_blobs();

// Loads the best blob of `algorithm` into the device: exact AMD arch, else exact NVIDIA cubin, else NVIDIA PTX.
// Returns false (with the reason in `why`) if no blob fits.
bool load_kernels(Device& dev, const std::string& algorithm, std::string& why);

// Architectures that have a blob for `algorithm`, e.g. {"gfx1201", "sm_86", "compute_80"}.
std::vector<std::string> kernel_archs(const std::string& algorithm);

}  // namespace xm::gpu
