#include "gpu/kernels.hpp"

namespace xm::gpu {

bool load_kernels(Device& dev, const std::string& algorithm, std::string& why) {
  const auto& info = dev.info();
  // Exact matches (AMD code object, NVIDIA cubin) first, then PTX.
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& b : kernel_blobs()) {
      if (algorithm != b.algorithm || b.vendor != info.vendor) continue;
      bool ptx = std::string(b.arch).rfind("compute_", 0) == 0;
      if ((pass == 0) == ptx) continue;
      if (dev.load_module(b)) return true;
    }
  }
  why = "no " + algorithm + " kernel for " + info.arch;
  if (!dev.last_error().empty()) why += " (" + dev.last_error() + ")";
  return false;
}

std::vector<std::string> kernel_archs(const std::string& algorithm) {
  std::vector<std::string> out;
  for (const auto& b : kernel_blobs())
    if (algorithm == b.algorithm) out.push_back(b.arch);
  return out;
}

}  // namespace xm::gpu
