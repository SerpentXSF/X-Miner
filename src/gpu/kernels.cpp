#include "gpu/kernels.hpp"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

namespace xm::gpu {

bool load_kernels(Device& dev, const std::string& algorithm, std::string& why) {
  const auto& info = dev.info();
  // Developer A/B switch: XM_KERNEL_FILE=<algo>_<arch>.<ext> loads that file instead of the embedded blob when the
  // algorithm and the GPU's architecture match its name.
  if (const char* path = std::getenv("XM_KERNEL_FILE")) {
    const std::string p = path, name = p.substr(p.find_last_of("/\\") + 1);
    if (name.rfind(algorithm + "_" + info.arch + ".", 0) == 0) {
      std::ifstream f(p, std::ios::binary);
      static std::vector<unsigned char> data;  // the driver may keep the pointer until the module is loaded
      data.assign(std::istreambuf_iterator<char>(f), {});
      data.push_back(0);  // PTX is loaded as a C string
      const KernelBlob b{algorithm.c_str(), info.vendor, info.arch.c_str(), data.data(), data.size() - 1};
      if (data.size() > 1 && dev.load_module(b)) return true;
      why = "XM_KERNEL_FILE " + p + " did not load";
      return false;
    }
  }
  // Exact matches (AMD code object, NVIDIA cubin) first, then PTX. XM_FORCE_PTX=1 skips the cubins (the driver
  // JIT-compiles the PTX with its own, possibly newer, compiler).
  const bool force_ptx = std::getenv("XM_FORCE_PTX") != nullptr;
  for (int pass = force_ptx ? 1 : 0; pass < 2; ++pass) {
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
