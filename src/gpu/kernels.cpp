#include "gpu/kernels.hpp"

#include <algorithm>
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
  // Exact matches (AMD code object, NVIDIA cubin) first, then the AMD family-generic code object, then PTX. XM_FORCE_PTX=1 skips the cubins (the driver
  // JIT-compiles the PTX with its own, possibly newer, compiler).
  // PTX: the newest version the GPU can run first (e.g. compute_80 before compute_61 on an A100: the older one is a
  // different kernel design for GPUs without the newer instructions).
  const bool force_ptx = std::getenv("XM_FORCE_PTX") != nullptr;
  const std::string generic = info.vendor == Vendor::Amd ? amd_generic_arch(info.arch) : "";
  for (int pass = force_ptx ? 1 : 0; pass < 2; ++pass) {
    std::vector<const KernelBlob*> cand;
    for (const auto& b : kernel_blobs()) {
      if (algorithm != b.algorithm || b.vendor != info.vendor) continue;
      bool ptx = std::string(b.arch).rfind("compute_", 0) == 0;
      if ((pass == 0) == ptx) continue;
      if (info.vendor == Vendor::Amd && b.arch != info.arch && b.arch != generic) continue;
      cand.push_back(&b);
    }
    if (pass == 0)  // the exact code object before the generic one
      std::stable_partition(cand.begin(), cand.end(), [&](const KernelBlob* b) { return b->arch == info.arch; });
    if (pass == 1)
      std::stable_sort(cand.begin(), cand.end(), [](const KernelBlob* x, const KernelBlob* y) {
        return std::atoi(x->arch + 8) > std::atoi(y->arch + 8);
      });
    for (const KernelBlob* b : cand)
      if (dev.load_module(*b)) return true;
  }
  why = "no " + algorithm + " kernel for " + info.arch;
  if (!dev.last_error().empty()) why += " (" + dev.last_error() + ")";
  return false;
}

bool add_kernels(Device& dev, const std::string& algorithm, std::string& why) {
  const auto& info = dev.info();
  if (const char* path = std::getenv("XM_KERNEL_FILE")) {  // developer A/B switch, as in load_kernels()
    const std::string p = path, name = p.substr(p.find_last_of("/\\") + 1);
    if (name.rfind(algorithm + "_" + info.arch + ".", 0) == 0) {
      std::ifstream f(p, std::ios::binary);
      static std::vector<unsigned char> data;
      data.assign(std::istreambuf_iterator<char>(f), {});
      const KernelBlob b{algorithm.c_str(), info.vendor, info.arch.c_str(), data.data(), data.size()};
      if (!data.empty() && dev.add_module(b)) return true;
      why = "XM_KERNEL_FILE " + p + " did not load";
      return false;
    }
  }
  const std::string generic = info.vendor == Vendor::Amd ? amd_generic_arch(info.arch) : "";
  for (const std::string& arch : {info.arch, generic})
    for (const auto& b : kernel_blobs())
      if (!arch.empty() && algorithm == b.algorithm && b.vendor == info.vendor && arch == b.arch && dev.add_module(b))
        return true;
  why = "no " + algorithm + " kernel for " + info.arch;
  return false;
}

std::string amd_generic_arch(const std::string& arch) {
  // "gfx" + 4 digits; the family generic targets of LLVM's AMDGPU backend (see AMDGPUUsage, "Generic Processors")
  if (arch.size() != 7 || arch.rfind("gfx1", 0) != 0) return "";
  const std::string fam = arch.substr(0, 6);  // gfx101, gfx103, gfx110, gfx115, gfx120
  const char last = arch[6];
  if (fam == "gfx101" && last >= '0' && last <= '3') return "gfx10-1-generic";
  if (fam == "gfx103" && last >= '0' && last <= '6') return "gfx10-3-generic";
  if ((fam == "gfx110" || fam == "gfx115") && last >= '0' && last <= '3') return "gfx11-generic";
  if (fam == "gfx120" && last >= '0' && last <= '1') return "gfx12-generic";
  return "";
}

std::vector<std::string> kernel_archs(const std::string& algorithm) {
  std::vector<std::string> out;
  for (const auto& b : kernel_blobs())
    if (algorithm == b.algorithm) out.push_back(b.arch);
  return out;
}

}  // namespace xm::gpu
