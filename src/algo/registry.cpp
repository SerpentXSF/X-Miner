#include "algo/registry.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "gpu/kernels.hpp"

namespace xm {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

const std::vector<AlgorithmInfo>& algorithms() {
  static const std::vector<AlgorithmInfo> list = {
      {"pearlhash", "pearl,pearlpow,prl", "Pearl (PRL)", "pearl", true},
      {"noid", "parano1d,poseidon2b", "Parano1d (NOID)", "noid", true},
      {"blake2b", "rdts,btcb2,blake2b-rdts", "Bitcoin Knots BLAKE2b (RDTS)", "blake2b", true},
  };
  return list;
}

const AlgorithmInfo* find_algorithm(const std::string& name) {
  std::string n = lower(name);
  for (const auto& a : algorithms()) {
    if (n == a.name) return &a;
    std::string al = std::string(",") + a.aliases + ",";
    if (al.find("," + n + ",") != std::string::npos) return &a;
  }
  return nullptr;
}

bool supports(const AlgorithmInfo& algo, const gpu::DeviceInfo& dev) {
  if (!algo.implemented) return false;
  // Embedded kernels decide: an exact AMD code object or NVIDIA cubin, or NVIDIA PTX for an equal/older arch.
  for (const auto& a : gpu::kernel_archs(algo.kernel)) {
    if (a == dev.arch) return true;
    if (dev.vendor == gpu::Vendor::Nvidia && a.rfind("compute_", 0) == 0 && dev.arch.rfind("sm_", 0) == 0 &&
        std::stoi(a.substr(8)) <= std::stoi(dev.arch.substr(3)))
      return true;
  }
  return false;
}

}  // namespace xm
