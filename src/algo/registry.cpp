#include "algo/registry.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "gpu/kernels.hpp"
#include "gpu/rtc.hpp"

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
      {"quantus", "qtc,qpow,quantus-network", "Quantus (QTC)", "quantus", true},
      {"kawpow", "rvn,ravencoin", "Ravencoin (RVN)", "@rtc", true},
      {"firopow", "firo", "Firo (FIRO)", "@rtc", true},
      {"progpow-epic", "epic,epiccash,progpow092", "Epic Cash (EPIC)", "@rtc", true},
      {"progpowz", "zano,progpow-zano", "Zano (ZANO)", "@rtc", true},
      {"etchash", "etc,ethash-etc", "Ethereum Classic (ETC)", "@rtc", true},
      {"octopus", "cfx,conflux", "Conflux (CFX)", "@rtc", true},
      {"fishhash", "iron,ironfish", "Iron Fish (IRON)", "@rtc", true},
      {"karlsenhashv2", "kls,karlsen,karlsenhash", "Karlsen (KLS)", "@rtc", true},
      {"autolykos2", "ergo,erg,autolykos", "Ergo (ERG)", "@rtc", true},
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
  // Run-time compiled (ProgPoW family): any RDNA4 / RTX 30+ GPU whose runtime compiler is installed.
  if (std::string(algo.kernel) == "@rtc")
    return !gpu::rtc_library(dev.vendor).empty() &&
           (dev.vendor == gpu::Vendor::Nvidia ? std::stoi(dev.arch.substr(3)) >= 61 : dev.arch.rfind("gfx1", 0) == 0);
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
