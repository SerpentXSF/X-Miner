#include "algo/registry.hpp"

#include <algorithm>
#include <cctype>

namespace xm {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

const std::vector<AlgorithmInfo>& algorithms() {
  static const std::vector<AlgorithmInfo> list = {
      {"pearlhash", "pearl,pearlpow,prl", "Pearl (PRL)", false, {}},
      {"noid", "parano1d,poseidon2b", "Parano1d (NOID)", false, {}},
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
  return std::find(algo.archs.begin(), algo.archs.end(), dev.arch) != algo.archs.end();
}

}  // namespace xm
