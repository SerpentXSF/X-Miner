// The algorithms X-Miner knows, and which GPU architectures each one has a verified kernel for.
#pragma once

#include <string>
#include <vector>

#include "gpu/gpu.hpp"

namespace xm {

struct AlgorithmInfo {
  const char* name;         // command-line name
  const char* aliases;      // comma-separated alternative names accepted on the command line
  const char* coin;         // "Pearl (PRL)"
  const char* kernel;       // kernel blob name: kernels/<kernel>/ -> kernels/prebuilt/<kernel>_<arch>.*
  bool implemented;         // false until the CPU reference and at least one GPU kernel pass known-answer tests
};

const std::vector<AlgorithmInfo>& algorithms();
const AlgorithmInfo* find_algorithm(const std::string& name);  // accepts aliases, case-insensitive
bool supports(const AlgorithmInfo& algo, const gpu::DeviceInfo& dev);

}  // namespace xm
