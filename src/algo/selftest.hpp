// --selftest: run each algorithm's GPU kernels against its CPU reference on every selected device.
#pragma once

#include <vector>

#include "gpu/gpu.hpp"

namespace xm {

// Returns the number of failed checks (0 = everything matched).
int run_selftest(const std::vector<gpu::DeviceInfo>& devices, const std::vector<size_t>& ids);  // ids: --list-devices numbers

}  // namespace xm
