// Pipeline smoke test: compiled for AMD (clang -x hip) and NVIDIA (nvcc) from this one file, embedded in the
// executable, loaded through the runtime-loaded HIP / CUDA driver APIs, and checked by --selftest.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

extern "C" __global__ void xm_smoke(const unsigned* a, const unsigned* b, unsigned* out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i] + (i ^ 0x5a5a5a5au);
}
