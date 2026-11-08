// X-Miner Parano1d (NOID) search kernel in the 64-bit tower basis (kernels/noid/noid_t64.h): NVIDIA only, built with
// CUDA 13.3+ (PTX clmad, 64 x 64-bit carry-less multiply). The engine falls back to kernels/noidcl, then
// kernels/noid, when the driver rejects it.
#define NOID_CLMAD
#define NOID_T64
#include "noid/noid.cu"
