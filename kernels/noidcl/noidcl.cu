// X-Miner Parano1d (NOID) search kernel with the hardware carry-less multiply (CLMAD, PTX ISA 9.3: CUDA 13.3+ and a
// driver that supports it, sm_80+). NVIDIA only; built by tools/build_kernels.ps1 with -Nvcc13. The miner loads it
// first and falls back to kernels/noid when the driver rejects it (src/algo/noid/noid_gpu.cpp).
// RTX 3060 at 99 W: 9.80 -> 18.0 MH/s with 8 of the 9 K-products on CLMAD (mask 0x0FF; 9 of 9: 16.97, 7: 16.96,
// 6: 15.3): the remaining one on IMAD.WIDE runs beside the CLMAD pipe.
#define NOID_CLMAD
#ifndef NOID_CLMAD_MASK
#define NOID_CLMAD_MASK 0x0FF
#endif
#include "noid/noid.cu"
