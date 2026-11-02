// CUDA shell for the generated sm_86 Nexa kernels (see ngen.py): kernels/nexa/nexa.cu in its three-kernel form with
// the runtime defaults the prebuilt cubin is made for; xm_nexa_b's marker block is replaced by generated code.
#define TPB 256
#define W 20
#define NW 13
#define MINB 2
#define XM_NEXA_SPLIT
#define XM_NEXA_GEN_SHELL
#include "../../kernels/nexa/nexa.cu"
