// X-Miner NexaPow comb table builder, prebuilt for every GPU target (kernels/prebuilt/nexatab_*): the field and point
// code of nexa/nexa.cu with only its xm_nexa_table kernel (TPB 256; the window width is a launch argument). The
// mining kernels themselves stay compiled at run time.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>
#define TPB 256
#define W 24  // (unused by the table kernel; the rest of nexa.cu expects it)
#define NW 11
#define MINB 1
#define XM_NEXA_TABLE_ONLY
#include "nexa/nexa.cu"
