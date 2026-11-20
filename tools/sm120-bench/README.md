# sm_120 (RTX 50) Pearl benchmarks

Developer tools behind docs/rtx5060ti/FINDINGS.md. Build with nvcc from a CUDA 13 toolkit, run on an idle GPU.

- `h2.cu`: bare harness for the miner's own GEMM kernels (kernels/pearl/pearl.cu). Prints the rate as a share of the
  tensor-core limit, a checksum of the debug transcripts and of the found[] hits (compare across builds).
  `nvcc -O3 -std=c++17 -arch=sm_120 -I ../../kernels -DH_THREADS=256 -DKERN=xm_pearl_gemm_w8 -DH_PRESWZ -DXM_PEARL_PRESWZ h2.cu -o h2 && ./h2 2304 8192 2048 400 2745`
  (`-DH_THREADS=288` and no other flags for the stock `xm_pearl_gemm`). The last numeric argument is the SM clock in MHz.
- `mb.cu` .. `mb9.cu`: microbenchmarks (MMA scheduling and instruction costs, operand loads, cp.async throughput, mbarrier
  cost, MMA shapes, in-SM operand generation). Each file's header says what it measures and how to build it.
- `pw.py <seconds> <warmup> <cmd...>`: mean board power of a command at thermal steady state (nvidia-smi).
- `reasons.py`: per-instruction stall reasons from an Nsight Compute report (`ncu --set full ...`).
