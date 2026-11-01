# qgen: generated SASS for Quantus on sm_86 (RTX 30)

`kernels/prebuilt/quantus_sm_86.cubin` is not compiled from `kernels/quantus/quantus.cu`. It is the same search
kernel with the two Poseidon2 permutations written as scheduled sm_86 SASS by `qgen.py`, assembled with
CuAssembler. On an RTX 3060 at 170 W this gives 171.8 MH/s, where the ptxas build gives 151.6 MH/s. The other NVIDIA
targets (sm_89, sm_120, PTX) and AMD still use the compiled CUDA/HIP kernel.

This is build-time tooling only. Releases ship just the miner packages; the cubin is embedded in `x-miner`.

## What it does

1. Writes `qgen_table.h` (round constants, the matrix diagonal and the diagonal times 2^32 mod p, as `kQgen[238]`).
2. nvcc compiles `qshell.cu`, which is `quantus.cu`'s helpers plus a search kernel whose permutations are two
   marker XORs. The nonce setup and the target check stay compiled CUDA.
3. CuAssembler turns the cubin into text. `qgen.py` replaces the markers with the generated block:
   - it builds the state from the nonce words and `job.pre`;
   - it runs permutation 3's tail, the absorb, and permutation 4, sharing one copy of the tail code between the two;
   - permutation 4's last round computes only s0;
   - it leaves s0 where the CUDA check expects it.

   Registers come from those the shell does not keep live across the block, and the register count is set to 64.
4. **Check**: `emu.py` runs the generated block, in its final order with its loops and branches, for 64 inputs
   (three edge cases plus random ones). Each result is compared with the Python reference (`ref_miner`). On any
   mismatch the cubin is not written.
5. CuAssembler assembles the result. `cuasm_fix.py` restores the exact encodings of the shell's memory instructions,
   which CuAssembler's sm_86 tables encode with a wrong bit.

The generated code:
- **Multiplies:** each multiply is split into a head (four `IMAD.WIDE`) and a tail (the reduction, 8 integer ops).
  One multiply's tail is interleaved with the next one's head, using alternating register and predicate sets.
- **Linear layers:** 96-bit sums with three-input adds (`IADD3` with two carry-outs). Six state elements keep their
  low word in the odd register of their pair (`SWAP`), so few adds read three registers of one bank.
- **Pipe placement:** carry-only adds go to the multiply pipe (`IMAD.X`) or the ALU (`IADD3.X`) as measured best;
  the comments in `qgen.py` list what was tried.
- **Loops:** the full and partial rounds are loops, with round constants loaded by `LDC` (the straight-line version
  stalled on instruction fetch).
- **Scheduling:** `sass.py` list-schedules each block. It then sets stall counts from measured GA106 latencies:
  - 4 cycles within a pipe, 5 across the ALU and multiply pipes;
  - 6 cycles from WIDE to WIDE, and for a WIDE carry read on the multiply pipe;
  - 13 cycles for `IMAD.HI`, and from `ISETP` to a branch.

  Latency state is padded at block boundaries.

## Requirements

- CUDA toolkit (nvcc, cuobjdump), and on Windows the MSVC host compiler.
- Python 3.8+.
- [CuAssembler](https://github.com/cloudcores/CuAssembler) (MIT) at commit `96a9f72`, with `cuassembler_sm86.patch`
  applied. The patch adds sm_86 encodings that its instruction tables lack:
  - `IADD3` with two carry-outs;
  - `IADD3.X` with two carries in and out;
  - negated predicate operands.

  CuAssembler needs `sympy` and `pyelftools`.

```bash
git clone https://github.com/cloudcores/CuAssembler && cd CuAssembler
git checkout 96a9f72
git apply ../X-Miner/tools/qgen/cuassembler_sm86.patch
python -m venv .venv && .venv/Scripts/pip install sympy pyelftools
```

## Build

`tools/build_kernels.ps1 -CuAssembler <path>` (or `XM_CUASSEMBLER=<path>`) regenerates the cubin together with the
other kernels. Without a CuAssembler path, the committed cubin is kept. Directly:

```bash
python tools/qgen/qgen.py --cuassembler <CuAssembler> --python <CuAssembler>/.venv/Scripts/python.exe --ccbin <cl.exe dir>
```

`--keep <dir>` keeps the intermediate files: the table, the shell cubin and its text, and the generated text.
`--set NAME=VALUE` overrides a generator option for experiments (`--set SWAP=0`, `--set DIAG2_FMA=0`). Compare
variants with the miner's `--benchmark` at a locked core clock (`--gpu-cclock 1400`), where runs repeat to
0.01 MH/s, and point `XM_KERNEL_FILE` at a variant (the file must be named `quantus_sm_86.cubin`) to run it without
rebuilding the miner.

The output is deterministic: the committed cubin is what this produces with CUDA 12.5. After a rebuild, run
`x-miner --selftest` on an RTX 30 card before committing. That checks GPU candidates against the CPU
reference; the emulator does not check timing.
