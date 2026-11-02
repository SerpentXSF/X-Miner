# ngen: generated SASS for Nexa on sm_86 (RTX 30)

`kernels/prebuilt/nexa_sm_86.cubin` holds one kernel, `xm_nexa_b`: the two comb scalar multiplications of NexaPow
(P = x G and R = k G), written as scheduled sm_86 SASS by `ngen.py` and assembled with CuAssembler. On an RTX 3060 at
170 W the miner then runs at 27.7 MH/s, against 25.3 MH/s with the compiled kernel. Other GPUs, and RTX 30 with
non-default Nexa settings, use the run-time compiled kernel only.

This is build-time tooling. Releases ship just the miner packages; the cubin is embedded in `x-miner`.

## How the miner uses it

`kernels/nexa/nexa.cu` has a three-kernel form (`XM_NEXA_SPLIT`):
- `xm_nexa_a`: hashes, RFC6979 nonce, private key;
- `xm_nexa_b`: the comb multiplications;
- `xm_nexa_c`: batched inversion, signature, powHash, target check.

On sm_86 with the defaults (W 20, 256 threads, 2 blocks per SM), the engine compiles that form at run time and adds the
prebuilt module next to it (`Device::add_module`), whose `xm_nexa_b` then takes precedence. `XM_NEXA_GEN=0` keeps
the single compiled kernel.

The split exists for the generated code. A large block that ptxas cannot see needs a kernel around it that keeps
almost nothing live. In the single kernel, ptxas sank, rematerialized and duplicated the markers, and kept ~40
registers live across the block.

## What it does

1. nvcc compiles `nshell.cu`: `nexa.cu` with `XM_NEXA_SPLIT` and `XM_NEXA_GEN_SHELL`. In this build only `xm_nexa_b`
   exists. It loads x and k, and runs 18 marker XORs: 0..7 x, 8..15 k, 16..17 the nonce's `pr` row address. Each
   marker result is then stored to the row's pin words; a store keeps the marker in place and alive.
2. CuAssembler turns the cubin into text. `splice.py` then rewrites it:
   - each marker becomes `MOV V, src`, with its control code kept;
   - the generated block goes after the last marker;
   - instructions that depend on markers move after the block, their reads renamed to the V registers;
   - each marker register's final value is restored.

   Registers live after the block are reserved, and so is everything the code before the markers touches (a load
   may still be reading its address registers). The block starts and ends by waiting on all six scoreboards. The
   register count is set to 128.
3. The block (`ngen.py`):
   - takes the table pointer from the kernel parameters;
   - two passes (x, then k), each with 13 windows: digit and carry, the entry's address, four `LDG.E.128.CONSTANT`,
     y negated for a negative digit, and a mixed Jacobian + affine addition (madd-2007-bl, as the CUDA kernel);
   - stores X, Y, Z (8 words each) to the row.

   The field arithmetic is `field.py`: 8 x 32-bit words. A multiply sums each column of products in a 64-bit
   accumulator (`IMAD.WIDE.U32` with a carry-out predicate), and one `IADD3.X` counts two carries at a time; the
   reduction folds the high half with 2^256 = 2^32 + 977. Squares add the doubled cross products and then the squares.
4. **Check**: `nemu.py` emulates the block, in its final order with its loops, for random (x, k) pairs. Table entries
   are computed on first read, and the results are compared with Python elliptic-curve arithmetic. On any mismatch
   the cubin is not written. `test_field.py` checks the field ops alone, against Python integers.
5. CuAssembler assembles the result, and `tools/qgen/cuasm_fix.py` restores the shell's memory instruction encodings.

`nsass.py` is `tools/qgen/sass.py` with loads and stores. A scoreboard is not set in the cycle its load or store
issues, so a wait on it is kept at least 4 cycles after it. Waiting one cycle later gave ~1% wrong nonces under full
load, while the emulator and a small selftest passed. Global, shared and local loads wait on scoreboard 5, and
their address registers on 4; stores' source registers wait on 3. A 64-bit address `[Rn.64]` reads Rn and Rn + 1.

## Accepted slips

These lose a nonce at most, and the CPU re-checks every candidate:
- a zero digit (probability 2^-19 per window) adds the entry for 1;
- a field op drops a final carry (operands within 2^33 of 2^256, probability below 2^-200);
- the equal / opposite point cases of the addition are not handled.

## Requirements

As tools/qgen: the CUDA toolkit (nvcc, cuobjdump), the MSVC host compiler on Windows, and Python 3.8+.
[CuAssembler](https://github.com/cloudcores/CuAssembler) (MIT) at commit `96a9f72` with
`tools/qgen/cuassembler_sm86.patch` applied.

```bash
python tools/ngen/ngen.py --cuassembler <CuAssembler dir> --python <CuAssembler venv python> [--ccbin <cl.exe dir>]
```

It writes `kernels/prebuilt/nexa_sm_86.cubin`; `cmake` embeds it on the next configure.
