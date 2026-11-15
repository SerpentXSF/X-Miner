# Compiles every GPU kernel into kernels/prebuilt/<algo>_<arch>.<ext>, which CMake embeds into x-miner.
# Run this on a machine with the AMD HIP SDK and the CUDA toolkit after changing a kernel; commit the outputs so
# builds without GPU toolchains (CI, Linux packagers) still produce a complete miner.
#   AMD:    <algo>_gfx1201.hsaco, _gfx1200 (RX 9070 / 9060 code objects, loaded with hipModuleLoadData), and RDNA1-3:
#           gfx1010-1012 (RX 5000, BC-160), gfx1030-1034 (RX 6000), gfx1100-1102 (RX 7000); Pearl runs its SIMT path there
#           <algo>_gfx10-1-generic / gfx10-3-generic / gfx11-generic / gfx12-generic: LLVM family-generic code objects
#           for chips without their own (BC-250, APUs such as gfx1035/1103/1150), loaded when no exact one exists
#   NVIDIA: <algo>_sm_86.cubin, _sm_89    (exact-arch binaries, RTX 30 / RTX 40)
#           <algo>_sm_120.cubin           (RTX 50; needs CUDA 12.8+, built separately so the PTX stays loadable by
#                                          older drivers: -Nvcc <cuda-12.8>\bin\nvcc.exe -NvArchs sm_120 -AmdArchs @() -NoPtx)
#           <algo>_compute_80.ptx         (portable PTX, JIT-compiled by the driver on newer GPUs)
#           <algo>_compute_61.ptx         (Pascal / Volta / Turing; the miner prefers the newest PTX a GPU can run)
#           pearl_compute_75.ptx          (Turing: Pearl's int8 m8n8k16 kernel; -NvPtxPearl)
#   Except quantus_sm_86.cubin: generated SASS from tools/qgen (Python + CuAssembler, see tools/qgen/README.md).
#   Without -CuAssembler (or XM_CUASSEMBLER) the committed file is kept.
#   noidt_* and noidcl_sm_86 / _sm_89 / _sm_120.cubin: NOID with the CLMAD instruction (tower64 and composite basis),
#   CUDA 13.3+ only (-Nvcc13 <cuda-13.3>/bin/nvcc.exe); NVIDIA only, no PTX. Without -Nvcc13 the committed files are kept.
param([string]$HipClang = "", [string[]]$AmdArchs = @("gfx1201", "gfx1200", "gfx1102", "gfx1101", "gfx1100", "gfx1034", "gfx1032",
                                 "gfx1031", "gfx1030", "gfx1012", "gfx1011", "gfx1010",
                                 "gfx12-generic", "gfx11-generic", "gfx10-3-generic", "gfx10-1-generic"), [string[]]$NvArchs = @("sm_86", "sm_89"),
      [string[]]$NvPtx = @("compute_80", "compute_61"), [string[]]$NvPtxPearl = @("compute_75"), [string]$Nvcc = "", [string]$Nvcc13 = "", [switch]$NoPtx, [string]$CuAssembler = $env:XM_CUASSEMBLER)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root "kernels\prebuilt"
New-Item -ItemType Directory -Force $out | Out-Null

if (-not $HipClang) {
  foreach ($c in @("$env:HIP_PATH\bin\clang++.exe", "G:\Program Files\AMD\ROCm\bin\clang++.exe",
                   "C:\Program Files\AMD\ROCm\6.4\bin\clang++.exe")) { if ($c -and (Test-Path $c)) { $HipClang = $c; break } }
}
$nvcc = if ($Nvcc) { $Nvcc } else { (Get-Command nvcc -ErrorAction SilentlyContinue).Source }
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
$ccbin = (Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1).DirectoryName

# Each kernel source: kernels/<algo>/<algo>.cu (HIP and CUDA from one file) unless a vendor-specific
# kernels/<algo>/<algo>.hip exists for AMD.
# Run-time compiled only (the miner prepends their parameters): nexa and c29. kernels/prebuilt/nexa_sm_86.cubin is
# generated SASS from tools/ngen (see tools/ngen/README.md); nexatab is nexa's prebuilt comb-table builder.
$rtcOnly = @("prebuilt", "nexa", "c29")
$dirs = Get-ChildItem (Join-Path $root "kernels") -Directory | Where-Object { $rtcOnly -notcontains $_.Name }
foreach ($dir in $dirs) {
  $algo = $dir.Name
  $cu = Join-Path $dir.FullName "$algo.cu"
  $hip = Join-Path $dir.FullName "$algo.hip"
  if ($algo -eq "noidcl" -or $algo -eq "noidt") {
    if (-not $Nvcc13) { Write-Warning "no -Nvcc13: keeping the committed $algo cubins"; continue }
    foreach ($a in @("sm_86", "sm_89", "sm_120")) {
      & $Nvcc13 -ccbin $ccbin -cubin "-arch=$a" -O3 -std=c++17 -I (Join-Path $root "kernels") `
        -o (Join-Path $out "${algo}_$a.cubin") $cu
      if ($LASTEXITCODE) { throw "nvcc 13 failed: $algo $a" }
      "built ${algo}_$a.cubin"
    }
    continue
  }
  $amdSrc = if (Test-Path $hip) { $hip } elseif (Test-Path $cu) { $cu } else { $null }
  if ($HipClang -and $amdSrc) {
    foreach ($a in $AmdArchs) {
      & $HipClang -x hip --cuda-device-only --no-gpu-bundle-output "--offload-arch=$a" -O3 -std=c++17 `
        -I (Join-Path $root "kernels") -o (Join-Path $out "${algo}_$a.hsaco") $amdSrc
      if ($LASTEXITCODE) { throw "HIP build failed: $algo $a" }
      "built ${algo}_$a.hsaco"
    }
  } elseif ($amdSrc) { Write-Warning "HIP clang not found; skipping AMD build of $algo" }
  if ($nvcc -and (Test-Path $cu)) {
    foreach ($a in $NvArchs) {
      if ($algo -eq "quantus" -and $a -eq "sm_86") {
        if (-not $CuAssembler) { Write-Warning "no CuAssembler: keeping the committed quantus_sm_86.cubin"; continue }
        $py = Join-Path $CuAssembler ".venv\Scripts\python.exe"
        if (-not (Test-Path $py)) { $py = "python" }
        & $py (Join-Path $PSScriptRoot "qgen\qgen.py") --cuassembler $CuAssembler --nvcc $nvcc --ccbin $ccbin --python $py `
          --out (Join-Path $out "quantus_sm_86.cubin")
        if ($LASTEXITCODE) { throw "qgen failed: quantus sm_86" }
        continue
      }
      & $nvcc -ccbin $ccbin -cubin "-arch=$a" -O3 -std=c++17 -I (Join-Path $root "kernels") `
        -o (Join-Path $out "${algo}_$a.cubin") $cu
      if ($LASTEXITCODE) { throw "nvcc failed: $algo $a" }
      "built ${algo}_$a.cubin"
    }
    if ($NoPtx) { continue }
    foreach ($p in $(if ($algo -eq "pearl") { $NvPtx + $NvPtxPearl } else { $NvPtx })) {
      & $nvcc -ccbin $ccbin -ptx "-arch=$p" -O3 -std=c++17 -I (Join-Path $root "kernels") `
        -o (Join-Path $out "${algo}_$p.ptx") $cu
      if ($LASTEXITCODE) { throw "nvcc ptx failed: $algo $p" }
      # nvcc tags the PTX with its own ISA version (8.5 for CUDA 12.5), and a driver rejects a newer ISA than it knows
      # (CUDA error 222). Our kernels need ISA 7.1 at most, so tag them 7.1: any driver since CUDA 11.1 can JIT them.
      # ptxas re-checks the file at that version so a kernel that does need a newer ISA fails the build here.
      $pf = Join-Path $out "${algo}_$p.ptx"
      $txt = [IO.File]::ReadAllText($pf) -replace '(?m)^\.version \d+\.\d+', '.version 7.1'
      [IO.File]::WriteAllText($pf, $txt)
      $ptxas = Join-Path (Split-Path $nvcc) "ptxas.exe"
      if (-not (Test-Path $ptxas)) { $ptxas = Join-Path (Split-Path $nvcc) "ptxas" }
      & $ptxas "-arch=$($p -replace 'compute_', 'sm_')" $pf -o "$pf.check.o"
      if ($LASTEXITCODE) { throw "${algo}_$p.ptx needs a PTX ISA newer than 7.1" }
      Remove-Item "$pf.check.o"
      "built ${algo}_$p.ptx"
    }
  } elseif (Test-Path $cu) { Write-Warning "nvcc not found; skipping NVIDIA build of $algo" }
}
