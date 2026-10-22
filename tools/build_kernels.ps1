# Compiles every GPU kernel into kernels/prebuilt/<algo>_<arch>.<ext>, which CMake embeds into x-miner.
# Run this on a machine with the AMD HIP SDK and the CUDA toolkit after changing a kernel; commit the outputs so
# builds without GPU toolchains (CI, Linux packagers) still produce a complete miner.
#   AMD:    <algo>_gfx1201.hsaco         (code object, loaded with hipModuleLoadData)
#   NVIDIA: <algo>_sm_86.cubin            (exact-arch binary)
#           <algo>_compute_80.ptx         (portable PTX, JIT-compiled by the driver on newer GPUs)
param([string]$HipClang = "", [string[]]$AmdArchs = @("gfx1201"), [string[]]$NvArchs = @("sm_86"),
      [string]$NvPtx = "compute_80")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root "kernels\prebuilt"
New-Item -ItemType Directory -Force $out | Out-Null

if (-not $HipClang) {
  foreach ($c in @("$env:HIP_PATH\bin\clang++.exe", "G:\Program Files\AMD\ROCm\bin\clang++.exe",
                   "C:\Program Files\AMD\ROCm\6.4\bin\clang++.exe")) { if ($c -and (Test-Path $c)) { $HipClang = $c; break } }
}
$nvcc = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
$ccbin = (Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1).DirectoryName

# Each kernel source: kernels/<algo>/<algo>.cu (HIP and CUDA from one file) unless a vendor-specific
# kernels/<algo>/<algo>.hip exists for AMD.
foreach ($dir in Get-ChildItem (Join-Path $root "kernels") -Directory | Where-Object Name -ne "prebuilt") {
  $algo = $dir.Name
  $cu = Join-Path $dir.FullName "$algo.cu"
  $hip = Join-Path $dir.FullName "$algo.hip"
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
      & $nvcc -ccbin $ccbin -cubin "-arch=$a" -O3 -std=c++17 -I (Join-Path $root "kernels") `
        -o (Join-Path $out "${algo}_$a.cubin") $cu
      if ($LASTEXITCODE) { throw "nvcc failed: $algo $a" }
      "built ${algo}_$a.cubin"
    }
    & $nvcc -ccbin $ccbin -ptx "-arch=$NvPtx" -O3 -std=c++17 -I (Join-Path $root "kernels") `
      -o (Join-Path $out "${algo}_$NvPtx.ptx") $cu
    if ($LASTEXITCODE) { throw "nvcc ptx failed: $algo" }
    "built ${algo}_$NvPtx.ptx"
  } elseif (Test-Path $cu) { Write-Warning "nvcc not found; skipping NVIDIA build of $algo" }
}
