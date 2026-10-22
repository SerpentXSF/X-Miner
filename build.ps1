# Windows build: Visual Studio 2022 (C++ workload) + its bundled CMake and Ninja.
#   .\build.ps1            Release build + unit tests
#   .\build.ps1 -Clean     wipe build\ first
param([switch]$Clean, [string]$Config = "Release")
$root = $PSScriptRoot
$installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$env:PATH = "$installer;$env:PATH"   # vcvars64.bat calls vswhere itself
$vs = & "$installer\vswhere.exe" -latest -property installationPath
if (-not $vs) { throw "Visual Studio 2022 with the C++ workload is required" }

# Import the x64 developer environment into this PowerShell session.
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
$cmake = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

& $cmake -S $root -B $build -G Ninja "-DCMAKE_BUILD_TYPE=$Config"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $cmake --build $build
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& "$build\xm_tests.exe"
exit $LASTEXITCODE
