# Builds a Release x-miner.exe and packs dist\X-Miner-<version>-win64.zip with the launchers and docs.
#   .\tools\package.ps1   (always builds: packing a stale build-release\x-miner.exe once shipped an old binary)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Select-String -Path src\core\version.hpp -Pattern '#define XM_VERSION "([^"]+)"').Matches[0].Groups[1].Value
& .\build.ps1 -Config Release -BuildDir build-release
$exe = Join-Path $root 'build-release\x-miner.exe'
if (-not (Test-Path $exe)) { throw "missing $exe (build first)" }

$name = "X-Miner-$version-win64"
$stage = Join-Path $root "dist\$name"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null

Copy-Item $exe $stage
foreach ($f in 'LICENSE', 'NOTICE', 'THIRD_PARTY.md', 'README.md') { Copy-Item $f $stage }
# Launchers with CRLF line endings whatever the checkout settings.
Get-ChildItem launchers\windows\*.bat | ForEach-Object {
  $text = [IO.File]::ReadAllText($_.FullName) -replace "`r?`n", "`r`n"
  [IO.File]::WriteAllText((Join-Path $stage $_.Name), $text, [Text.Encoding]::ASCII)
}

& $exe --version
$zip = Join-Path $root "dist\$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Set-Content -Encoding ascii "$zip.sha256"
Write-Host "packed $zip"
Write-Host "sha256 $hash"
