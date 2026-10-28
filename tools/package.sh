#!/bin/sh
# Builds a Release x-miner and packs dist/X-Miner-<version>-linux-x64.tar.gz with the launchers and docs.
#   tools/package.sh [--skip-build]
# The binary links libstdc++ statically; it still needs a glibc at least as new as the build host's
# (build on the oldest distro you want to support, e.g. Ubuntu 20.04 for HiveOS).
set -e
cd "$(dirname "$0")/.."

build="${BUILD_DIR:-build-release}"
version=$(sed -n 's/^#define XM_VERSION "\(.*\)"/\1/p' src/core/version.hpp)
if [ "$1" != "--skip-build" ]; then
  cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build"
  ./"$build"/xm_tests
fi

name="X-Miner-$version-linux-x64"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage"
cp "$build"/x-miner LICENSE NOTICE THIRD_PARTY.md README.md "$stage/"
# NVRTC for KawPow on NVIDIA (the driver does not include it): NVRTC_DIR=<cuda>/lib64. AMD uses ROCm's libhiprtc.
nvrtc() { [ -n "$NVRTC_DIR" ] && cp -L "$NVRTC_DIR"/libnvrtc.so.1[0-9] "$NVRTC_DIR"/libnvrtc-builtins.so.1[0-9].[0-9]* "$1/"; }
nvrtc "$stage" || echo "warning: NVRTC_DIR not set: KawPow will not run on NVIDIA from this package"
for f in launchers/linux/*.sh; do
  tr -d '\r' < "$f" > "$stage/$(basename "$f")"
  chmod +x "$stage/$(basename "$f")"
done
chmod +x "$stage/x-miner"

"$stage/x-miner" --version
tar -C dist -czf "dist/$name.tar.gz" "$name"
(cd dist && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "packed dist/$name.tar.gz"
cat "dist/$name.tar.gz.sha256"

# HiveOS custom-miner package: x-miner-<version>-hiveos.tar.gz with the folder x-miner/.
hive="dist/hiveos/x-miner"
rm -rf dist/hiveos
mkdir -p "$hive"
cp "$build"/x-miner LICENSE NOTICE "$hive/"
nvrtc "$hive" || true
for f in launchers/hiveos/*; do tr -d '\r' < "$f" > "$hive/$(basename "$f")"; done
sed -i "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=$version/" "$hive/h-manifest.conf"
chmod 644 "$hive"/*
chmod 755 "$hive"/*.so* 2>/dev/null || true
chmod +x "$hive"/*.sh "$hive/x-miner"
tar -C dist/hiveos -czf "dist/x-miner-$version-hiveos.tar.gz" x-miner
(cd dist && sha256sum "x-miner-$version-hiveos.tar.gz" > "x-miner-$version-hiveos.tar.gz.sha256")
echo "packed dist/x-miner-$version-hiveos.tar.gz"
