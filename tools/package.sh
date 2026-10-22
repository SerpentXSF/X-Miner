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
