#!/bin/bash
# Collects the ROCm HIP runtime that the Linux and HiveOS packages ship in hip/ (HiveOS installs only ROCm's OpenCL,
# so X-Miner would find no AMD GPU there). Run on Ubuntu 22.04 as root, e.g. in a container:
#   docker run --rm -v "$PWD:/src" ubuntu:22.04 bash /src/tools/fetch_hip_runtime.sh /src/build-hip
# then package with HIP_RUNTIME_DIR=build-hip tools/package.sh. Unmodified files from AMD's repository, plus the four
# system libraries the runtime needs that minimal installs lack (libnuma, libelf, libdrm, libdrm_amdgpu, from Ubuntu
# 22.04; the miner prefers the system's copies), each with its license files in hip/licenses/. The libraries need
# glibc 2.35 (Ubuntu 22.04+, Debian 12+, Fedora 36+, current HiveOS).
set -e
out="${1:?usage: fetch_hip_runtime.sh <output dir>}"
rocm="${ROCM_VERSION:-6.4.4}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$rocm jammy main" \
  > /etc/apt/sources.list.d/rocm.list
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends hip-runtime-amd libnuma1 libelf1 libdrm2 libdrm-amdgpu1 >/dev/null
r=/opt/rocm-$rocm
rm -rf "$out"
mkdir -p "$out/licenses"
# Only files from AMD's packages (ROCr 6.4 has the thunk library built in: no libhsakmt).
for l in libamdhip64.so.6 libhiprtc.so.6 libhiprtc-builtins.so.6 libhsa-runtime64.so.1 librocprofiler-register.so.0 \
         libamd_comgr.so.3; do
  f=$(find "$r/lib" -maxdepth 1 -name "$l" 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "missing $l" >&2; exit 1; }
  cp -L "$f" "$out/$l"
done
for l in libnuma.so.1 libelf.so.1 libdrm.so.2 libdrm_amdgpu.so.1; do
  cp -L "/usr/lib/x86_64-linux-gnu/$l" "$out/$l"
done
# Every library must resolve inside hip/ or against base libraries every distro has (libc, libstdc++, libz, libzstd).
LD_LIBRARY_PATH="$out" ldd "$out"/*.so* | grep "not found" && { echo "unresolved dependency" >&2; exit 1; }
for p in hip-runtime-amd hsa-rocr comgr rocprofiler-register libnuma1 libelf1 libdrm2 libdrm-amdgpu1; do
  dpkg -L "$p" 2>/dev/null | grep -iE "/(licen[cs]e|copyright|notices?)[^/]*$" | while read -r f; do
    [ -f "$f" ] && cp -L "$f" "$out/licenses/$p-$(basename "$f")"
  done
  ls "$out/licenses" | grep -q "^$p-" || { echo "no license file for $p" >&2; exit 1; }
done
# The Debian copyright files of libnuma (LGPL-2.1) and libelf (LGPL-3+ / GPL-2+) point at these full texts.
for l in LGPL-2.1 LGPL-3 GPL-2 GPL-3; do cp -L "/usr/share/common-licenses/$l" "$out/licenses/$l.txt"; done
dpkg-query -W -f '${Package} ${Version}\n' hip-runtime-amd hsa-rocr comgr rocprofiler-register libnuma1 libelf1 \
  libdrm2 libdrm-amdgpu1 2>/dev/null \
  > "$out/licenses/VERSIONS.txt" || true
ls "$out" "$out/licenses"
du -sh "$out"
