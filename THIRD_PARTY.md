# Third-party code

| Component | Path | Licence | Source |
|---|---|---|---|
| nlohmann/json 3.11.3 | third_party/nlohmann/json.hpp | MIT | https://github.com/nlohmann/json (sha256 9bea4c80…103ea6) |
| HIP device-properties struct | third_party/hip/hip_device_prop.h | MIT (AMD) | HIP SDK hip_runtime_api.h |
| BLAKE3 test vectors | tests/vectors/blake3_test_vectors.json | CC0 / Apache-2.0 | https://github.com/BLAKE3-team/BLAKE3 |
| Pearl reference algorithm (ported to C++) | src/algo/pearl/ | ISC (Pearl Research Labs) | https://github.com/pearl-research-labs/pearl @82e6eee |
| Pearl golden vectors g3, g5 | tests/vectors/pearl_golden_*.json | ISC (PRL-3090 contributors) | https://github.com/rheza/prl-3090-miner tests/golden |
| Parano1d PoW constants (generated) | src/algo/noid/noid_consts.hpp | Apache-2.0 (Paranoid Zero) | https://github.com/ignotusnemo/parano1d @5302b3f; see NOTICE |
| Parano1d digest vectors | tests/vectors/noid_vectors.txt | Apache-2.0 | https://github.com/gacanga/NoidMiner tests/vectors.txt (generated from the official crates) |
| Parano1d GF(2^128) arithmetic and permutation (ported) | kernels/noid/noid_math.h, noid_perm.h | Apache-2.0 (gacanga) | https://github.com/gacanga/NoidMiner @31fa4a8, modified for HIP |
| ADL structure layouts and IDs (AdapterInfo, PMLog, Overdrive8) | src/gpu/sensors.cpp, src/gpu/tuning.cpp | MIT (Copyright (c) Advanced Micro Devices, Inc.) | https://github.com/GPUOpen-LibrariesAndSDKs/display-library include/adl_defines.h, adl_structures.h |
| NVML function signatures and nvmlClockOffset_t layout | src/gpu/sensors.cpp, src/gpu/tuning.cpp | NVIDIA NVML API (no code copied; runtime library ships with the driver) | NVML API reference |
