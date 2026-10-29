cmake_minimum_required(VERSION 3.20)
project(x-miner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /utf-8 /permissive- /EHsc)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
  # Static CRT: x-miner.exe runs without the Visual C++ redistributable.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  add_compile_options(-Wall -Wextra)
  # Portable Linux binary: no dependency on the target's libstdc++ version.
  add_link_options(-static-libstdc++ -static-libgcc)
endif()

find_package(Threads REQUIRED)

# Embed the web dashboard and logo into the executable.
set(XM_GEN ${CMAKE_BINARY_DIR}/generated)
function(xm_embed in name)
  add_custom_command(OUTPUT ${XM_GEN}/${name}.hpp
    COMMAND ${CMAKE_COMMAND} -DIN=${CMAKE_SOURCE_DIR}/${in} -DOUT=${XM_GEN}/${name}.hpp -DNAME=${name}
            -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${CMAKE_SOURCE_DIR}/${in} ${CMAKE_SOURCE_DIR}/cmake/embed.cmake VERBATIM)
endfunction()
xm_embed(web/dashboard.html dashboard_html)
xm_embed(assets/serpentx-logo.png logo_png)
include(cmake/kernels.cmake)

add_library(xm_core STATIC
  ${XM_GEN}/dashboard_html.hpp
  ${XM_GEN}/logo_png.hpp
  ${XM_KERNEL_HEADERS}
  ${XM_GEN}/kernel_table.cpp
  src/gpu/kernels.cpp
  src/algo/selftest.cpp
  src/net/tcp.cpp
  src/net/tls.cpp
  src/core/gzip.cpp
  src/crypto/blake3.cpp
  src/crypto/sha256.cpp
  src/algo/pearl/pearl.cpp
  src/algo/noid/noid.cpp
  src/algo/noid/noid_gpu.cpp
  src/algo/noid/bench.cpp
  src/algo/pearl/share_test.cpp
  src/algo/pearl/pearl_gpu.cpp
  src/algo/pearl/bench.cpp
  src/pool/pearl_pool.cpp
  src/mining/pearl_run.cpp
  src/pool/noid_pool.cpp
  src/mining/noid_run.cpp
  src/algo/blake2b/blake2b.cpp
  src/algo/blake2b/blake2b_gpu.cpp
  src/algo/blake2b/bench.cpp
  src/pool/blake2b_pool.cpp
  src/mining/blake2b_run.cpp
  src/algo/quantus/quantus.cpp
  src/algo/quantus/quantus_gpu.cpp
  src/algo/quantus/bench.cpp
  src/algo/kawpow/kawpow.cpp
  src/algo/etchash/etchash.cpp
  src/algo/autolykos/autolykos.cpp
  src/algo/autolykos/autolykos_gpu.cpp
  src/algo/autolykos/bench.cpp
  src/pool/ergo_pool.cpp
  src/mining/autolykos_run.cpp
  src/algo/etchash/etchash_gpu.cpp
  src/algo/etchash/bench.cpp
  src/algo/xelis/xelis.cpp
  src/algo/xelis/xelis_gpu.cpp
  src/algo/xelis/bench.cpp
  src/algo/qhash/qhash.cpp
  src/algo/qhash/qhash_gpu.cpp
  src/algo/qhash/bench.cpp
  src/algo/fishhash/fishhash.cpp
  src/algo/fishhash/fishhash_gpu.cpp
  src/algo/fishhash/bench.cpp
  src/algo/octopus/octopus.cpp
  src/algo/octopus/octopus_gpu.cpp
  src/algo/octopus/bench.cpp
  src/pool/ethstratum_pool.cpp
  src/mining/etchash_run.cpp
  src/mining/octopus_run.cpp
  src/pool/cfx_pool.cpp
  src/pool/iron_pool.cpp
  src/pool/kls_pool.cpp
  src/pool/epic_pool.cpp
  src/pool/zano_pool.cpp
  src/pool/xelis_pool.cpp
  src/mining/xelis_run.cpp
  src/pool/qhash_pool.cpp
  src/mining/qhash_run.cpp
  src/mining/fishhash_run.cpp
  src/algo/kawpow/kawpow_gpu.cpp
  src/algo/kawpow/bench.cpp
  src/pool/kawpow_pool.cpp
  src/mining/kawpow_run.cpp
  src/gpu/rtc.cpp
  src/pool/quantus_pool.cpp
  src/mining/quantus_run.cpp
  src/api/api.cpp
  src/core/console.cpp
  src/core/config.cpp
  src/core/devfee.cpp
  src/core/stats.cpp
  src/core/banner.cpp
  src/core/report.cpp
  src/algo/registry.cpp
  src/gpu/hip_backend.cpp
  src/gpu/cuda_backend.cpp
  src/gpu/sensors.cpp
  src/gpu/tuning.cpp
)
target_include_directories(xm_core PUBLIC src third_party ${XM_GEN} kernels)
target_link_libraries(xm_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(xm_core PUBLIC ws2_32 secur32)
else()
  target_link_libraries(xm_core PUBLIC ${CMAKE_DL_LIBS})
endif()

add_executable(x-miner src/main.cpp)
target_link_libraries(x-miner PRIVATE xm_core)

enable_testing()
add_executable(xm_tests tests/test_main.cpp tests/test_core.cpp tests/test_blake3.cpp tests/test_pearl.cpp tests/test_noid.cpp tests/test_blake2b.cpp tests/test_quantus.cpp tests/test_kawpow.cpp tests/test_etchash.cpp tests/test_autolykos.cpp tests/test_octopus.cpp tests/test_fishhash.cpp tests/test_pools.cpp tests/test_xelis.cpp tests/test_qhash.cpp)
target_compile_definitions(xm_tests PRIVATE XM_TEST_VECTORS="${CMAKE_SOURCE_DIR}/tests/vectors")
target_link_libraries(xm_tests PRIVATE xm_core)
add_test(NAME unit COMMAND xm_tests)
