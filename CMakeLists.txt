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
endif()

find_package(Threads REQUIRED)

add_library(xm_core STATIC
  src/core/console.cpp
  src/core/config.cpp
  src/core/devfee.cpp
  src/core/stats.cpp
  src/core/banner.cpp
  src/algo/registry.cpp
  src/gpu/hip_backend.cpp
  src/gpu/cuda_backend.cpp
)
target_include_directories(xm_core PUBLIC src third_party)
target_link_libraries(xm_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(xm_core PUBLIC ws2_32)
else()
  target_link_libraries(xm_core PUBLIC ${CMAKE_DL_LIBS})
endif()

add_executable(x-miner src/main.cpp)
target_link_libraries(x-miner PRIVATE xm_core)

enable_testing()
add_executable(xm_tests tests/test_main.cpp tests/test_core.cpp)
target_link_libraries(xm_tests PRIVATE xm_core)
add_test(NAME unit COMMAND xm_tests)
