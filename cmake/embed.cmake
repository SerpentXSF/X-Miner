# Turns a file into a C++ byte array in a generated header.
#   cmake -DIN=<file> -DOUT=<header> -DNAME=<symbol> -P embed.cmake
# The array is NUL-terminated (usable as a C string for text) and kNAME_size excludes the terminator.
file(READ "${IN}" hex HEX)
string(LENGTH "${hex}" hexlen)
math(EXPR size "${hexlen} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
# Break into lines of 24 bytes so compilers and diffs stay happy.
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){24})" "\\1\n  " bytes "${bytes}")
file(WRITE "${OUT}" "// Generated from ${IN} by cmake/embed.cmake. Do not edit.\n#pragma once\n#include <cstddef>\n"
  "inline constexpr unsigned char ${NAME}_bytes[] = {\n  ${bytes}0x00};\n"
  "inline constexpr std::size_t ${NAME}_size = ${size};\n")
