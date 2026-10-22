// A tiny test harness: TEST(name) { CHECK(cond); CHECK_EQ(a, b); }
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace xmtest {

struct Case {
  const char* name;
  std::function<void()> fn;
};
std::vector<Case>& registry();
extern int failures;

struct Register {
  Register(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

template <class A, class B>
void check_eq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
  if (!(a == b)) {
    std::ostringstream o;
    o << a << " != " << b;
    std::printf("  FAIL %s:%d  %s == %s  (%s)\n", file, line, ea, eb, o.str().c_str());
    ++failures;
  }
}

}  // namespace xmtest

#define XM_CAT2(a, b) a##b
#define XM_CAT(a, b) XM_CAT2(a, b)
#define TEST(name)                                                              \
  static void name();                                                           \
  static xmtest::Register XM_CAT(reg_, name)(#name, name);                      \
  static void name()
#define CHECK(c)                                                                \
  do {                                                                          \
    if (!(c)) {                                                                 \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                \
      ++xmtest::failures;                                                       \
    }                                                                           \
  } while (0)
#define CHECK_EQ(a, b) xmtest::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, eps) CHECK(std::fabs((a) - (b)) <= (eps))
