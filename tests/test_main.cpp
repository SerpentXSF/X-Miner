#include "test.hpp"

namespace xmtest {
std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
int failures = 0;
}  // namespace xmtest

int main(int argc, char** argv) {
  std::string only = argc > 1 ? argv[1] : "";
  int run = 0;
  for (auto& c : xmtest::registry()) {
    if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
    int before = xmtest::failures;
    c.fn();
    ++run;
    std::printf("%s %s\n", xmtest::failures == before ? "ok  " : "FAIL", c.name);
  }
  std::printf("%d tests, %d failures\n", run, xmtest::failures);
  return xmtest::failures ? 1 : 0;
}
