#include "core/devfee.hpp"

#include <algorithm>

namespace xm {

const FeeTarget* fee_target(const std::string& algorithm) {
  for (const auto& t : kFeeTargets)
    if (algorithm == t.algorithm) return &t;
  return nullptr;
}

bool FeeSchedule::advance(double dt) {
  if (dt <= 0) return in_window();
  mined_total_ += dt;
  while (dt > 0) {
    // Split dt at the window boundary and at the end of the cycle so fee time is counted exactly.
    double boundary = in_window() ? cycle_s_ : cycle_s_ - window_s_;
    double step = std::min(dt, boundary - pos_);
    if (in_window()) fee_total_ += step;
    pos_ += step;
    dt -= step;
    if (pos_ >= cycle_s_) pos_ -= cycle_s_;
  }
  return in_window();
}

}  // namespace xm
