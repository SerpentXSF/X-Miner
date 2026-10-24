// Dev fee routing shared by the mining coordinators: hands GPU workers jobs from the user's pool or, during a fee
// window, from a second connection logged in with the developer wallet, and sends each share back to the pool its
// job came from.
//
// The schedule (core/devfee.hpp) advances only with mining time: seconds in which the user's pool has a job, or in
// which the fee pool does during a window. The fee connection opens shortly before a window so a job is ready when
// it starts, and closes when the window ends. If the fee pool cannot be reached the window still runs out and the
// GPUs keep mining for the user.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/console.hpp"
#include "core/devfee.hpp"

namespace xm {

struct FeePlan {
  bool enabled = false;
  std::vector<PoolEndpoint> pools;
  std::string login;  // wallet.worker
  double percent = 0;
  double cycle_s = kCycleSeconds;
};

// The fee plan for an algorithm. XM_DEV_FEE_TEST="<wallet>:<cycle seconds>[:<percent>]" exercises the routing
// with a test wallet and a short cycle, and is honoured only by builds without a developer wallet (where the fee
// is off). XM_DEV_FEE_CYCLE="<seconds>[:<percent>]" shortens the cycle in any build, for testing the routing with
// the real fee wallet; the percentage can only be raised (up to 50), never lowered.
inline FeePlan fee_plan(const std::string& algorithm) {
  FeePlan p;
  const FeeTarget* t = fee_target(algorithm);
  if (!t || t->percent <= 0 || !t->pool[0]) return p;
  std::string wallet = t->wallet;
  p.percent = t->percent;
  if (wallet.empty()) {
    const char* e = std::getenv("XM_DEV_FEE_TEST");
    std::string v = e ? e : "";
    size_t c1 = v.find(':');
    if (c1 == std::string::npos) return p;
    size_t c2 = v.find(':', c1 + 1);
    wallet = v.substr(0, c1);
    p.cycle_s = std::atof(v.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1).c_str());
    if (c2 != std::string::npos) p.percent = std::atof(v.c_str() + c2 + 1);
    if (wallet.empty() || p.cycle_s < 60 || p.percent <= 0 || p.percent > 50) return p;
    char buf[160];
    std::snprintf(buf, sizeof buf, "XM_DEV_FEE_TEST: routing %.1f%% of every %.0f s to %s", p.percent, p.cycle_s,
                  wallet.c_str());
    warn("FEE", buf);
  }
  if (const char* e = std::getenv("XM_DEV_FEE_CYCLE")) {  // testing: shorter cycle, optionally a larger share
    const char* colon = std::strchr(e, ':');
    double c = std::atof(e), pct = colon ? std::atof(colon + 1) : 0;
    if (c >= 60) {
      p.cycle_s = c;
      if (pct >= p.percent && pct <= 50) p.percent = pct;
      char buf[96];
      std::snprintf(buf, sizeof buf, "XM_DEV_FEE_CYCLE: %.1f%% of every %.0f s", p.percent, p.cycle_s);
      warn("FEE", buf);
    }
  }
  if (parse_pool_list(t->pool, p.pools) || p.pools.empty()) return p;
  p.login = wallet + "." + t->worker;
  p.enabled = true;
  return p;
}

// Pool must provide: Pool(pools, login, password, ResultFn), start(), stop(), connected(),
// bool job(Job&, uint64_t&) const, endpoint().
template <class Pool, class Job>
class FeeRouter {
 public:
  using ResultFn = typename Pool::ResultFn;

  // on_fee_result is the result callback for the fee connection (it should count shares as fee shares).
  FeeRouter(Pool& user, FeePlan plan, ResultFn on_fee_result)
      : user_(user), plan_(std::move(plan)), on_fee_result_(std::move(on_fee_result)),
        sched_(plan_.enabled ? plan_.percent : 0.0, plan_.cycle_s) {}
  ~FeeRouter() { close_fee(); }

  // The job GPUs should mine now. version changes whenever the job or its source changes; fee says which pool the
  // job belongs to (pass it back to the submit call).
  bool job(Job& out, uint64_t& version, bool& fee) const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t v = 0;
    bool ok = on_fee_ ? fee_->job(out, v) : user_.job(out, v);
    version = generation_ << 40 | v;
    fee = on_fee_;
    return ok;
  }

  // Runs f(pool) on the pool a share's job came from; returns false if that pool is gone (the share is stale).
  template <class F>
  bool with_pool(bool fee, F&& f) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fee) {
      f(user_);
      return true;
    }
    if (!fee_) return false;
    f(*fee_);
    return true;
  }

  // Called once per second by the coordinator.
  void tick(double dt) {
    if (!plan_.enabled) return;
    std::unique_ptr<Pool> closing;
    {
      std::lock_guard<std::mutex> lk(mu_);
      Job j;
      uint64_t v;
      bool mining = on_fee_ ? fee_->job(j, v) : user_.job(j, v);
      bool window = mining ? sched_.advance(dt) : sched_.in_window();
      // Open the fee connection 20 s ahead so the first fee job is ready when the window starts.
      if (!fee_ && (window || sched_.seconds_to_window() < 20)) {
        fee_ = std::make_unique<Pool>(plan_.pools, plan_.login, "x", on_fee_result_);
        fee_->start();
      }
      bool want_fee = window && fee_ && fee_->job(j, v);
      if (want_fee != on_fee_) {
        on_fee_ = want_fee;
        ++generation_;
        log_line(Tone::Gold, "FEE", on_fee_ ? "dev fee window: mining for the developer for " +
                                                   std::to_string(static_cast<int>(sched_.window_seconds())) + " s"
                                             : "dev fee window over: back to your pool");
      }
      if (!window && fee_ && sched_.seconds_to_window() >= 20) {
        closing = std::move(fee_);  // joined outside the lock
        on_fee_ = false;
      }
    }
    if (closing) closing->stop();
  }

  bool enabled() const { return plan_.enabled; }
  double percent() const { return plan_.enabled ? plan_.percent : 0.0; }
  bool on_fee() const {
    std::lock_guard<std::mutex> lk(mu_);
    return on_fee_;
  }
  double seconds_to_window() const {
    std::lock_guard<std::mutex> lk(mu_);
    return sched_.seconds_to_window();
  }
  std::string endpoint() const {
    std::lock_guard<std::mutex> lk(mu_);
    return on_fee_ ? fee_->endpoint() : user_.endpoint();
  }
  bool connected() const {
    std::lock_guard<std::mutex> lk(mu_);
    return on_fee_ ? fee_->connected() : user_.connected();
  }

 private:
  void close_fee() {
    std::unique_ptr<Pool> p;
    {
      std::lock_guard<std::mutex> lk(mu_);
      p = std::move(fee_);
      on_fee_ = false;
    }
    if (p) p->stop();
  }

  Pool& user_;
  FeePlan plan_;
  ResultFn on_fee_result_;
  mutable std::mutex mu_;
  FeeSchedule sched_;
  std::unique_ptr<Pool> fee_;
  bool on_fee_ = false;
  uint64_t generation_ = 0;
};

}  // namespace xm
