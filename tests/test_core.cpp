#include <vector>

#include "algo/registry.hpp"
#include "core/config.hpp"
#include "core/devfee.hpp"
#include "core/stats.hpp"
#include "test.hpp"

using namespace xm;

TEST(fee_is_one_percent_of_mining_time) {
  FeeSchedule f(1.0);
  CHECK_NEAR(f.window_seconds(), 60.0, 1e-9);
  // Tick in uneven steps for 10 full cycles.
  double t = 0;
  int i = 0;
  while (t < 10 * kCycleSeconds) {
    double dt = (i++ % 7) * 0.37 + 0.05;
    if (t + dt > 10 * kCycleSeconds) dt = 10 * kCycleSeconds - t;
    f.advance(dt);
    t += dt;
  }
  CHECK_NEAR(f.fee_seconds_total(), 600.0, 1e-6);
  CHECK_NEAR(f.mining_seconds_total(), 10 * kCycleSeconds, 1e-6);
}

TEST(fee_window_comes_at_end_of_cycle) {
  FeeSchedule f(1.0);
  CHECK(!f.advance(1000));
  CHECK_NEAR(f.seconds_to_window(), 4940.0, 1e-9);
  CHECK(!f.advance(4939.9));
  CHECK(f.advance(0.2));          // inside the 60-second window
  CHECK(f.advance(59.0));
  CHECK(!f.advance(1.0));         // cycle wrapped: back to the user
}

TEST(zero_fee_never_windows) {
  FeeSchedule f(0.0);
  for (int i = 0; i < 10000; ++i) CHECK(!f.advance(1.0));
  CHECK_NEAR(f.fee_seconds_total(), 0.0, 1e-12);
}

TEST(fee_targets_listed) {
  CHECK(fee_target("pearlhash") != nullptr);
  CHECK(fee_target("noid") != nullptr);
  CHECK_NEAR(fee_target("blake2b")->percent, 0.0, 0.0);
  CHECK(fee_target("nope") == nullptr);
}

TEST(pool_list_parsing) {
  std::vector<PoolEndpoint> p;
  CHECK(!parse_pool_list("prl-us.kryptex.network:7048!stratum+ssl://prl.kryptex.network:8048", p));
  CHECK_EQ(p.size(), size_t(2));
  CHECK_EQ(p[0].host, std::string("prl-us.kryptex.network"));
  CHECK_EQ(p[0].port, 7048);
  CHECK(!p[0].tls);
  CHECK(p[1].tls);
  CHECK_EQ(p[1].port, 8048);
  std::vector<PoolEndpoint> bad;
  CHECK(parse_pool_list("nohost", bad).has_value());
  CHECK(parse_pool_list("a:99999", bad).has_value());
}

TEST(cli_srbminer_style_flags) {
  const char* argv[] = {"x-miner", "--algorithm", "pearlhash", "--pool", "a.b:1!c.d:2", "--wallet", "krxABC",
                        "--worker", "rig1", "--gpu-id", "0,2", "--api-port", "4000", "--no-color"};
  Config c;
  auto err = parse_args(14, const_cast<char**>(argv), c);
  CHECK(!err);
  CHECK_EQ(c.algorithm, std::string("pearlhash"));
  CHECK_EQ(c.pools.size(), size_t(2));
  CHECK_EQ(c.login(), std::string("krxABC.rig1"));
  CHECK_EQ(c.gpu_ids.size(), size_t(2));
  CHECK_EQ(c.gpu_ids[1], 2);
  CHECK_EQ(c.api_port, 4000);
  CHECK(!c.color);
  const char* bad[] = {"x-miner", "--frobnicate"};
  Config c2;
  CHECK(parse_args(2, const_cast<char**>(bad), c2).has_value());
}

TEST(login_keeps_existing_worker) {
  Config c;
  c.wallet = "krxABC.XSF1";
  c.worker = "other";
  CHECK_EQ(c.login(), std::string("krxABC.XSF1"));
}

TEST(algorithm_aliases) {
  CHECK(find_algorithm("PearlHash") && std::string(find_algorithm("PearlHash")->name) == "pearlhash");
  CHECK(find_algorithm("parano1d") && std::string(find_algorithm("parano1d")->name) == "noid");
  CHECK(find_algorithm("kawpow") == nullptr);
}

TEST(hashrate_formatting) {
  CHECK_EQ(format_hashrate(100.43e12), std::string("100.43 TH/s"));
  CHECK_EQ(format_hashrate(25.06e6), std::string("25.06 MH/s"));
  CHECK_EQ(format_hashrate(950), std::string("950.00 H/s"));
  CHECK_EQ(format_duration(22320), std::string("6h 12m"));
}

TEST(hashrate_is_hashes_over_covered_time) {
  Stats st(1);
  for (int i = 0; i < 5; ++i) st.add_hashes(0, 2.0e12, 0.5);  // 4 TH/s in half-second reports
  auto snap = st.snapshot();
  CHECK_NEAR(snap.hr_now, 4.0e12, 1e3);
  CHECK_NEAR(snap.hr_1m, 4.0e12, 1e3);
}
