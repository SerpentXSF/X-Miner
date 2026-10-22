// Developer tool: CPU-mine one small Pearl share with an easy target and write it for tools/pearl-oracle.
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include "algo/pearl/pearl.hpp"
#include "core/console.hpp"

namespace xm::pearl {

int share_test(const std::string& dir) {
  // A real 76-byte LuckyPool header (from ARIA-DevCode/ariaminer luckypool_job_check.rs, MIT; see pearl spec 1.1).
  const char* hh =
      "00004020df2a4132ba5893f1954bf490d9852173e307dd71dec8f68533d27fc0c9ca31276b5380962696afc7352641624e7cb7565a"
      "776af3934f19b2784b5750141f20cacc35296a9cfe0018";
  Work w;
  for (int i = 0; i < 76; ++i) w.header[i] = static_cast<uint8_t>(std::stoul(std::string(hh + 2 * i, 2), nullptr, 16));
  w.config.k = 2048;  // consensus: 16r <= k, k % 64 == 0, k >= 1024
  w.config.rank = 128;
  w.m = 32;
  w.n = 48;
  std::mt19937 rng(20261021);
  std::uniform_int_distribution<int> d(-64, 63);
  w.a.resize(size_t(w.m) * w.config.k);
  w.bt.resize(size_t(w.n) * w.config.k);
  for (auto& x : w.a) x = static_cast<int8_t>(d(rng));
  for (auto& x : w.bt) x = static_cast<int8_t>(d(rng));
  const uint32_t nbits = 0x1f00ffff;  // very easy share target: most tiles qualify
  auto found = cpu_mine(w, nbits, 1);
  if (found.empty()) {
    error("TEST", "no tile met the test target");
    return 1;
  }
  PlainProof p = build_proof(w, found[0]);
  std::string b64 = base64(serialize(p));
  std::ofstream(dir + "/share.b64") << b64;
  std::ofstream(dir + "/header.hex") << hh;
  char nb[16];
  std::snprintf(nb, sizeof nb, "%08x", nbits);
  std::ofstream(dir + "/nbits.hex") << nb;
  info("TEST", "tile (" + std::to_string(found[0].t_rows) + ", " + std::to_string(found[0].t_cols) + ") share " +
                   std::to_string(b64.size()) + " base64 bytes written to " + dir);
  return 0;
}

}  // namespace xm::pearl
