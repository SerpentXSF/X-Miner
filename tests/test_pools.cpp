// Pool message parsing that needs no network: Epic Cash jobs and targets, Zano getWork results (a HeroMiners reply
// captured 2026-10-28), and the wallet/worker login join.
#include <cstring>
#include <string>

#include "core/config.hpp"
#include "pool/epic_pool.hpp"
#include "pool/zano_pool.hpp"
#include "test.hpp"

using namespace xm;

TEST(epic_target_division) {
  CHECK_EQ(kawpow::hash_to_hex(epic_target(1)), std::string(64, 'f'));
  CHECK_EQ(kawpow::hash_to_hex(epic_target(2)), "7" + std::string(63, 'f'));
  // (2^256 - 1) / 200000 = 0x000053e2d6238da3c21187e7c06e19b9...
  CHECK_EQ(kawpow::hash_to_hex(epic_target(200000)).substr(0, 16), std::string("000053e2d6238da3"));
}

TEST(epic_job_parse) {
  kawpow::Job j;
  CHECK(epic_parse_job(R"({"algorithm":"randomx","difficulty":[["cuckoo",3],["randomx",20000],["progpow",200000]],
                          "height":3774291,"job_id":194123,"pre_pow":"0006"})", j));
  CHECK_EQ(j.id, std::string("3774291:194123"));
  CHECK_EQ(j.height, 3774291);
  CHECK(j.difficulty == 200000.0);
  const auto k = kawpow::keccak256(reinterpret_cast<const uint8_t*>("\x00\x06"), 2);
  CHECK(std::memcmp(j.header.data(), k.data(), 32) == 0);
  CHECK(j.seed == kawpow::epoch_seed(3774291 / 30000));
  kawpow::Job z;
  CHECK(!epic_parse_job(R"({"difficulty":[["progpow",0]],"height":1,"job_id":0,"pre_pow":"00"})", z));
}

TEST(zano_work_parse) {
  kawpow::Job j;
  CHECK(zano_parse_work(R"(["0xa81d78c03c272d8be8a46763d5e5830c3dbe0f75679c56bba93564be2327780f",
      "0x3868abc5dcee23341dee7f75f1be7f90ca9904961fef9e0f102cedba1171408d",
      "0x000000044b82fa09b5a52cb98b405447c4a98187eebb22f008d5d64f9c394ae9","0x00000000003b327e"])", j));
  CHECK_EQ(j.height, 0x3b327e);
  CHECK_EQ(kawpow::hash_to_hex(j.header), std::string("a81d78c03c272d8be8a46763d5e5830c3dbe0f75679c56bba93564be2327780f"));
  CHECK_EQ(kawpow::hash_to_hex(j.target).substr(0, 16), std::string("000000044b82fa09"));
  CHECK(j.difficulty > 0.999999e9 && j.difficulty < 1.000001e9);  // HeroMiners difficulty 1e9
  kawpow::Job k;
  CHECK(!zano_parse_work(R"(["0xa81d78c03c272d8be8a46763d5e5830c3dbe0f75679c56bba93564be2327780f",
      "0x3868abc5dcee23341dee7f75f1be7f90ca9904961fef9e0f102cedba1171408d",
      "0x000000044b82fa09b5a52cb98b405447c4a98187eebb22f008d5d64f9c394ae9"])", k));
}

TEST(login_join) {
  CHECK_EQ(join_login("krxABC", "rig1"), std::string("krxABC.rig1"));
  CHECK_EQ(join_login("krxABC.rig2", "rig1"), std::string("krxABC.rig2"));
  CHECK_EQ(join_login("esWj@epicbox.epiccash.com", "rig1"), std::string("esWj@epicbox.epiccash.com+rig1"));
  CHECK_EQ(join_login("esWj@epicbox.epiccash.com+rig2", "rig1"), std::string("esWj@epicbox.epiccash.com+rig2"));
  CHECK_EQ(join_login("karlsen:qq", ""), std::string("karlsen:qq"));
  CHECK_EQ(join_login("me@mail.com", "rig1"), std::string("me@mail.com"));  // an e-mail login is left alone
}
