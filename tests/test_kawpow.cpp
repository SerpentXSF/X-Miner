// KawPow reference. Vectors: cpp-kawpow test/unittests/progpow_test_vectors.hpp (@061d341, Apache-2.0).
#include <map>
#include <string>

#include "algo/kawpow/kawpow.hpp"
#include "test.hpp"

using namespace xm::kawpow;

namespace {

struct Vector {
  int height;
  const char* header;
  uint64_t nonce;
  const char* mix;
  const char* final;
};

const Vector kVectors[] = {
    {0, "0000000000000000000000000000000000000000000000000000000000000000", 0x0000000000000000ull,
     "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
     "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"},
    {49, "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b", 0x0000000007073c07ull,
     "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
     "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"},
    {99, "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d", 0x000000003917afabull,
     "fa706860e5e0e830d5d1d7157e5bea7f5f8a350c7c8612ac1d1fcf2974d64244",
     "aa85340690f2e907054324a5021937910e15edfd1ef1577231843e7d32ec3a61"},
    {29950, "ac7b55e801511b77e11d52e9599206101550144525b5679f2dab19386f23dcce", 0x005d409dbc23a62aull,
     "5359807b77a74878269c3a3044df8618a576ce8dc52e1c48d927d4a60e7c6b79",
     "022019e5408683f7f8326b4e46b42864a3a069f17b6151e434fcaedecaadd918"},
    {30000, "d34519f72c97cae8892c277776259db3320820cb5279a299d0ef1e155e5c6454", 0x005db8607994ff30ull,
     "de0348b69bf91dfe2c3d3dba6f0132e9048a5284e57b8d9d20adc5f3dc0d3236",
     "c7953d848cda6e304f77b4c6d735645c8e8508a5e74c9e9814ef37b19087cd6c"},
    {170915, "5b3e8dfa1aafd3924a51f33e2d672d8dae32fa528d8b1d378d6e4db0ec5d665d", 0x0000000044975727ull,
     "efb29147484c434f1cc59629da90fd0343e3b047407ecd36e9ad973bd51bbac5",
     "e7e6bb3b2f9acd3864bc86f72f87237eaf475633ef650c726ac80eb0adf116b6"},
};

std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

}  // namespace

TEST(kawpow_keccak) {
  auto h = keccak256(nullptr, 0);
  CHECK_EQ(hex(h.data(), 32), std::string("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"));
  auto h5 = keccak512(nullptr, 0);
  CHECK_EQ(hex(h5.data(), 8), std::string("0eab42de4c3ceb92"));
}

TEST(kawpow_epoch_sizes) {
  CHECK_EQ(light_cache_items(0), 262139);
  CHECK_EQ(dataset_items_1024(0), 8388593);
  CHECK_EQ(epoch_from_seed(epoch_seed(22)), 22);
  // Past epoch 255 the byte sizes overflow 32 bits (values from the reference formula in 64-bit arithmetic).
  CHECK_EQ(dataset_items_1024(613), 48562127);  // 6,215,952,256-byte DAG
  CHECK_EQ(light_cache_items(613), 1517567);
}

// FiroPow: firoorg/firo src/crypto/progpow/firopow_test_vectors.hpp @8246465 ({height, header, boundary, nonce, mix,
// final}; boundary unused here).
const Vector kFiroVectors[] = {
    {1, "2d794e900dcad779e658de9078d9a88eee87d75f7b09a8fdd270d3a8e76650c7", 0x85f22c9b3cd2f123ull,
     "cfab3766331d6c4e6913e6688a71e4c26b7f36c1581cdbec0f5b19db8956eb50",
     "00017c7de1fa499314f9e3dd3537546982073624f7d478592cf28a6d13929f2d"},
    {1300, "1fc36bd5d1bff8d134e24a997cfa43cbb2a0b956379bdc0c8df444f2553f6b7d", 0xdec25420bac29b01ull,
     "5f8fe6069efb88d9af861999973b3523295d3b1ec7e8423c965f33b3d12b20b1",
     "000038a1804eb3976346bf84e679884d56122e113c3b4a301f6180a246644262"},
    {96590, "0231f310fb1a0cfd63f363802ae3f0a90c50b3b879d8cedb243efc1446d0d6b2", 0x6fec0856f969dd81ull,
     "a91de8ff0849aed667d4e11f610e3fdc4612ea7d8fe6c58e902eaf95c978bf54",
     "000016f6ebb36334946238e06d484b8ed8ff50477729eb28c4c294a9e8218f97"},
};

// ProgPoW 0.9.2 (Epic Cash): EpicCash/progpow-rust pp_light/res/progpow_testvectors.json (v0.2.0), the canonical
// 0.9.2 spec vectors.
const Vector kProgpow092Vectors[] = {
    {0, "0000000000000000000000000000000000000000000000000000000000000000", 0x0000000000000000ull,
     "faeb1be51075b03a4ff44b335067951ead07a3b078539ace76fd56fc410557a3",
     "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b"},
    {49, "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b", 0x0000000006ff2c47ull,
     "c789c1180f890ec555ff42042913465481e8e6bc512cb981e1c1108dc3f2227d",
     "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922"},
    {50, "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922", 0x00000000076e482eull,
     "c7340542c2a06b3a7dc7222635f7cd402abf8b528ae971ddac6bbe2b0c7cb518",
     "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d"},
    {29950, "ac7b55e801511b77e11d52e9599206101550144525b5679f2dab19386f23dcce", 0x005d409dbc23a62aull,
     "07393d15805eb08ee6fc6cb3ad4ad1010533bd0ff92d6006850246829f18fd6e",
     "e43d7e0bdc8a4a3f6e291a5ed790b9fa1a0948a2b9e33c844888690847de19f5"},
};

void check_vectors(const Variant& var, const Vector* vs, size_t n) {
  std::map<int, std::shared_ptr<const Light>> lights;
  for (size_t k = 0; k < n; ++k) {
    const Vector& v = vs[k];
    const int e = epoch_for_height(var, v.height);
    if (!lights.count(e)) lights[e] = make_light(e, var.parents, var.dataset_init);
    const Result r = hash(*lights[e], v.height, hash_from_hex(v.header), v.nonce, var);
    CHECK_EQ(hash_to_hex(r.mix_hash), std::string(v.mix));
    CHECK_EQ(hash_to_hex(r.final_hash), std::string(v.final));
  }
}

TEST(firopow_hash_vectors) { check_vectors(kFiropow, kFiroVectors, sizeof kFiroVectors / sizeof kFiroVectors[0]); }

TEST(progpow092_hash_vectors) {
  check_vectors(kProgpow092, kProgpow092Vectors, sizeof kProgpow092Vectors / sizeof kProgpow092Vectors[0]);
}

TEST(firopow_epoch_clamp) {
  CHECK_EQ(epoch_for_height(kFiropow, 926 * 1300), 926);
  CHECK_EQ(epoch_for_height(kFiropow, 927 * 1300), 650);
  CHECK_EQ(epoch_for_height(kKawpow, 4601007), 613);
}

TEST(kawpow_hash_vectors) {
  std::map<int, std::shared_ptr<const Light>> lights;
  for (const auto& v : kVectors) {
    const int e = v.height / kEpochLength;
    if (!lights.count(e)) lights[e] = make_light(e);
    const Result r = hash(*lights[e], v.height, hash_from_hex(v.header), v.nonce);
    CHECK_EQ(hash_to_hex(r.mix_hash), std::string(v.mix));
    CHECK_EQ(hash_to_hex(r.final_hash), std::string(v.final));
  }
}
