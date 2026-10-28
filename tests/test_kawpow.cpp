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
