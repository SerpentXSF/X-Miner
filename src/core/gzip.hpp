// Minimal gzip (RFC 1952) writer: one DEFLATE block with dynamic Huffman codes over literals (no LZ77 matching).
// Enough for share proofs, whose bytes are near-random but low-entropy; no zlib dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xm {

std::vector<uint8_t> gzip_compress(const uint8_t* data, size_t n);
uint32_t crc32(const uint8_t* data, size_t n);

}  // namespace xm
