// TLS client for stratum+ssl pools: SChannel on Windows, OpenSSL (libssl loaded at run time) elsewhere.
//
// Pool certificates are not verified: stratum+ssl pools commonly use self-signed certificates, and every other
// miner accepts them too. TLS here protects against passive eavesdropping, not an active man in the middle.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace xm::net {

class Tls {
 public:
  virtual ~Tls() = default;
  // Runs the handshake on a connected, blocking socket.
  virtual bool handshake(uintptr_t sock, const std::string& host, int timeout_ms, std::string& err) = 0;
  // Sends all of data. False on error.
  virtual bool write(const char* data, size_t n, std::string& err) = 0;
  // Decrypted bytes already buffered (no socket read needed).
  virtual bool pending() const = 0;
  // Appends decrypted bytes to out after one socket read (call when the socket is readable, or when pending()).
  // Returns >0 bytes appended, 0 = need more data, -1 = closed or error (err set).
  virtual int read(std::string& out, std::string& err) = 0;

  static std::unique_ptr<Tls> create(std::string& err);  // nullptr if TLS is unavailable (err says why)
};

}  // namespace xm::net
