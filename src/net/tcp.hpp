// Blocking TCP client with timeouts and a line reader, for stratum (newline-delimited JSON), optionally over TLS.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "net/tls.hpp"

namespace xm::net {

bool startup();  // WSAStartup on Windows; no-op elsewhere. Safe to call more than once.

class TcpLine {
 public:
  TcpLine() = default;
  ~TcpLine();
  TcpLine(const TcpLine&) = delete;
  TcpLine& operator=(const TcpLine&) = delete;

  // Resolves and connects (then runs the TLS handshake when tls is set); returns false with error() set on failure.
  bool connect(const std::string& host, uint16_t port, int timeout_ms, bool tls = false);
  void close();
  bool connected() const { return fd_ != kInvalid; }

  bool send_line(const std::string& line);  // appends '\n'
  // Waits up to timeout_ms for a full line. Returns 1 = line read, 0 = timeout, -1 = closed or error.
  int read_line(std::string& out, int timeout_ms);

  const std::string& error() const { return err_; }
  int latency_ms() const { return connect_ms_; }

 private:
#ifdef _WIN32
  using Sock = uintptr_t;
  static constexpr Sock kInvalid = ~static_cast<Sock>(0);
#else
  using Sock = int;
  static constexpr Sock kInvalid = -1;
#endif
  Sock fd_ = kInvalid;
  std::unique_ptr<Tls> tls_;
  std::string buf_;
  std::string err_;
  int connect_ms_ = 0;
};

}  // namespace xm::net
