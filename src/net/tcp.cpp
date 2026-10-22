#include "net/tcp.hpp"

#include "net/tls.hpp"

#include <chrono>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define XM_CLOSE closesocket
#define XM_ERRNO WSAGetLastError()
#else
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define XM_CLOSE ::close
#define XM_ERRNO errno
#endif

namespace xm::net {

bool startup() {
#ifdef _WIN32
  static bool ok = [] {
    WSADATA w;
    return WSAStartup(MAKEWORD(2, 2), &w) == 0;
  }();
  return ok;
#else
  return true;
#endif
}

namespace {

void set_nonblocking(uintptr_t fd, bool on) {
#ifdef _WIN32
  u_long mode = on ? 1 : 0;
  ioctlsocket(static_cast<SOCKET>(fd), FIONBIO, &mode);
#else
  int flags = fcntl(static_cast<int>(fd), F_GETFL, 0);
  fcntl(static_cast<int>(fd), F_SETFL, on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
#endif
}

// select() on one socket for read or write readiness. Returns >0 ready, 0 timeout, <0 error.
int wait_fd(uintptr_t fd, bool for_write, int timeout_ms) {
  fd_set set;
  FD_ZERO(&set);
#ifdef _WIN32
  FD_SET(static_cast<SOCKET>(fd), &set);
  int nfds = 0;
#else
  FD_SET(static_cast<int>(fd), &set);
  int nfds = static_cast<int>(fd) + 1;
#endif
  timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
  return select(nfds, for_write ? nullptr : &set, for_write ? &set : nullptr, nullptr, &tv);
}

}  // namespace

TcpLine::~TcpLine() { close(); }

void TcpLine::close() {
  tls_.reset();
  if (fd_ != kInvalid) XM_CLOSE(fd_);
  fd_ = kInvalid;
  buf_.clear();
}

bool TcpLine::connect(const std::string& host, uint16_t port, int timeout_ms, bool tls) {
  close();
  startup();
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
    err_ = "cannot resolve " + host;
    return false;
  }
  auto t0 = std::chrono::steady_clock::now();
  for (addrinfo* a = res; a; a = a->ai_next) {
    Sock s = static_cast<Sock>(socket(a->ai_family, a->ai_socktype, a->ai_protocol));
    if (s == kInvalid) continue;
    set_nonblocking(s, true);
    int r = ::connect(s, a->ai_addr, static_cast<socklen_t>(a->ai_addrlen));
    bool ok = r == 0;
    if (!ok && wait_fd(s, true, timeout_ms) > 0) {
      int so = 0;
      socklen_t len = sizeof so;
      getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so), &len);
      ok = so == 0;
    }
    if (ok) {
      set_nonblocking(s, false);
      int one = 1;
      setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
      fd_ = s;
      break;
    }
    XM_CLOSE(s);
  }
  freeaddrinfo(res);
  if (fd_ == kInvalid) {
    err_ = "cannot connect to " + host + ":" + std::to_string(port);
    return false;
  }
  connect_ms_ = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
  if (tls) {
    std::string err;
    auto t = Tls::create(err);
    if (!t || !t->handshake(static_cast<uintptr_t>(fd_), host, timeout_ms, err)) {
      err_ = host + ":" + std::to_string(port) + ": " + err;
      close();
      return false;
    }
    tls_ = std::move(t);
  }
  return true;
}

bool TcpLine::send_line(const std::string& line) {
  if (fd_ == kInvalid) return false;
  std::string data = line + "\n";
  if (tls_) {
    if (!tls_->write(data.data(), data.size(), err_)) {
      close();
      return false;
    }
    return true;
  }
  size_t off = 0;
  while (off < data.size()) {
    int n = ::send(fd_, data.data() + off, static_cast<int>(data.size() - off), 0);
    if (n <= 0) {
      err_ = "send failed (" + std::to_string(XM_ERRNO) + ")";
      close();
      return false;
    }
    off += static_cast<size_t>(n);
  }
  return true;
}

int TcpLine::read_line(std::string& out, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (;;) {
    auto nl = buf_.find('\n');
    if (nl != std::string::npos) {
      out = buf_.substr(0, nl);
      if (!out.empty() && out.back() == '\r') out.pop_back();
      buf_.erase(0, nl + 1);
      return 1;
    }
    if (fd_ == kInvalid) return -1;
    int left = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
    if (left <= 0) return 0;
    if (tls_ && tls_->pending()) {
      if (tls_->read(buf_, err_) < 0) {
        close();
        return -1;
      }
      continue;
    }
    int r = wait_fd(fd_, false, left);
    if (r == 0) return 0;
    if (r < 0) {
      err_ = "select failed";
      close();
      return -1;
    }
    if (tls_) {
      if (tls_->read(buf_, err_) < 0) {
        close();
        return -1;
      }
    } else {
      char tmp[65536];
      int n = ::recv(fd_, tmp, sizeof tmp, 0);
      if (n <= 0) {
        err_ = n == 0 ? "pool closed the connection" : "recv failed (" + std::to_string(XM_ERRNO) + ")";
        close();
        return -1;
      }
      buf_.append(tmp, static_cast<size_t>(n));
    }
    if (buf_.size() > (16u << 20)) {  // a pool never sends a 16 MB line; refuse to grow without bound
      err_ = "oversized message from pool";
      close();
      return -1;
    }
  }
}

}  // namespace xm::net
