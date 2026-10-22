#include "api/api.hpp"

#include <cstring>
#include <string>

#include "dashboard_html.hpp"  // generated at build time from web/dashboard.html
#include "logo_png.hpp"        // generated from assets/serpentx-logo.png
#include "net/tcp.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#define XM_CLOSE closesocket
using SockLen = int;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define XM_CLOSE ::close
using SockLen = socklen_t;
#endif

namespace xm {

namespace {
constexpr uintptr_t kBad = ~static_cast<uintptr_t>(0);
#ifdef _WIN32
using S = SOCKET;
#else
using S = int;
#endif
S sock(uintptr_t s) { return static_cast<S>(s); }
}  // namespace

ApiServer::ApiServer(std::string host, uint16_t port, SummaryFn summary)
    : host_(std::move(host)), port_(port), summary_(std::move(summary)) {}

ApiServer::~ApiServer() { stop(); }

bool ApiServer::start(std::string& error) {
  net::startup();
  S s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (static_cast<uintptr_t>(s) == kBad || static_cast<intptr_t>(s) < 0) {
    error = "cannot create API socket";
    return false;
  }
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    error = "bad --api-host " + host_;
    XM_CLOSE(s);
    return false;
  }
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(s, 16) != 0) {
    error = "API port " + std::to_string(port_) + " is in use (change it with --api-port)";
    XM_CLOSE(s);
    return false;
  }
  listen_fd_ = static_cast<uintptr_t>(s);
  thread_ = std::thread([this] { run(); });
  return true;
}

void ApiServer::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  if (listen_fd_ != kBad) XM_CLOSE(sock(listen_fd_));
  listen_fd_ = kBad;
}

void ApiServer::run() {
  while (!stop_) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(sock(listen_fd_), &set);
    timeval tv{0, 250000};
#ifdef _WIN32
    int r = select(0, &set, nullptr, nullptr, &tv);
#else
    int r = select(sock(listen_fd_) + 1, &set, nullptr, nullptr, &tv);
#endif
    if (r <= 0) continue;
    S c = ::accept(sock(listen_fd_), nullptr, nullptr);
    if (static_cast<intptr_t>(c) < 0 || static_cast<uintptr_t>(c) == kBad) continue;
    handle(static_cast<uintptr_t>(c));  // requests are tiny; serve inline
  }
}

void ApiServer::handle(uintptr_t client) {
  S c = sock(client);
#ifdef _WIN32
  DWORD to = 2000;
  setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&to), sizeof to);
#else
  timeval to{2, 0};
  setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);
#endif
  char buf[4096];
  int n = ::recv(c, buf, sizeof buf - 1, 0);
  if (n <= 0) {
    XM_CLOSE(c);
    return;
  }
  buf[n] = 0;
  std::string req(buf);
  std::string path;
  if (req.rfind("GET ", 0) == 0) path = req.substr(4, req.find(' ', 4) - 4);

  std::string body, type = "application/json";
  int code = 200;
  if (path == "/" || path == "/index.html") {
    body.assign(reinterpret_cast<const char*>(dashboard_html_bytes), dashboard_html_size);
    type = "text/html; charset=utf-8";
  } else if (path == "/logo.png" || path == "/favicon.ico") {
    body.assign(reinterpret_cast<const char*>(logo_png_bytes), logo_png_size);
    type = "image/png";
  } else if (path == "/api" || path == "/api/" || path == "/api/summary" || path == "/summary") {
    body = summary_().dump();
  } else {
    code = 404;
    body = R"({"error":"not found"})";
  }
  std::string head = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK" : " Not Found") +
                     "\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) +
                     "\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
  std::string out = head + body;
  size_t off = 0;
  while (off < out.size()) {
    int w = ::send(c, out.data() + off, static_cast<int>(out.size() - off), 0);
    if (w <= 0) break;
    off += static_cast<size_t>(w);
  }
  XM_CLOSE(c);
}

}  // namespace xm
