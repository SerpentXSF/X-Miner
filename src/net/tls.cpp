#include "net/tls.hpp"

#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define SECURITY_WIN32
#include <winsock2.h>
#include <windows.h>
#include <schannel.h>
#include <security.h>
#else
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/time.h>
#endif

namespace xm::net {

#ifdef _WIN32

namespace {

class SchannelTls : public Tls {
 public:
  ~SchannelTls() override {
    if (have_ctx_) DeleteSecurityContext(&ctx_);
    if (have_cred_) FreeCredentialsHandle(&cred_);
  }

  bool handshake(uintptr_t sock, const std::string& host, int timeout_ms, std::string& err) override {
    s_ = static_cast<SOCKET>(sock);
    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
    SCHANNEL_CRED sc{};
    sc.dwVersion = SCHANNEL_CRED_VERSION;
    sc.dwFlags = SCH_CRED_MANUAL_CRED_VALIDATION | SCH_CRED_NO_DEFAULT_CREDS;
    if (AcquireCredentialsHandleA(nullptr, const_cast<char*>(UNISP_NAME_A), SECPKG_CRED_OUTBOUND, nullptr, &sc,
                                  nullptr, nullptr, &cred_, nullptr) != SEC_E_OK) {
      err = "TLS: cannot acquire SChannel credentials";
      return false;
    }
    have_cred_ = true;
    const DWORD req = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT | ISC_REQ_CONFIDENTIALITY |
                      ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_STREAM | ISC_REQ_MANUAL_CRED_VALIDATION;
    std::string in;
    bool first = true;
    for (;;) {
      SecBuffer ib[2] = {{static_cast<unsigned long>(in.size()), SECBUFFER_TOKEN, in.empty() ? nullptr : &in[0]},
                         {0, SECBUFFER_EMPTY, nullptr}};
      SecBufferDesc id{SECBUFFER_VERSION, 2, ib};
      SecBuffer ob{0, SECBUFFER_TOKEN, nullptr};
      SecBufferDesc od{SECBUFFER_VERSION, 1, &ob};
      DWORD attr = 0;
      SECURITY_STATUS st = InitializeSecurityContextA(&cred_, first ? nullptr : &ctx_, const_cast<char*>(host.c_str()),
                                                      req, 0, 0, first ? nullptr : &id, 0, &ctx_, &od, &attr, nullptr);
      if (first) have_ctx_ = true;
      first = false;
      if (ob.cbBuffer && ob.pvBuffer) {
        bool sent = send_all(static_cast<const char*>(ob.pvBuffer), ob.cbBuffer);
        FreeContextBuffer(ob.pvBuffer);
        if (!sent) {
          err = "TLS: handshake send failed";
          return false;
        }
      }
      if (st == SEC_E_INCOMPLETE_MESSAGE) {
        if (!recv_more(in)) {
          err = "TLS: connection closed during handshake";
          return false;
        }
        continue;
      }
      if (st == SEC_E_OK || st == SEC_I_CONTINUE_NEEDED || st == SEC_I_INCOMPLETE_CREDENTIALS) {
        if (ib[1].BufferType == SECBUFFER_EXTRA && ib[1].cbBuffer)
          in = in.substr(in.size() - ib[1].cbBuffer);
        else if (st != SEC_I_INCOMPLETE_CREDENTIALS)
          in.clear();
        if (st == SEC_E_OK) break;
        if (st == SEC_I_CONTINUE_NEEDED && in.empty() && !recv_more(in)) {
          err = "TLS: connection closed during handshake";
          return false;
        }
        continue;
      }
      char buf[64];
      std::snprintf(buf, sizeof buf, "TLS: handshake failed (0x%08lx)", static_cast<unsigned long>(st));
      err = buf;
      return false;
    }
    if (QueryContextAttributesA(&ctx_, SECPKG_ATTR_STREAM_SIZES, &sizes_) != SEC_E_OK) {
      err = "TLS: cannot query stream sizes";
      return false;
    }
    raw_ = in;  // application data that arrived with the last handshake message
    tv = 0;
    setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof tv);
    return true;
  }

  bool write(const char* data, size_t n, std::string& err) override {
    std::vector<char> msg;
    while (n) {
      size_t chunk = n < sizes_.cbMaximumMessage ? n : sizes_.cbMaximumMessage;
      msg.assign(sizes_.cbHeader + chunk + sizes_.cbTrailer, 0);
      std::memcpy(msg.data() + sizes_.cbHeader, data, chunk);
      SecBuffer b[4] = {{sizes_.cbHeader, SECBUFFER_STREAM_HEADER, msg.data()},
                        {static_cast<unsigned long>(chunk), SECBUFFER_DATA, msg.data() + sizes_.cbHeader},
                        {sizes_.cbTrailer, SECBUFFER_STREAM_TRAILER, msg.data() + sizes_.cbHeader + chunk},
                        {0, SECBUFFER_EMPTY, nullptr}};
      SecBufferDesc d{SECBUFFER_VERSION, 4, b};
      if (EncryptMessage(&ctx_, 0, &d, 0) != SEC_E_OK) {
        err = "TLS: encrypt failed";
        return false;
      }
      if (!send_all(msg.data(), b[0].cbBuffer + b[1].cbBuffer + b[2].cbBuffer)) {
        err = "TLS: send failed";
        return false;
      }
      data += chunk;
      n -= chunk;
    }
    return true;
  }

  bool pending() const override { return !plain_.empty() || decryptable_; }

  int read(std::string& out, std::string& err) override {
    if (!plain_.empty()) {
      out += plain_;
      int n = static_cast<int>(plain_.size());
      plain_.clear();
      return n;
    }
    if (!decryptable_) {
      char tmp[16384];
      int n = recv(s_, tmp, sizeof tmp, 0);
      if (n <= 0) {
        err = n == 0 ? "pool closed the connection" : "recv failed (" + std::to_string(WSAGetLastError()) + ")";
        return -1;
      }
      raw_.append(tmp, static_cast<size_t>(n));
    }
    decryptable_ = false;
    size_t before = out.size();
    while (!raw_.empty()) {
      SecBuffer b[4] = {{static_cast<unsigned long>(raw_.size()), SECBUFFER_DATA, &raw_[0]},
                        {0, SECBUFFER_EMPTY, nullptr},
                        {0, SECBUFFER_EMPTY, nullptr},
                        {0, SECBUFFER_EMPTY, nullptr}};
      SecBufferDesc d{SECBUFFER_VERSION, 4, b};
      SECURITY_STATUS st = DecryptMessage(&ctx_, &d, 0, nullptr);
      if (st == SEC_E_INCOMPLETE_MESSAGE) break;
      if (st == SEC_I_CONTEXT_EXPIRED) {
        err = "pool closed the TLS session";
        return -1;
      }
      if (st != SEC_E_OK && st != SEC_I_RENEGOTIATE) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "TLS: decrypt failed (0x%08lx)", static_cast<unsigned long>(st));
        err = buf;
        return -1;
      }
      std::string extra;
      for (auto& x : b) {
        if (x.BufferType == SECBUFFER_DATA && x.cbBuffer) out.append(static_cast<char*>(x.pvBuffer), x.cbBuffer);
        if (x.BufferType == SECBUFFER_EXTRA && x.cbBuffer) extra.assign(static_cast<char*>(x.pvBuffer), x.cbBuffer);
      }
      raw_.swap(extra);
      // TLS 1.3 post-handshake messages (session tickets) surface as SEC_I_RENEGOTIATE with no data; SChannel
      // handles them internally when the next DecryptMessage runs, so just keep going.
    }
    return static_cast<int>(out.size() - before);
  }

 private:
  bool send_all(const char* p, size_t n) {
    while (n) {
      int k = send(s_, p, static_cast<int>(n), 0);
      if (k <= 0) return false;
      p += k;
      n -= static_cast<size_t>(k);
    }
    return true;
  }
  bool recv_more(std::string& in) {
    char tmp[16384];
    int n = recv(s_, tmp, sizeof tmp, 0);
    if (n <= 0) return false;
    in.append(tmp, static_cast<size_t>(n));
    return true;
  }

  SOCKET s_ = INVALID_SOCKET;
  CredHandle cred_{};
  CtxtHandle ctx_{};
  bool have_cred_ = false, have_ctx_ = false;
  SecPkgContext_StreamSizes sizes_{};
  std::string raw_;    // received ciphertext not yet decrypted
  std::string plain_;  // decrypted, not yet returned
  bool decryptable_ = false;
};

}  // namespace

std::unique_ptr<Tls> Tls::create(std::string&) { return std::make_unique<SchannelTls>(); }

#else  // OpenSSL 1.1 / 3.x, loaded at run time so the binary has no libssl dependency.

namespace {

struct Ssl {
  void* lib = nullptr;
  const void* (*TLS_client_method)() = nullptr;
  void* (*SSL_CTX_new)(const void*) = nullptr;
  void (*SSL_CTX_free)(void*) = nullptr;
  void (*SSL_CTX_set_verify)(void*, int, void*) = nullptr;
  void* (*SSL_new)(void*) = nullptr;
  void (*SSL_free)(void*) = nullptr;
  int (*SSL_set_fd)(void*, int) = nullptr;
  long (*SSL_ctrl)(void*, int, long, void*) = nullptr;
  int (*SSL_connect)(void*) = nullptr;
  int (*SSL_read)(void*, void*, int) = nullptr;
  int (*SSL_write)(void*, const void*, int) = nullptr;
  int (*SSL_pending)(const void*) = nullptr;
  int (*SSL_get_error)(const void*, int) = nullptr;
  bool ok = false;

  Ssl() {
    for (const char* n : {"libssl.so.3", "libssl.so.1.1", "libssl.so"})
      if ((lib = dlopen(n, RTLD_NOW | RTLD_LOCAL))) break;
    if (!lib) return;
    auto sym = [&](auto& f, const char* name) {
      f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(dlsym(lib, name));
      return f != nullptr;
    };
    ok = sym(TLS_client_method, "TLS_client_method") && sym(SSL_CTX_new, "SSL_CTX_new") &&
         sym(SSL_CTX_free, "SSL_CTX_free") && sym(SSL_CTX_set_verify, "SSL_CTX_set_verify") &&
         sym(SSL_new, "SSL_new") && sym(SSL_free, "SSL_free") && sym(SSL_set_fd, "SSL_set_fd") &&
         sym(SSL_ctrl, "SSL_ctrl") && sym(SSL_connect, "SSL_connect") && sym(SSL_read, "SSL_read") &&
         sym(SSL_write, "SSL_write") && sym(SSL_pending, "SSL_pending") && sym(SSL_get_error, "SSL_get_error");
  }
  static Ssl& get() {
    static Ssl s;
    return s;
  }
};

constexpr int kSslErrorWantRead = 2, kSslErrorWantWrite = 3, kSslErrorZeroReturn = 6;
constexpr int kCtrlSetTlsextHostname = 55, kNametypeHostName = 0;

class OpenSslTls : public Tls {
 public:
  ~OpenSslTls() override {
    auto& L = Ssl::get();
    if (ssl_) L.SSL_free(ssl_);
    if (ctx_) L.SSL_CTX_free(ctx_);
  }

  bool handshake(uintptr_t sock, const std::string& host, int timeout_ms, std::string& err) override {
    auto& L = Ssl::get();
    fd_ = static_cast<int>(sock);
    // Socket timeouts bound every blocking SSL call (handshake now, record reads later).
    timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    ctx_ = L.SSL_CTX_new(L.TLS_client_method());
    if (!ctx_) {
      err = "TLS: SSL_CTX_new failed";
      return false;
    }
    L.SSL_CTX_set_verify(ctx_, 0 /* SSL_VERIFY_NONE */, nullptr);
    ssl_ = L.SSL_new(ctx_);
    if (!ssl_ || L.SSL_set_fd(ssl_, fd_) != 1) {
      err = "TLS: SSL_new failed";
      return false;
    }
    L.SSL_ctrl(ssl_, kCtrlSetTlsextHostname, kNametypeHostName, const_cast<char*>(host.c_str()));
    if (L.SSL_connect(ssl_) != 1) {
      err = "TLS: handshake failed";
      return false;
    }
    timeval rt{5, 0};  // a record that has started arriving completes quickly
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof rt);
    return true;
  }

  bool write(const char* data, size_t n, std::string& err) override {
    auto& L = Ssl::get();
    while (n) {
      int k = L.SSL_write(ssl_, data, static_cast<int>(n));
      if (k <= 0) {
        err = "TLS: send failed";
        return false;
      }
      data += k;
      n -= static_cast<size_t>(k);
    }
    return true;
  }

  bool pending() const override { return Ssl::get().SSL_pending(ssl_) > 0; }

  int read(std::string& out, std::string& err) override {
    auto& L = Ssl::get();
    char tmp[16384];
    int n = L.SSL_read(ssl_, tmp, sizeof tmp);
    if (n > 0) {
      out.append(tmp, static_cast<size_t>(n));
      return n;
    }
    int e = L.SSL_get_error(ssl_, n);
    if (e == kSslErrorWantRead || e == kSslErrorWantWrite) return 0;  // e.g. a session ticket, no app data
    err = e == kSslErrorZeroReturn ? "pool closed the TLS session" : "TLS: read failed";
    return -1;
  }

 private:
  int fd_ = -1;
  void* ctx_ = nullptr;
  void* ssl_ = nullptr;
};

}  // namespace

std::unique_ptr<Tls> Tls::create(std::string& err) {
  if (!Ssl::get().ok) {
    err = "TLS needs OpenSSL (libssl.so.3 or libssl.so.1.1), which was not found";
    return nullptr;
  }
  return std::make_unique<OpenSslTls>();
}

#endif

}  // namespace xm::net
