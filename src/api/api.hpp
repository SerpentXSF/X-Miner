// Local HTTP API and web dashboard.
//   GET /                web dashboard (embedded HTML)
//   GET /api, /api/summary   JSON summary (SRBMiner-like shape, so existing monitors and HiveOS stats work)
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"

namespace xm {

class ApiServer {
 public:
  using SummaryFn = std::function<nlohmann::json()>;
  ApiServer(std::string host, uint16_t port, SummaryFn summary);
  ~ApiServer();
  bool start(std::string& error);
  void stop();

 private:
  void run();
  void handle(uintptr_t client);
  std::string host_;
  uint16_t port_;
  SummaryFn summary_;
  std::atomic<bool> stop_{false};
  uintptr_t listen_fd_ = ~static_cast<uintptr_t>(0);
  std::thread thread_;
};

}  // namespace xm
