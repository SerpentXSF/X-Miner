#include "core/console.hpp"

#include <chrono>
#include <ctime>
#include <deque>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace xm {

namespace {

struct Rgb {
  int r, g, b;
};

Rgb rgb(Tone t) {
  switch (t) {
    case Tone::Plain: return {250, 235, 202};  // armor  #faebca
    case Tone::Dim:   return {93, 104, 150};   // slate  #5d6896
    case Tone::Radar: return {57, 224, 208};   // radar  #39e0d0
    case Tone::Hud:   return {255, 61, 139};   // HUD    #ff3d8b
    case Tone::Gold:  return {245, 197, 24};   // V-fin  #f5c518
    case Tone::Red:   return {232, 50, 60};    // Wing red #e8323c
    case Tone::Blue:  return {47, 111, 220};   // Wing blue #2f6fdc
    case Tone::Ok:    return {57, 224, 160};   // green  #39e0a0
  }
  return {250, 235, 202};
}

}  // namespace

Console& Console::get() {
  static Console c;
  return c;
}

Console::Console() {
#ifdef _WIN32
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  if (h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode)) {
    if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) color_ = false;
  } else {
    color_ = false;  // redirected to a file or pipe
  }
  SetConsoleOutputCP(CP_UTF8);
#endif
}

void Console::set_log_file(const std::string& path) {
  std::lock_guard<std::mutex> lk(mu_);
  if (log_) std::fclose(log_);
  log_ = path.empty() ? nullptr : std::fopen(path.c_str(), "a");
}

std::string Console::paint(Tone tone, const std::string& text, bool bold) const {
  if (!color_) return text;
  Rgb c = rgb(tone);
  std::string s = "\x1b[";
  if (bold) s += "1;";
  s += "38;2;" + std::to_string(c.r) + ";" + std::to_string(c.g) + ";" + std::to_string(c.b) + "m";
  return s + text + "\x1b[0m";
}

void Console::line(std::initializer_list<Part> parts) {
  std::string colored, plain;
  for (const auto& p : parts) {
    colored += paint(p.tone, p.text, p.bold);
    plain += p.text;
  }
  std::lock_guard<std::mutex> lk(mu_);
  std::fputs(colored.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
  if (log_) {
    std::fputs(plain.c_str(), log_);
    std::fputc('\n', log_);
    std::fflush(log_);
  }
}

void Console::raw(const std::string& text) {
  std::lock_guard<std::mutex> lk(mu_);
  std::fputs(text.c_str(), stdout);
  std::fflush(stdout);
}

std::string now_hms() {
  auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[16];
  std::strftime(buf, sizeof buf, "%H:%M:%S", &tm);
  return buf;
}

namespace {
std::mutex g_events_mu;
std::deque<Event> g_events;
}  // namespace

void record_event(const std::string& tag, const std::string& msg) {
  std::lock_guard<std::mutex> lk(g_events_mu);
  g_events.push_back({now_hms(), tag, msg});
  while (g_events.size() > 60) g_events.pop_front();
}

std::vector<Event> recent_events() {
  std::lock_guard<std::mutex> lk(g_events_mu);
  return {g_events.begin(), g_events.end()};
}

void log_line(Tone tag_tone, const char* tag, const std::string& msg, Tone msg_tone) {
  std::string t = tag;
  if (t.size() < 4) t.resize(4, ' ');
  Console::get().line({{Tone::Dim, "  " + now_hms() + " "}, {tag_tone, t, true}, {Tone::Dim, "  "}, {msg_tone, msg}});
}

}  // namespace xm
