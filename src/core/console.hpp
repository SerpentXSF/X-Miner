// Colored console output in the SerpentX / Gundam Wing cockpit palette.
// 24-bit ANSI escapes; Windows consoles are switched into VT mode at startup.
#pragma once

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace xm {

enum class Tone {
  Plain,   // armor ivory: normal text
  Dim,     // muted slate: timestamps, units, separators
  Radar,   // cyan: hashrate, unit ids
  Hud,     // magenta: frames, highlights
  Gold,    // V-fin gold: fee, share counters, kernel names
  Red,     // Wing red: rejects, errors
  Blue,    // Wing blue: jobs
  Ok,      // green: accepted, ready
};

class Console {
 public:
  static Console& get();

  void set_color(bool on) { color_ = on; }
  bool color() const { return color_; }
  void set_log_file(const std::string& path);

  // A styled fragment builder: Console::get().line({{Tone::Dim, "21:14:01 "}, {Tone::Ok, "LOCK"}, ...});
  struct Part {
    Tone tone;
    std::string text;
    bool bold = false;
  };
  void line(std::initializer_list<Part> parts);
  void raw(const std::string& text);  // already formatted text, written as is

  std::string paint(Tone tone, const std::string& text, bool bold = false) const;

 private:
  Console();
  bool color_ = true;
  std::mutex mu_;
  std::FILE* log_ = nullptr;
};

// Log helpers with a timestamp and a short tag, e.g. info("POOL", "connected to ...").
void log_line(Tone tag_tone, const char* tag, const std::string& msg, Tone msg_tone = Tone::Plain);
inline void info(const char* tag, const std::string& msg) { log_line(Tone::Radar, tag, msg); }
inline void warn(const char* tag, const std::string& msg) { log_line(Tone::Gold, tag, msg); }
inline void error(const char* tag, const std::string& msg) { log_line(Tone::Red, tag, msg, Tone::Red); }

std::string now_hms();

// The last events (share results, jobs, pool and fee changes) for the dashboard's combat log.
struct Event {
  std::string t, tag, msg;
};
void record_event(const std::string& tag, const std::string& msg);
std::vector<Event> recent_events();

}  // namespace xm
