#include "core/banner.hpp"

#include <cstdio>

#include "core/console.hpp"
#include "core/devfee.hpp"
#include "core/version.hpp"

namespace xm {

void print_banner(const std::string& algorithm) {
  auto& c = Console::get();
  const char* art[] = {
      " ██╗  ██╗      ███╗   ███╗██╗███╗   ██╗███████╗██████╗ ",
      " ╚██╗██╔╝      ████╗ ████║██║████╗  ██║██╔════╝██╔══██╗",
      "  ╚███╔╝ █████╗██╔████╔██║██║██╔██╗ ██║█████╗  ██████╔╝",
      "  ██╔██╗ ╚════╝██║╚██╔╝██║██║██║╚██╗██║██╔══╝  ██╔══██╗",
      " ██╔╝ ██╗      ██║ ╚═╝ ██║██║██║ ╚████║███████╗██║  ██║",
      " ╚═╝  ╚═╝      ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝",
  };
  std::string rule(78, '=');
  c.line({{Tone::Hud, "  " + rule}});
  for (const char* l : art) c.line({{Tone::Radar, "   " + std::string(l), true}});
  double fee = kFeePercent;
  if (const FeeTarget* t = fee_target(algorithm)) fee = t->percent;
  char feebuf[16];
  std::snprintf(feebuf, sizeof feebuf, "%.2f%%", fee);
  c.line({{Tone::Plain, "   v" XM_VERSION " "},
          {Tone::Dim, "· "},
          {Tone::Gold, XM_VENDOR " "},
          {Tone::Dim, "· "},
          {Tone::Plain, "open source "},
          {Tone::Dim, "· "},
          {Tone::Plain, "dev fee "},
          {Tone::Gold, feebuf},
          {Tone::Dim, "          ZERO SYSTEM ONLINE"}});
  c.line({{Tone::Hud, "  " + rule}});
}

void print_section(const std::string& title) {
  std::string t = "  --- " + title + " ";
  if (t.size() < 80) t += std::string(80 - t.size(), '-');
  Console::get().line({{Tone::Dim, t}});
}

}  // namespace xm
