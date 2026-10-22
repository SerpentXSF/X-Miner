#include <cstdio>
#include <string>

#include "algo/registry.hpp"
#include "core/banner.hpp"
#include "core/config.hpp"
#include "core/console.hpp"
#include "core/version.hpp"
#include "gpu/gpu.hpp"

using namespace xm;

namespace {

std::string gb(size_t bytes) {
  char b[16];
  std::snprintf(b, sizeof b, "%.0f GB", bytes / 1073741824.0);
  return b;
}

int list_devices() {
  auto& c = Console::get();
  print_section("SYSTEM CHECK");
  std::string amd = gpu::amd_runtime_name(), nv = gpu::nvidia_runtime_name();
  c.line({{Tone::Radar, "  [AMD   ] "}, {amd.empty() ? Tone::Dim : Tone::Plain, amd.empty() ? "HIP runtime not found" : amd}});
  c.line({{Tone::Radar, "  [NVIDIA] "}, {nv.empty() ? Tone::Dim : Tone::Plain, nv.empty() ? "CUDA driver not found" : nv}});
  auto devs = gpu::all_devices();
  if (devs.empty()) {
    c.line({{Tone::Red, "  no supported GPU found"}});
    return 1;
  }
  for (size_t i = 0; i < devs.size(); ++i) {
    const auto& d = devs[i];
    char id[32];
    std::snprintf(id, sizeof id, "  [GPU %zu ] ", i);
    std::string algos;
    for (const auto& a : algorithms())
      if (supports(a, d)) algos += std::string(algos.empty() ? "" : ", ") + a.name;
    c.line({{Tone::Radar, id},
            {Tone::Plain, d.name + "  "},
            {Tone::Dim, d.arch + " · " + gb(d.memory) + " · bus " + std::to_string(d.pci_bus) + " · " +
                            std::to_string(d.compute_units) +
                            (d.vendor == gpu::Vendor::Nvidia ? " SM  " : (d.arch >= "gfx10" ? " WGP  " : " CU  "))},
            {algos.empty() ? Tone::Dim : Tone::Ok, algos.empty() ? "no kernel yet" : algos}});
  }
  return 0;
}

int list_algorithms() {
  auto& c = Console::get();
  print_section("ALGORITHMS");
  for (const auto& a : algorithms()) {
    std::string archs;
    for (const auto& s : a.archs) archs += (archs.empty() ? "" : " ") + s;
    c.line({{Tone::Radar, "  " + std::string(a.name)},
            {Tone::Dim, "  " + std::string(a.coin) + "  "},
            {a.implemented ? Tone::Ok : Tone::Gold, a.implemented ? "ready: " + archs : "in development"}});
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (auto err = parse_args(argc, argv, cfg)) {
    std::fprintf(stderr, "x-miner: %s\n", err->c_str());
    return 2;
  }
  Console::get().set_color(cfg.color && Console::get().color());
  if (!cfg.log_file.empty()) Console::get().set_log_file(cfg.log_file);
  if (cfg.help) {
    std::fputs(usage().c_str(), stdout);
    return 0;
  }
  if (cfg.version) {
    std::printf("%s %s (%s)\n", XM_NAME, XM_VERSION, XM_VENDOR);
    return 0;
  }
  print_banner(cfg.algorithm);
  if (cfg.list_devices) return list_devices();
  if (cfg.list_algorithms) return list_algorithms();

  const AlgorithmInfo* algo = find_algorithm(cfg.algorithm);
  if (!algo) {
    error("ALGO", cfg.algorithm.empty() ? "no --algorithm given (see --list-algorithms)"
                                        : "unknown algorithm '" + cfg.algorithm + "' (see --list-algorithms)");
    return 2;
  }
  if (!algo->implemented) {
    error("ALGO", std::string(algo->name) + " is still in development in this build");
    return 3;
  }
  return 0;
}
