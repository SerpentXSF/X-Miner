#include "gpu/tuning.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>

#include "core/console.hpp"
#include "gpu/dynlib.hpp"

#ifndef _WIN32
#include <dirent.h>
#endif

namespace xm::gpu {

namespace {

std::string unit_name(size_t i) { return "unit " + std::to_string(i); }

// ---------------------------------------------------------------- NVML
struct NvmlPci {  // nvmlPciInfo_t (v3)
  char busIdLegacy[16];
  unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
  char busId[32];
};

struct NvmlClockOffset {  // nvmlClockOffset_t (v1, NVML 12.4+)
  unsigned version;
  int type;      // nvmlClockType_t: 0 graphics, 2 memory
  int pstate;    // nvmlPstates_t: 0 = P0, 2 = P2 (CUDA compute runs in P2 on GeForce)
  int offset_mhz, min_mhz, max_mhz;
};

struct NvmlTune {
  DynLib lib;
  using H = void*;
  int (*init)() = nullptr;
  int (*count)(unsigned*) = nullptr;
  int (*by_index)(unsigned, H*) = nullptr;
  int (*pci)(H, NvmlPci*) = nullptr;
  const char* (*err)(int) = nullptr;
  int (*lock_gpu)(H, unsigned, unsigned) = nullptr;
  int (*reset_gpu)(H) = nullptr;
  int (*lock_mem)(H, unsigned, unsigned) = nullptr;
  int (*reset_mem)(H) = nullptr;
  int (*get_pl)(H, unsigned*) = nullptr;
  int (*get_pl_default)(H, unsigned*) = nullptr;
  int (*get_pl_range)(H, unsigned*, unsigned*) = nullptr;
  int (*set_pl)(H, unsigned) = nullptr;
  int (*get_core_off)(H, int*) = nullptr;
  int (*set_core_off)(H, int) = nullptr;
  int (*get_mem_off)(H, int*) = nullptr;
  int (*set_mem_off)(H, int) = nullptr;
  int (*get_offsets)(H, NvmlClockOffset*) = nullptr;  // newer per-P-state API (preferred)
  int (*set_offsets)(H, NvmlClockOffset*) = nullptr;
  std::map<int, H> by_bus;
  bool ready = false;

  NvmlTune() {
#ifdef _WIN32
    if (!lib.open("nvml.dll")) lib.open("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
    lib.open("libnvidia-ml.so.1");
#endif
    if (!lib.ok() || !lib.sym("nvmlInit_v2", init) || !lib.sym("nvmlDeviceGetCount_v2", count) ||
        !lib.sym("nvmlDeviceGetHandleByIndex_v2", by_index) || !lib.sym("nvmlDeviceGetPciInfo_v3", pci))
      return;
    lib.sym("nvmlErrorString", err);
    lib.sym("nvmlDeviceSetGpuLockedClocks", lock_gpu);
    lib.sym("nvmlDeviceResetGpuLockedClocks", reset_gpu);
    lib.sym("nvmlDeviceSetMemoryLockedClocks", lock_mem);
    lib.sym("nvmlDeviceResetMemoryLockedClocks", reset_mem);
    lib.sym("nvmlDeviceGetPowerManagementLimit", get_pl);
    lib.sym("nvmlDeviceGetPowerManagementDefaultLimit", get_pl_default);
    lib.sym("nvmlDeviceGetPowerManagementLimitConstraints", get_pl_range);
    lib.sym("nvmlDeviceSetPowerManagementLimit", set_pl);
    lib.sym("nvmlDeviceGetGpcClkVfOffset", get_core_off);
    lib.sym("nvmlDeviceSetGpcClkVfOffset", set_core_off);
    lib.sym("nvmlDeviceGetMemClkVfOffset", get_mem_off);
    lib.sym("nvmlDeviceSetMemClkVfOffset", set_mem_off);
    lib.sym("nvmlDeviceGetClockOffsets", get_offsets);
    lib.sym("nvmlDeviceSetClockOffsets", set_offsets);
    if (init() != 0) return;
    unsigned n = 0;
    count(&n);
    for (unsigned i = 0; i < n; ++i) {
      H h = nullptr;
      NvmlPci p{};
      if (by_index(i, &h) == 0 && pci(h, &p) == 0) by_bus[static_cast<int>(p.bus)] = h;
    }
    ready = true;
  }
  std::string why(int rc) const {
    if (rc == 4) return "needs administrator rights (run the miner as administrator / root)";
    if (rc == 3) return "not supported by this GPU or driver";
    return err ? std::string(err(rc)) : "NVML error " + std::to_string(rc);
  }
};

#ifndef _WIN32
// ---------------------------------------------------------------- amdgpu sysfs
std::string read_line(const std::string& path) {
  std::ifstream f(path);
  std::string s;
  std::getline(f, s);
  return s;
}
bool write_text(const std::string& path, const std::string& text) {
  std::ofstream f(path);
  if (!f) return false;
  f << text << "\n";
  return static_cast<bool>(f.flush());
}
// PCI bus -> /sys/class/drm/cardN/device
std::map<int, std::string> amdgpu_devices() {
  std::map<int, std::string> out;
  for (int card = 0; card < 32; ++card) {
    std::string dev = "/sys/class/drm/card" + std::to_string(card) + "/device";
    std::ifstream ue(dev + "/uevent");
    std::string line, slot;
    bool amd = false;
    while (std::getline(ue, line)) {
      if (line.rfind("DRIVER=amdgpu", 0) == 0) amd = true;
      if (line.rfind("PCI_SLOT_NAME=", 0) == 0) slot = line.substr(14);
    }
    if (amd && slot.size() >= 7) out[static_cast<int>(std::strtol(slot.substr(5, 2).c_str(), nullptr, 16))] = dev;
  }
  return out;
}
std::string hwmon_of(const std::string& dev) {
  if (DIR* d = opendir((dev + "/hwmon").c_str())) {
    std::string h;
    while (dirent* e = readdir(d))
      if (std::string(e->d_name).rfind("hwmon", 0) == 0) h = dev + "/hwmon/" + e->d_name;
    closedir(d);
    return h;
  }
  return {};
}
#endif

}  // namespace

struct Tuner::Impl {
  std::vector<DeviceInfo> devs;
  std::unique_ptr<NvmlTune> nvml;
  std::vector<std::function<void()>> undo;  // run in reverse order on restore
  std::mutex mu;
#ifndef _WIN32
  std::map<int, std::string> amd_dev;
#endif
};

Tuner::Tuner(const std::vector<DeviceInfo>& devices) : impl_(std::make_unique<Impl>()) { impl_->devs = devices; }

Tuner::~Tuner() { restore(); }

void Tuner::restore() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  if (impl_->undo.empty()) return;
  for (auto it = impl_->undo.rbegin(); it != impl_->undo.rend(); ++it) (*it)();
  impl_->undo.clear();
  info("TUNE", "GPU clocks, offsets and power limits restored");
}

bool Tuner::apply(size_t index, const TuneRequest& req) {
  if (!req.any() || index >= impl_->devs.size()) return false;
  std::lock_guard<std::mutex> lk(impl_->mu);
  const DeviceInfo& d = impl_->devs[index];
  const std::string unit = unit_name(index);
  bool applied = false;
  auto ok = [&](const std::string& what) {
    info("TUNE", unit + ": " + what);
    applied = true;
  };
  auto fail = [&](const std::string& what, const std::string& why) { warn("TUNE", unit + ": " + what + " failed: " + why); };

  if (d.vendor == Vendor::Nvidia) {
    if (!impl_->nvml) impl_->nvml = std::make_unique<NvmlTune>();
    NvmlTune& N = *impl_->nvml;
    auto hit = N.by_bus.find(d.pci_bus);
    if (!N.ready || hit == N.by_bus.end()) {
      warn("TUNE", unit + ": NVML not available, tuning skipped");
      return false;
    }
    void* h = hit->second;
    if (req.power_w || req.power_pct) {
      unsigned cur = 0, def = 0, lo = 0, hi = 0;
      if (!N.set_pl || !N.get_pl || N.get_pl(h, &cur) != 0) {
        fail("power limit", "not available");
      } else {
        if (N.get_pl_default) N.get_pl_default(h, &def);
        if (N.get_pl_range) N.get_pl_range(h, &lo, &hi);
        double want = req.power_w ? *req.power_w * 1000.0 : (def ? def : cur) * *req.power_pct / 100.0;
        unsigned mw = static_cast<unsigned>(std::lround(want));
        if (lo && hi) mw = std::min(std::max(mw, lo), hi);
        int rc = N.set_pl(h, mw);
        if (rc == 0) {
          char buf[96];
          std::snprintf(buf, sizeof buf, "power limit %.0f W (was %.0f W)", mw / 1000.0, cur / 1000.0);
          ok(buf);
          impl_->undo.push_back([&N, h, cur] { N.set_pl(h, cur); });
        } else {
          fail("power limit", N.why(rc));
        }
      }
    }
    auto offset = [&](const std::optional<int>& v, int clock_type, int (*get)(void*, int*), int (*set)(void*, int),
                      const char* what) {
      if (!v) return;
      // Newer drivers: per-P-state offsets (P0 and P2, the compute state). Older: one V/F curve offset.
      if (N.get_offsets && N.set_offsets) {
        int done = 0, last_rc = 3;
        for (int ps : {0, 2}) {
          NvmlClockOffset o{};
          o.version = static_cast<unsigned>(sizeof o) | (1u << 24);
          o.type = clock_type;
          o.pstate = ps;
          if ((last_rc = N.get_offsets(h, &o)) != 0) continue;
          NvmlClockOffset prev = o;
          o.offset_mhz = *v;
          if ((last_rc = N.set_offsets(h, &o)) != 0) continue;
          ++done;
          impl_->undo.push_back([&N, h, prev]() mutable { N.set_offsets(h, &prev); });
        }
        if (done) return ok(std::string(what) + " " + std::to_string(*v) + " MHz");
        if (!get || !set) return fail(what, N.why(last_rc));
      }
      int cur = 0, rc = (!set || !get) ? 3 : get(h, &cur);
      if (rc != 0) return fail(what, N.why(rc));
      if ((rc = set(h, *v)) != 0) return fail(what, N.why(rc));
      ok(std::string(what) + " " + std::to_string(*v) + " MHz (was " + std::to_string(cur) + ")");
      impl_->undo.push_back([set, h, cur] { set(h, cur); });
    };
    offset(req.core_offset_mhz, 0, N.get_core_off, N.set_core_off, "core clock offset");
    offset(req.mem_offset_mhz, 2, N.get_mem_off, N.set_mem_off, "memory clock offset");
    if (req.core_lock_mhz) {
      unsigned f = static_cast<unsigned>(*req.core_lock_mhz);
      int rc = N.lock_gpu ? N.lock_gpu(h, f, f) : 3;
      if (rc == 0) {
        ok("core clock locked at " + std::to_string(f) + " MHz");
        impl_->undo.push_back([&N, h] { N.reset_gpu(h); });
      } else {
        fail("core clock lock", N.why(rc));
      }
    }
    if (req.mem_lock_mhz) {
      unsigned f = static_cast<unsigned>(*req.mem_lock_mhz);
      int rc = N.lock_mem ? N.lock_mem(h, f, f) : 3;
      if (rc == 0) {
        ok("memory clock locked at " + std::to_string(f) + " MHz");
        impl_->undo.push_back([&N, h] { N.reset_mem(h); });
      } else {
        fail("memory clock lock", N.why(rc));
      }
    }
    return applied;
  }

#ifdef _WIN32
  warn("TUNE", unit + ": AMD tuning on Windows is not supported yet; set clocks and power in AMD Adrenalin");
  return false;
#else
  if (impl_->amd_dev.empty()) impl_->amd_dev = amdgpu_devices();
  auto it = impl_->amd_dev.find(d.pci_bus);
  if (it == impl_->amd_dev.end()) {
    warn("TUNE", unit + ": amdgpu sysfs device not found, tuning skipped");
    return false;
  }
  const std::string dev = it->second, hw = hwmon_of(dev);
  if (req.power_w || req.power_pct) {
    std::string cap = hw + "/power1_cap", cur = read_line(cap), def = read_line(hw + "/power1_cap_default");
    if (cur.empty()) {
      fail("power limit", "power1_cap not found");
    } else {
      double base = std::atof((def.empty() ? cur : def).c_str());
      double uw = req.power_w ? *req.power_w * 1e6 : base * *req.power_pct / 100.0;
      if (write_text(cap, std::to_string(static_cast<long long>(uw)))) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "power limit %.0f W (was %.0f W)", uw / 1e6, std::atof(cur.c_str()) / 1e6);
        ok(buf);
        impl_->undo.push_back([cap, cur] { write_text(cap, cur); });
      } else {
        fail("power limit", "cannot write " + cap + " (needs root)");
      }
    }
  }
  if (req.core_lock_mhz || req.mem_lock_mhz) {
    // Overdrive: highest core / memory DPM state ("s 1" / "m 1"), then commit. Needs amdgpu.ppfeaturemask with OD.
    const std::string od = dev + "/pp_od_clk_voltage";
    bool good = true;
    if (req.core_lock_mhz) good = good && write_text(od, "s 1 " + std::to_string(*req.core_lock_mhz));
    if (req.mem_lock_mhz) good = good && write_text(od, "m 1 " + std::to_string(*req.mem_lock_mhz));
    good = good && write_text(od, "c");
    if (good) {
      ok("overdrive max clocks set" + (req.core_lock_mhz ? " core " + std::to_string(*req.core_lock_mhz) + " MHz" : "") +
         (req.mem_lock_mhz ? " memory " + std::to_string(*req.mem_lock_mhz) + " MHz" : ""));
      impl_->undo.push_back([od] { write_text(od, "r"); write_text(od, "c"); });
    } else {
      fail("clock limits", "cannot write " + od + " (needs root and overdrive enabled)");
    }
  }
  if (req.core_offset_mhz || req.mem_offset_mhz) warn("TUNE", unit + ": clock offsets are NVIDIA-only for now");
  return applied;
#endif
}

}  // namespace xm::gpu
