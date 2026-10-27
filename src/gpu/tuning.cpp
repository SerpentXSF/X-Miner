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
  int (*num_fans)(H, unsigned*) = nullptr;
  int (*set_fan)(H, unsigned, unsigned) = nullptr;
  int (*default_fan)(H, unsigned) = nullptr;
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
    lib.sym("nvmlDeviceGetNumFans", num_fans);
    lib.sym("nvmlDeviceSetFanSpeed_v2", set_fan);
    lib.sym("nvmlDeviceSetDefaultFanSpeed_v2", default_fan);
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

#ifdef _WIN32
// ---------------------------------------------------------------- ADL Overdrive8 (AMD, Windows)
// Layouts and IDs from AMD's ADL SDK (github.com/GPUOpen-LibrariesAndSDKs/display-library, include/adl_defines.h and
// adl_structures.h; Copyright (c) Advanced Micro Devices, Inc., MIT licence). The set procedure follows the SDK's
// Overdrive8 sample: switch the optimized power mode to manual first, request only the settings being changed.
enum {
  kOd8GfxclkFmax = 0, kOd8GfxclkFmin = 1, kOd8UclkFmax = 8, kOd8PowerPercentage = 9, kOd8FanMinSpeed = 10,
  kOd8FanZeroRpm = 15, kOd8FanCurveTemp1 = 19, kOd8FanCurveSpeed1 = 20, kOd8FanCurvePoints = 5, kOd8UclkFmin = 34,
  kOd8OptimizedPowerMode = 36, kOd8OdVoltage = 37, kOd8Count = 77,
  kAdlOd8FanSpeedMin = 1 << 5, kAdlOd8FanZeroRpmControl = 1 << 9, kAdlOd8FanCurve = 1 << 13,
  kAdlOd8OptimizedGpuPowerMode = 1 << 16,
};
struct AdlAdapterInfoT {
  int iSize, iAdapterIndex;
  char strUDID[256];
  int iBusNumber, iDeviceNumber, iFunctionNumber, iVendorID;
  char strAdapterName[256], strDisplayName[256];
  int iPresent, iExist;
  char strDriverPath[256], strDriverPathExt[256], strPNPString[256];
  int iOSDisplayIndex;
};
struct Od8SingleInit {
  int featureID, minValue, maxValue, defaultValue;
};
struct Od8Init {  // ADLOD8InitSetting
  int count;
  int caps;
  Od8SingleInit table[kOd8Count];
};
struct Od8Current {  // ADLOD8CurrentSetting
  int count;
  int table[kOd8Count];
};
struct Od8SingleSet {
  int value, requested, reset;
};
struct Od8Set {  // ADLOD8SetSetting
  int count;
  Od8SingleSet table[kOd8Count];
};
void* __stdcall adl_alloc_t(int n) { return std::malloc(static_cast<size_t>(n)); }

// The fixed-size Overdrive8 calls (the caller sets count = OD8_COUNT): the X2 variants return ADL_ERR_NULL_POINTER
// on RDNA4 with Adrenalin 25.x, the fixed-size ones work.
struct AdlOd8 {
  DynLib lib;
  void* ctx = nullptr;
  int (*init_get)(void*, int, Od8Init*) = nullptr;
  int (*cur_get)(void*, int, Od8Current*) = nullptr;
  int (*set)(void*, int, Od8Set*, Od8Current*) = nullptr;
  std::map<int, int> adapter_by_bus;  // first adapter per PCI bus with Overdrive8
  bool ready = false;

  AdlOd8() {
    if (!lib.open("atiadlxx.dll")) return;
    int (*create)(void* (__stdcall*)(int), int, void**) = nullptr;
    int (*num)(void*, int*) = nullptr;
    int (*info)(void*, AdlAdapterInfoT*, int) = nullptr;
    if (!lib.sym("ADL2_Main_Control_Create", create) || !lib.sym("ADL2_Adapter_NumberOfAdapters_Get", num) ||
        !lib.sym("ADL2_Adapter_AdapterInfo_Get", info) || !lib.sym("ADL2_Overdrive8_Init_Setting_Get", init_get) ||
        !lib.sym("ADL2_Overdrive8_Current_Setting_Get", cur_get) || !lib.sym("ADL2_Overdrive8_Setting_Set", set))
      return;
    if (create(adl_alloc_t, 1, &ctx) != 0) return;
    int n = 0;
    num(ctx, &n);
    if (n <= 0) return;
    std::vector<AdlAdapterInfoT> ai(static_cast<size_t>(n));
    for (auto& a : ai) a.iSize = sizeof(AdlAdapterInfoT);
    if (info(ctx, ai.data(), static_cast<int>(sizeof(AdlAdapterInfoT) * ai.size())) != 0) return;
    for (const auto& a : ai) {
      if (adapter_by_bus.count(a.iBusNumber)) continue;
      auto init = std::make_unique<Od8Init>();
      init->count = kOd8Count;
      if (init_get(ctx, a.iAdapterIndex, init.get()) == 0 && init->caps != 0) adapter_by_bus[a.iBusNumber] = a.iAdapterIndex;
    }
    ready = true;
  }
  // Ranges/defaults and current values; false if the adapter has no Overdrive8.
  bool read(int adapter, int& caps, std::vector<Od8SingleInit>& init, std::vector<int>& cur) {
    auto i = std::make_unique<Od8Init>();
    auto c = std::make_unique<Od8Current>();
    i->count = kOd8Count;
    c->count = kOd8Count;
    if (init_get(ctx, adapter, i.get()) != 0 || cur_get(ctx, adapter, c.get()) != 0 || i->caps == 0) return false;
    caps = i->caps;
    init.assign(i->table, i->table + kOd8Count);
    cur.assign(c->table, c->table + kOd8Count);
    return true;
  }
  // Requests the given (id, value) pairs; reset ids go back to their defaults.
  int apply(int adapter, const std::vector<std::pair<int, int>>& values, const std::vector<int>& reset_ids,
            const std::vector<Od8SingleInit>& init) {
    Od8Set s{};
    Od8Current out{};
    s.count = kOd8Count;
    out.count = kOd8Count;
    for (auto [id, v] : values) {
      s.table[id].requested = 1;
      s.table[id].value = v;
    }
    for (int id : reset_ids) {
      s.table[id].requested = 1;
      s.table[id].reset = 1;
      s.table[id].value = id < static_cast<int>(init.size()) ? init[id].defaultValue : 0;
    }
    return set(ctx, adapter, &s, &out);
  }
};
#endif

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
#ifdef _WIN32
  std::unique_ptr<AdlOd8> od8;
#else
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
  info("TUNE", "GPU clocks, offsets, power limits and fans restored");
}

void Tuner::report(size_t index) {
  if (index >= impl_->devs.size()) return;
  const DeviceInfo& d = impl_->devs[index];
  const std::string unit = unit_name(index) + " " + d.name;
  if (d.vendor == Vendor::Nvidia) {
    if (!impl_->nvml) impl_->nvml = std::make_unique<NvmlTune>();
    NvmlTune& N = *impl_->nvml;
    auto hit = N.by_bus.find(d.pci_bus);
    if (!N.ready || hit == N.by_bus.end()) return warn("TUNE", unit + ": NVML not available");
    unsigned cur = 0, def = 0, lo = 0, hi = 0;
    if (N.get_pl) N.get_pl(hit->second, &cur);
    if (N.get_pl_default) N.get_pl_default(hit->second, &def);
    if (N.get_pl_range) N.get_pl_range(hit->second, &lo, &hi);
    char buf[160];
    unsigned fans = 0;
    if (N.num_fans) N.num_fans(hit->second, &fans);
    std::snprintf(buf, sizeof buf, ": power limit %.0f W (default %.0f, range %.0f-%.0f W), %u fan(s)", cur / 1e3,
                  def / 1e3, lo / 1e3, hi / 1e3, fans);
    return info("TUNE", unit + buf);
  }
#ifdef _WIN32
  if (!impl_->od8) impl_->od8 = std::make_unique<AdlOd8>();
  AdlOd8& A = *impl_->od8;
  auto at = A.adapter_by_bus.find(d.pci_bus);
  int caps = 0;
  std::vector<Od8SingleInit> init;
  std::vector<int> cur;
  if (!A.ready || at == A.adapter_by_bus.end() || !A.read(at->second, caps, init, cur))
    return warn("TUNE", unit + ": AMD Overdrive (ADL) not available");
  char buf[64];
  std::snprintf(buf, sizeof buf, ": Overdrive8 caps 0x%08x, %zu settings", static_cast<unsigned>(caps), init.size());
  info("TUNE", unit + buf);
  const std::pair<int, const char*> shown[] = {{kOd8GfxclkFmax, "core clock offset MHz"}, {kOd8GfxclkFmin, "min core clock MHz"},
                                               {kOd8UclkFmax, "max memory clock MHz"}, {kOd8PowerPercentage, "power limit % offset"},
                                               {kOd8UclkFmin, "min memory clock MHz"}, {kOd8OptimizedPowerMode, "power mode (3 = manual)"},
                                               {kOd8OdVoltage, "core voltage offset mV"}, {kOd8FanMinSpeed, "fan min speed"}, {kOd8FanZeroRpm, "fan zero RPM"},
                                               {kOd8FanCurveTemp1, "fan curve 1 temp C"}, {kOd8FanCurveSpeed1, "fan curve 1 speed %"},
                                               {kOd8FanCurveTemp1 + 2, "fan curve 2 temp C"}, {kOd8FanCurveSpeed1 + 2, "fan curve 2 speed %"},
                                               {kOd8FanCurveTemp1 + 4, "fan curve 3 temp C"}, {kOd8FanCurveSpeed1 + 4, "fan curve 3 speed %"},
                                               {kOd8FanCurveTemp1 + 6, "fan curve 4 temp C"}, {kOd8FanCurveSpeed1 + 6, "fan curve 4 speed %"},
                                               {kOd8FanCurveTemp1 + 8, "fan curve 5 temp C"}, {kOd8FanCurveSpeed1 + 8, "fan curve 5 speed %"}};
  auto row = [&](int id, const std::string& name) {
    char line[160];
    std::snprintf(line, sizeof line, "  %-26s current %6d  default %6d  range %6d to %6d", name.c_str(),
                  id < static_cast<int>(cur.size()) ? cur[id] : 0, init[id].defaultValue, init[id].minValue, init[id].maxValue);
    info("TUNE", line);
  };
  std::vector<bool> listed(init.size(), false);
  for (auto [id, name] : shown) {
    if (id >= static_cast<int>(init.size())) continue;
    row(id, name);
    listed[id] = true;
  }
  // The rest of the table (ADL OD8 setting ids) where the driver offers a range.
  for (int id = 0; id < static_cast<int>(init.size()); ++id)
    if (!listed[id] && init[id].minValue != init[id].maxValue) row(id, "OD8 setting " + std::to_string(id));
#else
  info("TUNE", unit + ": amdgpu sysfs (power1_cap, pp_od_clk_voltage)");
#endif
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
    if (req.fan_pct) {
      unsigned fans = 0;
      int rc = (!N.num_fans || !N.set_fan || !N.default_fan) ? 3 : N.num_fans(h, &fans);
      unsigned done = 0;
      for (unsigned f = 0; rc == 0 && f < fans; ++f) {
        if ((rc = N.set_fan(h, f, static_cast<unsigned>(*req.fan_pct))) != 0) break;
        ++done;
        impl_->undo.push_back([&N, h, f] { N.default_fan(h, f); });
      }
      if (rc == 0 && done)
        ok("fan speed " + std::to_string(*req.fan_pct) + "% (" + std::to_string(done) + " fan" + (done > 1 ? "s)" : ")"));
      else
        fail("fan speed", rc == 0 ? "no controllable fans" : N.why(rc));
    }
    if (req.volt_offset_mv)
      warn("TUNE", unit + ": NVIDIA has no voltage control through NVML; lock the core clock (--gpu-cclock) to run at "
                          "that point of the voltage curve");
    return applied;
  }

#ifdef _WIN32
  if (!impl_->od8) impl_->od8 = std::make_unique<AdlOd8>();
  AdlOd8& A = *impl_->od8;
  auto at = A.adapter_by_bus.find(d.pci_bus);
  int caps = 0;
  std::vector<Od8SingleInit> init;
  std::vector<int> cur;
  if (!A.ready || at == A.adapter_by_bus.end() || !A.read(at->second, caps, init, cur)) {
    warn("TUNE", unit + ": AMD Overdrive (ADL) not available, tuning skipped");
    return false;
  }
  const int adapter = at->second;
  // Power: Overdrive8 takes a percentage offset from the default board power (Adrenalin's "Power Limit %").
  std::vector<std::pair<int, int>> want;
  auto in_range = [&](int id, int v, const char* what) {
    if (id >= static_cast<int>(init.size()) || init[id].minValue > v || init[id].maxValue < v) {
      fail(what, "out of range (" + std::to_string(id < static_cast<int>(init.size()) ? init[id].minValue : 0) + " to " +
                     std::to_string(id < static_cast<int>(init.size()) ? init[id].maxValue : 0) + ")");
      return false;
    }
    return true;
  };
  if (req.power_w) warn("TUNE", unit + ": AMD power limits are a percentage of default; use e.g. --gpu-plimit 70%");
  if (req.power_pct) {
    int off = static_cast<int>(std::lround(*req.power_pct - 100.0));
    if (in_range(kOd8PowerPercentage, off, "power limit")) want.push_back({kOd8PowerPercentage, off});
  }
  // RDNA3/4 Overdrive: the core setting is an offset to the max clock (Adrenalin "Max Frequency Offset"); memory is
  // an absolute max clock. There is no core clock lock or memory offset.
  if (req.core_offset_mhz && in_range(kOd8GfxclkFmax, *req.core_offset_mhz, "core clock offset"))
    want.push_back({kOd8GfxclkFmax, *req.core_offset_mhz});
  if (req.mem_lock_mhz && in_range(kOd8UclkFmax, *req.mem_lock_mhz, "max memory clock"))
    want.push_back({kOd8UclkFmax, *req.mem_lock_mhz});
  if (req.core_lock_mhz) warn("TUNE", unit + ": AMD Overdrive has no core clock lock; use --gpu-coffset (max clock offset)");
  if (req.mem_offset_mhz) warn("TUNE", unit + ": AMD Overdrive has no memory offset; use --gpu-mclock (max memory clock)");
  // Adrenalin's "Voltage Offset" (RDNA3/4, OD8_OD_VOLTAGE): mV below the stock voltage curve, e.g. -200 to 0.
  if (req.volt_offset_mv && in_range(kOd8OdVoltage, *req.volt_offset_mv, "core voltage offset"))
    want.push_back({kOd8OdVoltage, *req.volt_offset_mv});
  if (req.fan_pct) {
    // RDNA3/4: a 5-point fan curve; a flat curve at N% (temperatures kept) is a fixed speed, with zero RPM off.
    // Older Overdrive8 GPUs: the minimum fan speed.
    if (caps & kAdlOd8FanCurve) {
      bool good = true;
      for (int k = 0; k < kOd8FanCurvePoints; ++k)
        good = good && in_range(kOd8FanCurveSpeed1 + 2 * k, *req.fan_pct, "fan speed");
      if (good) {
        for (int k = 0; k < kOd8FanCurvePoints; ++k) want.push_back({kOd8FanCurveSpeed1 + 2 * k, *req.fan_pct});
        if (caps & kAdlOd8FanZeroRpmControl) want.push_back({kOd8FanZeroRpm, 0});
      }
    } else if (caps & kAdlOd8FanSpeedMin) {
      if (in_range(kOd8FanMinSpeed, *req.fan_pct, "fan speed")) want.push_back({kOd8FanMinSpeed, *req.fan_pct});
    } else {
      fail("fan speed", "this GPU has no Overdrive fan control");
    }
  }
  if (want.empty()) return false;
  std::vector<int> ids;
  for (auto& w : want) ids.push_back(w.first);
  if (caps & kAdlOd8OptimizedGpuPowerMode) {
    int prev = kOd8OptimizedPowerMode < static_cast<int>(cur.size()) ? cur[kOd8OptimizedPowerMode] : 0;
    if (prev != 3 && A.apply(adapter, {{kOd8OptimizedPowerMode, 3}}, {}, init) == 0)  // 3 = manual tuning
      impl_->undo.push_back([&A, adapter, prev, init] { A.apply(adapter, {{kOd8OptimizedPowerMode, prev}}, {}, init); });
  }
  int rc = A.apply(adapter, want, {}, init);
  if (rc != 0) {
    fail("Overdrive settings", "ADL error " + std::to_string(rc));
    return false;
  }
  std::vector<std::pair<int, int>> previous;
  std::string msg;
  for (auto [id, v] : want) {
    previous.push_back({id, id < static_cast<int>(cur.size()) ? cur[id] : init[id].defaultValue});
    if (id == kOd8PowerPercentage) msg += " power " + std::to_string(100 + v) + "%";
    else if (id == kOd8GfxclkFmax) msg += " core offset " + std::to_string(v) + " MHz";
    else if (id == kOd8UclkFmax) msg += " max memory " + std::to_string(v) + " MHz";
    else if (id == kOd8OdVoltage) msg += " voltage offset " + std::to_string(v) + " mV";
    else if (id == kOd8FanMinSpeed || id == kOd8FanCurveSpeed1) msg += " fan " + std::to_string(v) + "%";
  }
  ok("Overdrive" + msg);
  impl_->undo.push_back([&A, adapter, previous, init] { A.apply(adapter, previous, {}, init); });
  return applied;
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
  if (req.volt_offset_mv) {
    // RDNA3/4 Overdrive: "vo <mV>" is the voltage offset (OD_VDDGFX_OFFSET), then commit.
    const std::string od = dev + "/pp_od_clk_voltage";
    if (write_text(od, "vo " + std::to_string(*req.volt_offset_mv)) && write_text(od, "c")) {
      ok("core voltage offset " + std::to_string(*req.volt_offset_mv) + " mV");
      impl_->undo.push_back([od] { write_text(od, "vo 0"); write_text(od, "c"); });
    } else {
      fail("core voltage offset", "cannot write " + od + " (needs root, overdrive enabled and an RDNA3/4 GPU)");
    }
  }
  if (req.fan_pct) {
    // RDNA3/4: gpu_od/fan_ctrl/fan_curve ("<point> <temp> <speed%>", then "c"); older: hwmon pwm1 in manual mode.
    const std::string curve = dev + "/gpu_od/fan_ctrl/fan_curve";
    std::ifstream cf(curve);
    std::vector<int> temps;
    for (std::string line; std::getline(cf, line);) {
      int pt = 0, t = 0, sp = 0;
      if (std::sscanf(line.c_str(), "%d: %dC %d%%", &pt, &t, &sp) == 3) temps.push_back(t);
    }
    if (!temps.empty()) {
      bool good = true;
      for (size_t k = 0; k < temps.size(); ++k)
        good = good && write_text(curve, std::to_string(k) + " " + std::to_string(temps[k]) + " " + std::to_string(*req.fan_pct));
      good = good && write_text(curve, "c");
      if (good) {
        ok("fan curve flat at " + std::to_string(*req.fan_pct) + "%");
        impl_->undo.push_back([curve] { write_text(curve, "r"); write_text(curve, "c"); });
      } else {
        fail("fan speed", "cannot write " + curve + " (needs root and overdrive enabled)");
      }
    } else {
      const std::string en = hw + "/pwm1_enable", pwm = hw + "/pwm1", prev = read_line(en);
      if (!prev.empty() && write_text(en, "1") && write_text(pwm, std::to_string(*req.fan_pct * 255 / 100))) {
        ok("fan speed " + std::to_string(*req.fan_pct) + "%");
        impl_->undo.push_back([en, prev] { write_text(en, prev); });
      } else {
        fail("fan speed", "cannot write " + pwm + " (needs root)");
      }
    }
  }
  return applied;
#endif
}

}  // namespace xm::gpu
