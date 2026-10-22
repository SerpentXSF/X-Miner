#include "gpu/sensors.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

#include "gpu/dynlib.hpp"

#ifndef _WIN32
#include <dirent.h>
#endif

namespace xm::gpu {

namespace {

// ---------------------------------------------------------------- NVML (both OSes)
struct NvmlPci {  // nvmlPciInfo_t (v3)
  char busIdLegacy[16];
  unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
  char busId[32];
};

struct Nvml {
  DynLib lib;
  int (*init)() = nullptr;
  int (*count)(unsigned*) = nullptr;
  int (*by_index)(unsigned, void**) = nullptr;
  int (*pci)(void*, NvmlPci*) = nullptr;
  int (*power)(void*, unsigned*) = nullptr;           // milliwatts
  int (*temp)(void*, int, unsigned*) = nullptr;       // sensor 0 = GPU core
  int (*fan)(void*, unsigned*) = nullptr;             // percent
  int (*clock)(void*, int, unsigned*) = nullptr;      // clock type 1 = SM
  std::map<int, void*> by_bus;
  bool ready = false;

  Nvml() {
#ifdef _WIN32
    if (!lib.open("nvml.dll")) lib.open("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
    lib.open("libnvidia-ml.so.1");
#endif
    if (!lib.ok()) return;
    if (!lib.sym("nvmlInit_v2", init) || !lib.sym("nvmlDeviceGetCount_v2", count) ||
        !lib.sym("nvmlDeviceGetHandleByIndex_v2", by_index) || !lib.sym("nvmlDeviceGetPciInfo_v3", pci))
      return;
    lib.sym("nvmlDeviceGetPowerUsage", power);
    lib.sym("nvmlDeviceGetTemperature", temp);
    lib.sym("nvmlDeviceGetFanSpeed", fan);
    lib.sym("nvmlDeviceGetClockInfo", clock);
    if (init() != 0) return;
    unsigned n = 0;
    count(&n);
    for (unsigned i = 0; i < n; ++i) {
      void* h = nullptr;
      NvmlPci p{};
      if (by_index(i, &h) == 0 && pci(h, &p) == 0) by_bus[static_cast<int>(p.bus)] = h;
    }
    ready = true;
  }

  Reading read(int bus) {
    Reading r;
    auto it = by_bus.find(bus);
    if (!ready || it == by_bus.end()) return r;
    unsigned v = 0;
    if (power && power(it->second, &v) == 0) r.power_w = v / 1000.0;
    if (temp && temp(it->second, 0, &v) == 0) r.temp_c = v;
    if (fan && fan(it->second, &v) == 0) r.fan_pct = v;
    if (clock && clock(it->second, 1, &v) == 0) r.core_mhz = v;
    return r;
  }
};

#ifdef _WIN32
// ---------------------------------------------------------------- ADL PMLog (AMD, Windows)
// Struct layouts from AMD's ADL SDK headers (MIT licence).
struct AdlAdapterInfo {
  int iSize, iAdapterIndex;
  char strUDID[256];
  int iBusNumber, iDeviceNumber, iFunctionNumber, iVendorID;
  char strAdapterName[256], strDisplayName[256];
  int iPresent, iExist;
  char strDriverPath[256], strDriverPathExt[256], strPNPString[256];
  int iOSDisplayIndex;
};
struct AdlSensor {
  int supported, value;
};
struct AdlPmLog {
  int size;
  AdlSensor sensors[256];
};
enum { kGfxClk = 1, kTempEdge = 8, kFanPct = 15, kBoardPower = 73 };

void* __stdcall adl_alloc(int n) { return std::malloc(static_cast<size_t>(n)); }

struct Adl {
  DynLib lib;
  void* ctx = nullptr;
  int (*query)(void*, int, AdlPmLog*) = nullptr;
  std::map<int, int> adapter_by_bus;  // first adapter index per PCI bus that returns PMLog data

  Adl() {
    if (!lib.open("atiadlxx.dll")) return;
    int (*create)(void* (__stdcall*)(int), int, void**) = nullptr;
    int (*num)(void*, int*) = nullptr;
    int (*info)(void*, AdlAdapterInfo*, int) = nullptr;
    if (!lib.sym("ADL2_Main_Control_Create", create) || !lib.sym("ADL2_Adapter_NumberOfAdapters_Get", num) ||
        !lib.sym("ADL2_Adapter_AdapterInfo_Get", info) || !lib.sym("ADL2_New_QueryPMLogData_Get", query))
      return;
    if (create(adl_alloc, 1, &ctx) != 0) return;
    int n = 0;
    num(ctx, &n);
    if (n <= 0) return;
    std::vector<AdlAdapterInfo> ai(static_cast<size_t>(n));
    for (auto& a : ai) a.iSize = sizeof(AdlAdapterInfo);
    if (info(ctx, ai.data(), static_cast<int>(sizeof(AdlAdapterInfo) * ai.size())) != 0) return;
    for (const auto& a : ai) {
      if (adapter_by_bus.count(a.iBusNumber)) continue;
      AdlPmLog out{};
      out.size = sizeof out;
      if (query(ctx, a.iAdapterIndex, &out) == 0) adapter_by_bus[a.iBusNumber] = a.iAdapterIndex;
    }
  }

  Reading read(int bus) {
    Reading r;
    auto it = adapter_by_bus.find(bus);
    if (it == adapter_by_bus.end()) return r;
    AdlPmLog out{};
    out.size = sizeof out;
    if (query(ctx, it->second, &out) != 0) return r;
    auto get = [&](int i) { return out.sensors[i].supported ? static_cast<double>(out.sensors[i].value) : -1.0; };
    r.power_w = get(kBoardPower);
    r.temp_c = get(kTempEdge);
    r.fan_pct = get(kFanPct);
    r.core_mhz = get(kGfxClk);
    return r;
  }
};
#else
// ---------------------------------------------------------------- amdgpu sysfs (AMD, Linux)
double read_num(const std::string& path) {
  std::ifstream f(path);
  double v = -1;
  if (!(f >> v)) return -1;
  return v;
}

struct AmdSysfs {
  std::map<int, std::string> hwmon_by_bus;  // PCI bus -> /sys/class/drm/cardN/device/hwmon/hwmonM

  AmdSysfs() {
    for (int card = 0; card < 32; ++card) {
      std::string dev = "/sys/class/drm/card" + std::to_string(card) + "/device";
      std::ifstream ue(dev + "/uevent");
      std::string line, slot;
      bool amd = false;
      while (std::getline(ue, line)) {
        if (line.rfind("DRIVER=amdgpu", 0) == 0) amd = true;
        if (line.rfind("PCI_SLOT_NAME=", 0) == 0) slot = line.substr(14);  // 0000:03:00.0
      }
      if (!amd || slot.size() < 7) continue;
      int bus = static_cast<int>(std::strtol(slot.substr(5, 2).c_str(), nullptr, 16));
      if (DIR* d = opendir((dev + "/hwmon").c_str())) {
        while (dirent* e = readdir(d))
          if (std::strncmp(e->d_name, "hwmon", 5) == 0) hwmon_by_bus[bus] = dev + "/hwmon/" + e->d_name;
        closedir(d);
      }
    }
  }

  Reading read(int bus) {
    Reading r;
    auto it = hwmon_by_bus.find(bus);
    if (it == hwmon_by_bus.end()) return r;
    const std::string& h = it->second;
    double p = read_num(h + "/power1_average");
    if (p < 0) p = read_num(h + "/power1_input");
    if (p >= 0) r.power_w = p / 1e6;
    double t = read_num(h + "/temp1_input");
    if (t >= 0) r.temp_c = t / 1000.0;
    double pwm = read_num(h + "/pwm1");
    if (pwm >= 0) r.fan_pct = pwm * 100.0 / 255.0;
    double clk = read_num(h + "/freq1_input");
    if (clk >= 0) r.core_mhz = clk / 1e6;
    return r;
  }
};
#endif

}  // namespace

struct Sensors::Impl {
  std::vector<DeviceInfo> devs;
  std::unique_ptr<Nvml> nvml;
#ifdef _WIN32
  std::unique_ptr<Adl> amd;
#else
  std::unique_ptr<AmdSysfs> amd;
#endif
};

Sensors::Sensors(const std::vector<DeviceInfo>& devices) : impl_(std::make_unique<Impl>()) {
  impl_->devs = devices;
  bool any_nv = false, any_amd = false;
  for (const auto& d : devices) (d.vendor == Vendor::Nvidia ? any_nv : any_amd) = true;
  if (any_nv) impl_->nvml = std::make_unique<Nvml>();
#ifdef _WIN32
  if (any_amd) impl_->amd = std::make_unique<Adl>();
#else
  if (any_amd) impl_->amd = std::make_unique<AmdSysfs>();
#endif
}

Sensors::~Sensors() = default;

Reading Sensors::read(size_t index) {
  if (index >= impl_->devs.size()) return {};
  const auto& d = impl_->devs[index];
  if (d.vendor == Vendor::Nvidia) return impl_->nvml ? impl_->nvml->read(d.pci_bus) : Reading{};
  return impl_->amd ? impl_->amd->read(d.pci_bus) : Reading{};
}

}  // namespace xm::gpu
