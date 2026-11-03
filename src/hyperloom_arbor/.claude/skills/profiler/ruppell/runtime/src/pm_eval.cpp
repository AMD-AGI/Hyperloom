// gray-parrot's device samples on CUDA, evaluated after the run (docs/gray-parrot/README.md, GP4).
//
//   ruppell_pm_eval PROCESS_DIR [THREADS]
//
// The CUDA tool writes PM sampling's raw counter data (device_counters.raw: chunks of {char[8]
// "RPPMRAW1", u32 device, u32 samples, u64 image bytes, samples x (u64 start, u64 end), the image}),
// the chip, interval and metrics (device_counters_pm.tsv) and each device's counter availability
// image (device_counters_pm_avail.<device>.bin). Evaluating a sample costs about 216 us of host
// time (runner plan 04, probe A7), more than a core at 10 kHz, so the tool stores and this program
// evaluates, with CUPTI's Host API alone: no GPU, no CUDA context. It writes device_counters.bin
// (records.md: one record per sample and metric, step 0, group 0), replacing it, and prints the
// samples evaluated and failed.
#include "records.h"

#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
struct Chunk
{
    uint32_t              device = 0, samples = 0;
    std::vector<uint64_t> ts;
    std::vector<uint8_t>  image;
};

struct Device
{
    std::string              chip;
    std::vector<std::string> metrics;
    std::vector<uint16_t>    ids;
    std::vector<uint8_t>     avail;
};

std::string result_name(CUptiResult r)
{
    const char* s = nullptr;
    cuptiGetResultString(r, &s);
    return s ? s : std::to_string(static_cast<int>(r));
}

std::vector<uint8_t> slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// evaluate chunks [lo, hi) with a host object of its own; records into out
bool evaluate(const std::vector<Chunk>& cs, size_t lo, size_t hi, std::map<uint32_t, Device>& devs,
              std::vector<ruppell_device_counter_rec>& out, std::atomic<uint64_t>& bad, std::string& err)
{
    std::map<uint32_t, CUpti_Profiler_Host_Object*> hosts;
    for(size_t c = lo; c < hi; ++c)
    {
        const Chunk& ch = cs[c];
        Device&      d  = devs[ch.device];
        std::vector<const char*> names;
        for(auto& m : d.metrics) names.push_back(m.c_str());
        CUpti_Profiler_Host_Object*& h = hosts[ch.device];
        if(!h)
        {
            CUpti_Profiler_Host_Initialize_Params hi = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
            hi.profilerType              = CUPTI_PROFILER_TYPE_PM_SAMPLING;
            hi.pChipName                 = d.chip.c_str();
            hi.pCounterAvailabilityImage = d.avail.empty() ? nullptr : d.avail.data();
            CUptiResult r                = cuptiProfilerHostInitialize(&hi);
            if(r != CUPTI_SUCCESS)
            {
                err = "cuptiProfilerHostInitialize: " + result_name(r);
                return false;
            }
            h = hi.pHostObject;
            CUpti_Profiler_Host_ConfigAddMetrics_Params a = {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
            a.pHostObject   = h;
            a.ppMetricNames = names.data();
            a.numMetrics    = names.size();
            if((r = cuptiProfilerHostConfigAddMetrics(&a)) != CUPTI_SUCCESS)
            {
                err = "cuptiProfilerHostConfigAddMetrics: " + result_name(r);
                return false;
            }
        }
        std::vector<double> vals(names.size());
        for(uint32_t i = 0; i < ch.samples; ++i)
        {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params e = {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
            e.pHostObject          = h;
            e.pCounterDataImage    = const_cast<uint8_t*>(ch.image.data());
            e.counterDataImageSize = ch.image.size();
            e.ppMetricNames        = names.data();
            e.numMetrics           = names.size();
            e.rangeIndex           = i;
            e.pMetricValues        = vals.data();
            if(cuptiProfilerHostEvaluateToGpuValues(&e) != CUPTI_SUCCESS)
            {
                ++bad;
                continue;
            }
            for(size_t k = 0; k < vals.size(); ++k)
            {
                ruppell_device_counter_rec r{};
                r.t0      = ch.ts[2 * i];
                r.t1      = ch.ts[2 * i + 1];
                r.agent   = ch.device;
                r.step    = 0;
                r.group   = 0;
                r.counter = d.ids[k];
                r.value   = vals[k];
                out.push_back(r);
            }
        }
    }
    for(auto& [dev, h] : hosts)
    {
        CUpti_Profiler_Host_Deinitialize_Params dd = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        dd.pHostObject = h;
        cuptiProfilerHostDeinitialize(&dd);
    }
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(stderr, "usage: ruppell_pm_eval PROCESS_DIR [THREADS]\n");
        return 2;
    }
    const std::string dir = argv[1];
    unsigned          nt  = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : std::thread::hardware_concurrency();
    nt                    = std::max(1u, std::min(nt, 32u));
    std::map<uint32_t, Device> devs;
    {
        std::ifstream f(dir + "/device_counters_pm.tsv");
        if(!f)
        {
            std::fprintf(stderr, "ruppell_pm_eval: no %s/device_counters_pm.tsv\n", dir.c_str());
            return 1;
        }
        std::string line;
        while(std::getline(f, line))
        {
            std::vector<std::string> v;
            std::stringstream        ss(line);
            std::string              x;
            while(std::getline(ss, x, '\t')) v.push_back(x);
            if(v.size() >= 3 && v[0] == "device")
            {
                Device& d = devs[static_cast<uint32_t>(std::stoul(v[1]))];
                d.chip    = v[2];
                d.metrics.clear();
                d.ids.clear();
                d.avail = slurp(dir + "/device_counters_pm_avail." + v[1] + ".bin");
            }
            else if(v.size() >= 4 && v[0] == "metric")
            {
                Device& d = devs[static_cast<uint32_t>(std::stoul(v[1]))];
                d.metrics.push_back(v[2]);
                d.ids.push_back(static_cast<uint16_t>(std::stoul(v[3])));
            }
        }
    }
    std::vector<Chunk> cs;
    {
        FILE* f = std::fopen((dir + "/device_counters.raw").c_str(), "rb");
        if(!f)
        {
            std::fprintf(stderr, "ruppell_pm_eval: no %s/device_counters.raw\n", dir.c_str());
            return 1;
        }
        char magic[8];
        while(std::fread(magic, 1, 8, f) == 8)
        {
            Chunk    c;
            uint64_t bytes = 0;
            if(std::memcmp(magic, "RPPMRAW1", 8) != 0 || std::fread(&c.device, 4, 1, f) != 1 ||
               std::fread(&c.samples, 4, 1, f) != 1 || std::fread(&bytes, 8, 1, f) != 1)
                break;
            c.ts.resize(2ull * c.samples);
            c.image.resize(bytes);
            if(std::fread(c.ts.data(), 8, c.ts.size(), f) != c.ts.size() ||
               std::fread(c.image.data(), 1, bytes, f) != bytes)
                break;   // a chunk cut short by a killed process
            if(devs.count(c.device)) cs.push_back(std::move(c));
        }
        std::fclose(f);
    }
    std::vector<std::vector<ruppell_device_counter_rec>> outs(nt);
    std::vector<std::string>                             errs(nt);
    std::atomic<uint64_t>                                bad{0};
    std::vector<std::thread>                             ths;
    const size_t                                         per = (cs.size() + nt - 1) / nt;
    for(unsigned t = 0; t < nt; ++t)
    {
        const size_t lo = std::min(cs.size(), t * per), hi = std::min(cs.size(), lo + per);
        if(lo >= hi) break;
        ths.emplace_back([&, t, lo, hi] { evaluate(cs, lo, hi, devs, outs[t], bad, errs[t]); });
    }
    for(auto& th : ths) th.join();
    for(auto& e : errs)
        if(!e.empty())
        {
            std::fprintf(stderr, "ruppell_pm_eval: %s\n", e.c_str());
            return 1;
        }
    FILE* o = std::fopen((dir + "/device_counters.bin").c_str(), "wb");
    if(!o) return 1;
    uint64_t n = 0, samples = 0;
    for(auto& v : outs)
    {
        std::fwrite(v.data(), sizeof(ruppell_device_counter_rec), v.size(), o);
        n += v.size();
    }
    std::fclose(o);
    for(auto& c : cs) samples += c.samples;
    std::printf("{\"dir\": \"%s\", \"chunks\": %zu, \"samples\": %llu, \"failed\": %llu, \"records\": %llu, \"threads\": %u}\n",
                dir.c_str(), cs.size(), static_cast<unsigned long long>(samples), static_cast<unsigned long long>(bad.load()),
                static_cast<unsigned long long>(n), nt);
    return 0;
}
