// gray-parrot's collection on CUDA (docs/gray-parrot/README.md §1.4's note; the runner's
// ruppell-accuracy/plans/04_GRAY_PARROT_CUDA.md, Part B).
//
// Per step: one CUPTI Range Profiler user range (user-replay mode, a single-pass configuration)
// around each counter step. Per kernel is not used: in auto range a counted step's CUDA graph
// launches silently did not run (CUPTI 130001, driver 580, B200; runner plan 04, probe A3).
// A user range keeps graph replays, outputs and stream concurrency as they were, and leaves
// ruppell's kernel records complete (probes A2, A6b); the profiler launches one WaitNs kernel per
// range, which the engine names as the profiler's.
//
// The Range Profiler is set up at the first counter step, on the calling thread's context (it
// needs one; about 0.6 s once): the counter availability image, a Host API object and a
// configuration image per group (one pass, or the run's counters are refused), a counter data
// image of two ranges per group, cuptiRangeProfilerEnable. A counter step sets the group's
// configuration when it differs from the last, starts and pushes the range; its end pops, stops
// and decodes, checks that exactly one range came back (CUPTI reports neither drops nor a broken
// session: probes A5, A7b), evaluates each metric and writes one device_counters.bin record per
// metric (docs/gray-parrot/records.md: a step reading, `step` set).
//
// Switches (unset: nothing here runs):
//   RUPPELL_COUNTERS          sets ("memory,compute"), which `ruppell run` resolves into
//                             RUPPELL_COUNTER_GROUPS_SM_<CC> ("set/group=A,B;..."), or raw:A,B[;C,D]
//   RUPPELL_COUNTER_SCOPE     "step", the only scope on CUDA; "kernel" is refused
//   RUPPELL_COUNTER_GROUP_STEPS  consecutive counter steps a group is kept for (default 10): a switch
//                             (Disable, Enable, SetConfig) costs 15-55 ms of host time on top of the
//                             step's own start and stop, 60-120 ms (served models on B200, plan 04
//                             Part B)
//   RUPPELL_DEVICE_COUNTERS   refused on CUDA for now: PM sampling is plan 04's Part D, and it cannot
//                             share a process with the Range Profiler (probe A7b)
#include "gray_parrot_cuda.h"

#include "records.h"

#include <cuda.h>
#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_pmsampling.h>
#include <cupti_target.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace gray_parrot_cuda
{
namespace
{
// Leaked on purpose, as tool_cuda.cpp's globals are: the exit handler may run after this
// library's static destructors.
struct Group
{
    std::string                 name;   // "set/group", or raw/gN
    std::string                 set;
    std::vector<std::string>    metrics;
    std::vector<uint16_t>       ids;    // counters.tsv's numbers
    std::vector<uint8_t>        config, data;
    CUpti_Profiler_Host_Object* host   = nullptr;
    int                         passes = 0;
};

std::string&          dir     = *new std::string();
std::string&          spec    = *new std::string();   // RUPPELL_COUNTERS as given
std::string&          arch    = *new std::string();   // sm_<CC> the groups were resolved for
std::string&          refused = *new std::string();
std::string&          last_error = *new std::string();
std::string&          chip    = *new std::string();
std::vector<Group>&   gs      = *new std::vector<Group>();
std::map<std::string, uint16_t>& ids = *new std::map<std::string, uint16_t>();
std::mutex&           m       = *new std::mutex();
FILE*                 f_dev   = nullptr;
bool                  any_on  = false;
bool                  built   = false;
bool                  is_counting = false;
CUcontext             ctx     = nullptr;
int                   device  = -1;
CUpti_RangeProfiler_Object* obj = nullptr;
int                   cur_config = -1, cur_group = -1;
// rotation: a group is kept for RUPPELL_COUNTER_GROUP_STEPS consecutive counter steps (default 10),
// because switching groups costs host time (15-55 ms on served models, B200; plan 04 Part B)
int                   rot_group = 0, rot_count = 0, group_steps = 10;
uint32_t              cur_step = 0;
uint64_t              t0       = 0;
double                setup_ms = 0;
std::atomic<uint64_t> n_steps{0}, n_records{0}, n_bad{0}, n_start_fail{0}, n_stop_fail{0}, n_other_ctx{0};
std::atomic<uint64_t> start_ns{0}, stop_ns{0};

std::string jstr(const std::string& s)
{
    std::string o = "\"";
    for(char c : s)
    {
        if(c == '"' || c == '\\') o += '\\';
        if(c == '\n') { o += "\\n"; continue; }
        o += c;
    }
    return o + "\"";
}

std::string env(const char* k)
{
    const char* v = std::getenv(k);
    return v ? v : "";
}

std::vector<std::string> split(const std::string& s, char c)
{
    std::vector<std::string> out;
    std::stringstream        ss(s);
    std::string              x;
    while(std::getline(ss, x, c))
        if(!x.empty()) out.push_back(x);
    return out;
}

std::string result_name(CUptiResult r)
{
    const char* s = nullptr;
    cuptiGetResultString(r, &s);
    return s ? s : std::to_string(static_cast<int>(r));
}

void refuse(const std::string& why)
{
    if(refused.empty())
    {
        refused = why;
        std::fprintf(stderr, "[ruppell gray-parrot] counters refused: %s; tracing carries on\n", why.c_str());
    }
}

// "set/group=A,B;set/group=C" (ruppell run's resolved form) or raw "A,B;C,D"
bool parse_groups(const std::string& v, bool raw)
{
    int n = 0;
    for(const std::string& g : split(v, ';'))
    {
        Group gr;
        std::string list = g;
        if(raw)
        {
            gr.name = "raw/g" + std::to_string(n);
            gr.set  = "raw";
        }
        else
        {
            const size_t eq = g.find('=');
            if(eq == std::string::npos) return false;
            gr.name = g.substr(0, eq);
            gr.set  = gr.name.substr(0, gr.name.find('/'));
            list    = g.substr(eq + 1);
        }
        gr.metrics = split(list, ',');
        if(gr.metrics.empty()) return false;
        gs.push_back(gr);
        ++n;
    }
    return !gs.empty();
}

double now_ms()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

CUptiResult profiler_init();   // cuptiProfilerInitialize once, for the Range Profiler and PM sampling

// a metric's unit, from its name's rollup and submetric (CUPTI gives the base metric's hardware
// unit, not the value's)
std::string unit_of(const std::string& name)
{
    if(name.find("pct") != std::string::npos) return "percent";
    if(name.find("per_second") != std::string::npos) return "per second";
    if(name.find("ratio") != std::string::npos) return "ratio";
    if(name.find("bytes") != std::string::npos) return "bytes";
    if(name.find("cycles") != std::string::npos) return "cycles";
    if(name.find("time_duration") != std::string::npos) return "ns";
    return "count";
}

bool host_config(Group& g)
{
    CUpti_Profiler_Host_Initialize_Params hi = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hi.profilerType              = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hi.pChipName                 = chip.c_str();
    static std::vector<uint8_t>& avail = *new std::vector<uint8_t>();
    if(avail.empty())
    {
        CUpti_Profiler_GetCounterAvailability_Params av = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
        av.ctx = ctx;
        CUptiResult r = cuptiProfilerGetCounterAvailability(&av);
        if(r == CUPTI_SUCCESS)
        {
            avail.resize(av.counterAvailabilityImageSize);
            av.pCounterAvailabilityImage = avail.data();
            r = cuptiProfilerGetCounterAvailability(&av);
        }
        if(r != CUPTI_SUCCESS)
        {
            avail.clear();
            refuse("the counter availability query: " + result_name(r) +
                   (r == CUPTI_ERROR_HARDWARE_BUSY ? " (another counter client holds this GPU)" : ""));
            return false;
        }
    }
    hi.pCounterAvailabilityImage = avail.data();
    CUptiResult r = cuptiProfilerHostInitialize(&hi);
    if(r != CUPTI_SUCCESS)
    {
        refuse("cuptiProfilerHostInitialize: " + result_name(r));
        return false;
    }
    g.host = hi.pHostObject;
    std::vector<const char*> names;
    for(auto& x : g.metrics) names.push_back(x.c_str());
    CUpti_Profiler_Host_ConfigAddMetrics_Params a = {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    a.pHostObject   = g.host;
    a.ppMetricNames = names.data();
    a.numMetrics    = names.size();
    if((r = cuptiProfilerHostConfigAddMetrics(&a)) != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": a metric the GPU does not have (" + result_name(r) + ")");
        return false;
    }
    CUpti_Profiler_Host_GetConfigImageSize_Params s = {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    s.pHostObject = g.host;
    if((r = cuptiProfilerHostGetConfigImageSize(&s)) != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": config image size: " + result_name(r));
        return false;
    }
    g.config.resize(s.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params ci = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    ci.pHostObject     = g.host;
    ci.pConfigImage    = g.config.data();
    ci.configImageSize = g.config.size();
    if((r = cuptiProfilerHostGetConfigImage(&ci)) != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": config image: " + result_name(r));
        return false;
    }
    CUpti_Profiler_Host_GetNumOfPasses_Params np = {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    np.pConfigImage    = g.config.data();
    np.configImageSize = g.config.size();
    if((r = cuptiProfilerHostGetNumOfPasses(&np)) != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": passes: " + result_name(r));
        return false;
    }
    g.passes = static_cast<int>(np.numOfPasses);
    if(g.passes != 1)
    {
        refuse("group " + g.name + " takes " + std::to_string(g.passes) + " passes; a counter step counts one");
        return false;
    }
    return true;
}

bool data_image(Group& g)
{
    std::vector<const char*> names;
    for(auto& x : g.metrics) names.push_back(x.c_str());
    CUpti_RangeProfiler_GetCounterDataSize_Params sz = {CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    sz.pRangeProfilerObject = obj;
    sz.pMetricNames         = names.data();
    sz.numMetrics           = names.size();
    sz.maxNumOfRanges       = 2;
    sz.maxNumRangeTreeNodes = 2;
    CUptiResult r = cuptiRangeProfilerGetCounterDataSize(&sz);
    if(r != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": counter data size: " + result_name(r));
        return false;
    }
    g.data.assign(sz.counterDataSize, 0);
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params in = {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    in.pRangeProfilerObject = obj;
    in.pCounterData         = g.data.data();
    in.counterDataSize      = g.data.size();
    if((r = cuptiRangeProfilerCounterDataImageInitialize(&in)) != CUPTI_SUCCESS)
    {
        refuse("group " + g.name + ": counter data image: " + result_name(r));
        return false;
    }
    return true;
}

// counters.tsv (records.md): one line per group and metric; kind from the Host API
void write_tsv()
{
    FILE* f = std::fopen((dir + "/counters.tsv").c_str(), "w");
    if(!f) return;
    std::fprintf(f, "group\tcounter\tname\tset\tkind\tsource\treduce\tunit\n");
    for(size_t gi = 0; gi < gs.size(); ++gi)
    {
        const Group& g = gs[gi];
        for(size_t i = 0; i < g.metrics.size(); ++i)
        {
            const std::string base = g.metrics[i].substr(0, g.metrics[i].find('.'));
            std::string       kind = "counter", hw = "-";
            CUpti_Profiler_Host_GetMetricProperties_Params mp = {CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
            mp.pHostObject = g.host;
            mp.pMetricName = base.c_str();
            if(g.host && cuptiProfilerHostGetMetricProperties(&mp) == CUPTI_SUCCESS)
            {
                kind = mp.metricType == CUPTI_METRIC_TYPE_RATIO        ? "ratio"
                       : mp.metricType == CUPTI_METRIC_TYPE_THROUGHPUT ? "throughput"
                                                                        : "counter";
                if(mp.pHwUnit && *mp.pHwUnit) hw = mp.pHwUnit;
            }
            std::fprintf(f, "%zu\t%u\t%s\t%s\t%s\t%s\trange\t%s\n", gi, g.ids[i], g.metrics[i].c_str(), g.set.c_str(),
                         kind.c_str(), hw.c_str(), unit_of(g.metrics[i]).c_str());
        }
    }
    std::fclose(f);
}

// the Range Profiler on the calling thread's context, every group's images, the files
bool build()
{
    const double t = now_ms();
    built          = true;   // once, whatever happens
    if(cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
    {
        refuse("no current CUDA context at the first counter step");
        return false;
    }
    CUdevice dev = 0;
    int      maj = 0, min = 0;
    cuCtxGetDevice(&dev);
    cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    device               = static_cast<int>(dev);
    const std::string me = "sm_" + std::to_string(maj) + std::to_string(min);
    if(!arch.empty() && arch != me)
    {
        refuse("the groups were resolved for " + arch + ", and this GPU is " + me);
        return false;
    }
    arch = me;
    CUptiResult r = profiler_init();
    if(r != CUPTI_SUCCESS)
    {
        refuse("cuptiProfilerInitialize: " + result_name(r) +
               (r == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES ? " (counters need CAP_SYS_ADMIN, or NVreg_RestrictProfilingToAdminUsers=0)" : ""));
        return false;
    }
    CUpti_Device_GetChipName_Params cn = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    cn.deviceIndex = device;
    if((r = cuptiDeviceGetChipName(&cn)) != CUPTI_SUCCESS)
    {
        refuse("cuptiDeviceGetChipName: " + result_name(r));
        return false;
    }
    chip = cn.pChipName;
    for(Group& g : gs)
        if(!host_config(g)) return false;
    CUpti_RangeProfiler_Enable_Params en = {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    en.ctx = ctx;
    if((r = cuptiRangeProfilerEnable(&en)) != CUPTI_SUCCESS)
    {
        refuse("cuptiRangeProfilerEnable: " + result_name(r) +
               (r == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES ? " (counters need CAP_SYS_ADMIN, or NVreg_RestrictProfilingToAdminUsers=0)" : ""));
        return false;
    }
    obj = en.pRangeProfilerObject;
    // each group's counter data image is made when a step switches to the group (set_config): with
    // every group's made here, resetting the first group's after its step failed (CUPTI_ERROR_UNKNOWN,
    // B200, plan 04 Part B's bw0, 2026-10-22); the Range Profiler keeps the image initialized last
    write_tsv();
    f_dev = std::fopen((dir + "/device_counters.bin").c_str(), "ab");
    if(!f_dev)
    {
        refuse("device_counters.bin could not be opened");
        return false;
    }
    setup_ms = now_ms() - t;
    std::fprintf(stderr, "[ruppell gray-parrot] counting per step on %s (%s), %zu groups, set up in %.0f ms\n",
                 arch.c_str(), chip.c_str(), gs.size(), setup_ms);
    return true;
}

std::atomic<uint64_t> n_switches{0}, switch_ns{0};

// ---- device activity over the run: PM sampling (GP4, RUPPELL_DEVICE_COUNTERS=SET@RATE) -----------
// One PM sampling session per device, started on a thread of its own when the process creates its
// first context on the device. The thread decodes the hardware buffer every RUPPELL_DEVICE_DECODE_MS
// (default 50) into a counter data image of RUPPELL_DEVICE_IMAGE_SAMPLES samples (default 1024), and
// each decoded sample, with its start and end (CUPTI's activity clock, the kernel records' domain:
// probe A7):
// - RUPPELL_DEVICE_EVAL=tool (the default): is evaluated on that thread (the Host API, about 77 us a
//   sample with the activity set on B200) into device_counters.bin, one record per metric;
// - RUPPELL_DEVICE_EVAL=after: goes raw to device_counters.raw, about 30 KB a sample with the activity
//   set (chunks of {char[8] "RPPMRAW1", u32 device, u32 samples, u64 image bytes, samples x (u64
//   start, u64 end), the image}), evaluated after the run by `ruppell_pm_eval` (no GPU needed), with
//   device_counters_pm.tsv (chip, interval, metrics) and device_counters_pm_avail.<device>.bin.
struct PmDev
{
    int                      device = -1;
    CUpti_PmSampling_Object* obj    = nullptr;
    CUpti_Profiler_Host_Object* host = nullptr;
    std::vector<uint8_t>     data;
    std::thread              th;
};
Group&                 pm_group   = *new Group();
std::string&           pm_spec    = *new std::string();   // RUPPELL_DEVICE_COUNTERS as given
std::string&           pm_refused = *new std::string();
std::string&           pm_error   = *new std::string();
std::vector<PmDev*>&   pm_devs    = *new std::vector<PmDev*>();
std::mutex&            pm_m       = *new std::mutex();   // pm_devs, the files
FILE*                  f_raw      = nullptr;
FILE*                  f_pmdev    = nullptr;   // device_counters.bin, evaluated in the tool
bool                   pm_eval_after = false;
bool                   pm_on      = false;
uint64_t               pm_interval_ns = 100000;
int                    pm_decode_ms   = 50;
uint32_t               pm_max_samples = 1024;
std::atomic<bool>      pm_stop{false};
std::mutex&            pm_wake_m  = *new std::mutex();
std::condition_variable& pm_wake  = *new std::condition_variable();
std::atomic<uint64_t>  pm_samples{0}, pm_chunks{0}, pm_overflows{0}, pm_decode_ns{0}, pm_decodes{0}, pm_errors{0};
std::atomic<uint64_t>  pm_eval_ns{0}, pm_eval_fail{0}, pm_records{0};

void pm_refuse(const std::string& why)
{
    std::lock_guard<std::mutex> lk(pm_m);
    if(pm_refused.empty())
    {
        pm_refused = why;
        std::fprintf(stderr, "[ruppell gray-parrot] device counters refused: %s; tracing carries on\n", why.c_str());
    }
}

// "activity@10kHz", "activity@1000Hz": the sampling interval in ns, or 0
uint64_t parse_rate(const std::string& r)
{
    char*        end = nullptr;
    const double v   = std::strtod(r.c_str(), &end);
    std::string  u   = end ? end : "";
    for(char& c : u) c = static_cast<char>(std::tolower(c));
    const double hz = u == "hz" ? v : u == "khz" ? v * 1e3 : u == "mhz" ? v * 1e6 : 0;
    return hz > 0 ? static_cast<uint64_t>(1e9 / hz) : 0;
}

std::once_flag& profiler_once = *new std::once_flag();
CUptiResult     profiler_rc   = CUPTI_SUCCESS;
CUptiResult     profiler_init()
{
    std::call_once(profiler_once, [] {
        CUpti_Profiler_Initialize_Params pi = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        profiler_rc                         = cuptiProfilerInitialize(&pi);
    });
    return profiler_rc;
}

bool pm_write_chunk(PmDev& d, size_t n)
{
    std::vector<uint64_t> ts(2 * n);
    const auto            a = std::chrono::steady_clock::now();
    for(size_t i = 0; i < n; ++i)
    {
        CUpti_PmSampling_CounterData_GetSampleInfo_Params s = {CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        s.pPmSamplingObject    = d.obj;
        s.pCounterDataImage    = d.data.data();
        s.counterDataImageSize = d.data.size();
        s.sampleIndex          = i;
        if(cuptiPmSamplingCounterDataGetSampleInfo(&s) != CUPTI_SUCCESS) return false;
        ts[2 * i]     = s.startTimestamp;
        ts[2 * i + 1] = s.endTimestamp;
    }
    if(!pm_eval_after)
    {
        std::vector<const char*> names;
        for(auto& x : pm_group.metrics) names.push_back(x.c_str());
        std::vector<double>                     vals(names.size());
        std::vector<ruppell_device_counter_rec> recs;
        recs.reserve(n * names.size());
        for(size_t i = 0; i < n; ++i)
        {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params e = {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
            e.pHostObject          = d.host;
            e.pCounterDataImage    = d.data.data();
            e.counterDataImageSize = d.data.size();
            e.ppMetricNames        = names.data();
            e.numMetrics           = names.size();
            e.rangeIndex           = i;
            e.pMetricValues        = vals.data();
            if(cuptiProfilerHostEvaluateToGpuValues(&e) != CUPTI_SUCCESS)
            {
                ++pm_eval_fail;
                continue;
            }
            for(size_t k = 0; k < vals.size(); ++k)
            {
                ruppell_device_counter_rec r{};
                r.t0      = ts[2 * i];
                r.t1      = ts[2 * i + 1];
                r.agent   = static_cast<uint64_t>(d.device);
                r.step    = 0;
                r.group   = 0;
                r.counter = pm_group.ids[k];
                r.value   = vals[k];
                recs.push_back(r);
            }
        }
        std::lock_guard<std::mutex> lk(pm_m);
        if(!f_pmdev) return false;
        std::fwrite(recs.data(), sizeof(ruppell_device_counter_rec), recs.size(), f_pmdev);
        std::fflush(f_pmdev);
        pm_records += recs.size();
        pm_samples += n;
        ++pm_chunks;
        pm_eval_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count());
        return true;
    }
    std::lock_guard<std::mutex> lk(pm_m);
    if(!f_raw) return false;
    const char     magic[8] = {'R', 'P', 'P', 'M', 'R', 'A', 'W', '1'};
    const uint32_t dev = static_cast<uint32_t>(d.device), ns = static_cast<uint32_t>(n);
    const uint64_t bytes = d.data.size();
    std::fwrite(magic, 1, 8, f_raw);
    std::fwrite(&dev, 4, 1, f_raw);
    std::fwrite(&ns, 4, 1, f_raw);
    std::fwrite(&bytes, 8, 1, f_raw);
    std::fwrite(ts.data(), 8, ts.size(), f_raw);
    std::fwrite(d.data.data(), 1, d.data.size(), f_raw);
    std::fflush(f_raw);
    pm_samples += n;
    ++pm_chunks;
    return true;
}

bool pm_reset(PmDev& d)
{
    CUpti_PmSampling_CounterDataImage_Initialize_Params r = {CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    r.pPmSamplingObject = d.obj;
    r.pCounterData      = d.data.data();
    r.counterDataSize   = d.data.size();
    return cuptiPmSamplingCounterDataImageInitialize(&r) == CUPTI_SUCCESS;
}

// decode until the hardware buffer is empty; a full image is written and reset. `last`: write
// what the image holds
bool pm_drain(PmDev& d, bool last)
{
    const auto a = std::chrono::steady_clock::now();
    for(;;)
    {
        CUpti_PmSampling_DecodeData_Params dd = {CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
        dd.pPmSamplingObject    = d.obj;
        dd.pCounterDataImage    = d.data.data();
        dd.counterDataImageSize = d.data.size();
        CUptiResult r           = cuptiPmSamplingDecodeData(&dd);
        if(r != CUPTI_SUCCESS)
        {
            std::lock_guard<std::mutex> lk(pm_m);
            pm_error = "cuptiPmSamplingDecodeData: " + result_name(r);
            ++pm_errors;
            return false;
        }
        if(dd.overflow) ++pm_overflows;
        CUpti_PmSampling_GetCounterDataInfo_Params info = {CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        info.pCounterDataImage    = d.data.data();
        info.counterDataImageSize = d.data.size();
        if(cuptiPmSamplingGetCounterDataInfo(&info) != CUPTI_SUCCESS) return false;
        const bool full = dd.decodeStopReason == CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL;
        // written at every decode: evaluated in the tool, a sample's values are small; raw, the image
        // is sized to about one decode interval's samples
        if(info.numCompletedSamples)
        {
            if(!pm_write_chunk(d, info.numCompletedSamples) || !pm_reset(d)) return false;
        }
        ++pm_decodes;
        if(!full) break;
    }
    pm_decode_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count());
    return true;
}

// every device's session stopped and its thread joined, once: from an exit handler registered after
// the first session started, so it runs before the exit handlers CUPTI registered while setting up
// (exit handlers run in reverse order). Stopped from the tool's own, earlier exit handler, a decode
// thread ran into CUPTI's teardown and corrupted the heap (B200, CUPTI 130001, 2026-10-22).
void pm_stop_all()
{
    {
        std::lock_guard<std::mutex> lk(pm_wake_m);
        pm_stop = true;
    }
    pm_wake.notify_all();
    std::vector<PmDev*> ds;
    {
        std::lock_guard<std::mutex> lk(pm_m);
        ds = pm_devs;
    }
    for(PmDev* d : ds)
        if(d->th.joinable() && d->th.get_id() != std::this_thread::get_id()) d->th.join();
    std::lock_guard<std::mutex> lk(pm_m);
    if(f_raw) std::fflush(f_raw);
    if(f_pmdev) std::fflush(f_pmdev);
}

std::once_flag& pm_exit_once = *new std::once_flag();

// the device's session: set up, sample until fini, stop. On the device's own thread.
void pm_run(PmDev* d)
{
    auto fail = [&](const std::string& why, CUptiResult r) {
        pm_refuse("device " + std::to_string(d->device) + ": " + why + ": " + result_name(r) +
                  (r == CUPTI_ERROR_HARDWARE_BUSY ? " (another counter client holds this GPU)" : "") +
                  (r == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES ? " (counters need CAP_SYS_ADMIN, or NVreg_RestrictProfilingToAdminUsers=0)" : ""));
    };
    CUptiResult r = profiler_init();
    if(r != CUPTI_SUCCESS) return fail("cuptiProfilerInitialize", r);
    CUpti_Device_GetChipName_Params cn = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    cn.deviceIndex                     = d->device;
    if((r = cuptiDeviceGetChipName(&cn)) != CUPTI_SUCCESS) return fail("cuptiDeviceGetChipName", r);
    const std::string chip_d = cn.pChipName;
    std::vector<uint8_t> avail;
    CUpti_PmSampling_GetCounterAvailability_Params av = {CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    av.deviceIndex = d->device;
    if((r = cuptiPmSamplingGetCounterAvailability(&av)) != CUPTI_SUCCESS) return fail("PM counter availability", r);
    avail.resize(av.counterAvailabilityImageSize);
    av.pCounterAvailabilityImage = avail.data();
    if((r = cuptiPmSamplingGetCounterAvailability(&av)) != CUPTI_SUCCESS) return fail("PM counter availability", r);
    CUpti_Profiler_Host_Initialize_Params hi = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hi.profilerType              = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    hi.pChipName                 = chip_d.c_str();
    hi.pCounterAvailabilityImage = avail.data();
    if((r = cuptiProfilerHostInitialize(&hi)) != CUPTI_SUCCESS) return fail("cuptiProfilerHostInitialize (PM)", r);
    std::vector<const char*> names;
    for(auto& x : pm_group.metrics) names.push_back(x.c_str());
    CUpti_Profiler_Host_ConfigAddMetrics_Params ad = {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    ad.pHostObject   = hi.pHostObject;
    ad.ppMetricNames = names.data();
    ad.numMetrics    = names.size();
    if((r = cuptiProfilerHostConfigAddMetrics(&ad)) != CUPTI_SUCCESS) return fail("a metric PM sampling does not have", r);
    CUpti_Profiler_Host_GetConfigImageSize_Params cs = {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    cs.pHostObject = hi.pHostObject;
    if((r = cuptiProfilerHostGetConfigImageSize(&cs)) != CUPTI_SUCCESS) return fail("config image size", r);
    std::vector<uint8_t> config(cs.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params ci = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    ci.pHostObject     = hi.pHostObject;
    ci.pConfigImage    = config.data();
    ci.configImageSize = config.size();
    if((r = cuptiProfilerHostGetConfigImage(&ci)) != CUPTI_SUCCESS) return fail("config image", r);
    CUpti_Profiler_Host_GetNumOfPasses_Params np = {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    np.pConfigImage    = config.data();
    np.configImageSize = config.size();
    if((r = cuptiProfilerHostGetNumOfPasses(&np)) != CUPTI_SUCCESS) return fail("passes", r);
    if(np.numOfPasses != 1)
        return pm_refuse("group " + pm_group.name + " takes " + std::to_string(np.numOfPasses) +
                         " passes; PM sampling needs one");
    {
        // the evaluator's inputs: chip, interval, metrics; the availability image
        std::lock_guard<std::mutex> lk(pm_m);
        if(FILE* f = std::fopen((dir + "/device_counters_pm_avail." + std::to_string(d->device) + ".bin").c_str(), "wb"))
        {
            std::fwrite(avail.data(), 1, avail.size(), f);
            std::fclose(f);
        }
        if(FILE* f = std::fopen((dir + "/device_counters_pm.tsv").c_str(), "a"))
        {
            std::fprintf(f, "device\t%d\t%s\t%llu\t%u\n", d->device, chip_d.c_str(),
                         static_cast<unsigned long long>(pm_interval_ns), 0u);
            for(size_t i = 0; i < pm_group.metrics.size(); ++i)
                std::fprintf(f, "metric\t%d\t%s\t%u\n", d->device, pm_group.metrics[i].c_str(), pm_group.ids[i]);
            std::fclose(f);
        }
        if(FILE* f = std::fopen((dir + "/counters.tsv").c_str(), "r"))
            std::fclose(f);
        else if((f = std::fopen((dir + "/counters.tsv").c_str(), "w")))
        {
            std::fprintf(f, "group\tcounter\tname\tset\tkind\tsource\treduce\tunit\n");
            for(size_t i = 0; i < pm_group.metrics.size(); ++i)
            {
                const std::string base = pm_group.metrics[i].substr(0, pm_group.metrics[i].find('.'));
                std::string       kind = "counter", hw = "-";
                CUpti_Profiler_Host_GetMetricProperties_Params mp = {CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE};
                mp.pHostObject = hi.pHostObject;
                mp.pMetricName = base.c_str();
                if(cuptiProfilerHostGetMetricProperties(&mp) == CUPTI_SUCCESS)
                {
                    kind = mp.metricType == CUPTI_METRIC_TYPE_RATIO        ? "ratio"
                           : mp.metricType == CUPTI_METRIC_TYPE_THROUGHPUT ? "throughput"
                                                                            : "counter";
                    if(mp.pHwUnit && *mp.pHwUnit) hw = mp.pHwUnit;
                }
                std::fprintf(f, "0\t%u\t%s\t%s\t%s\t%s\tsample\t%s\n", pm_group.ids[i], pm_group.metrics[i].c_str(),
                             pm_group.set.c_str(), kind.c_str(), hw.c_str(), unit_of(pm_group.metrics[i]).c_str());
            }
            std::fclose(f);
        }
        if(chip.empty()) chip = chip_d;
    }
    // the Host object lives as long as the session: deinitialized before cuptiPmSamplingEnable, every
    // later DecodeData corrupted the heap (B200, CUPTI 130001, 2026-10-22; probe A7 kept it, and ran clean)
    struct HostGuard
    {
        CUpti_Profiler_Host_Object* h;
        ~HostGuard()
        {
            CUpti_Profiler_Host_Deinitialize_Params hd = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
            hd.pHostObject = h;
            cuptiProfilerHostDeinitialize(&hd);
        }
    } host_guard{hi.pHostObject};
    d->host = hi.pHostObject;
    CUpti_PmSampling_Enable_Params en = {CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    en.deviceIndex = d->device;
    if((r = cuptiPmSamplingEnable(&en)) != CUPTI_SUCCESS) return fail("cuptiPmSamplingEnable", r);
    d->obj = en.pPmSamplingObject;
    CUpti_PmSampling_SetConfig_Params sc = {CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    sc.pPmSamplingObject  = d->obj;
    sc.configSize         = config.size();
    sc.pConfig            = config.data();
    sc.hardwareBufferSize = 64ull << 20;   // probe A7's
    sc.samplingInterval   = pm_interval_ns;
    sc.triggerMode        = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    sc.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    if((r = cuptiPmSamplingSetConfig(&sc)) != CUPTI_SUCCESS) return fail("cuptiPmSamplingSetConfig", r);
    CUpti_PmSampling_GetCounterDataSize_Params sz = {CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    sz.pPmSamplingObject = d->obj;
    sz.pMetricNames      = names.data();
    sz.numMetrics        = names.size();
    sz.maxSamples        = pm_max_samples;
    if((r = cuptiPmSamplingGetCounterDataSize(&sz)) != CUPTI_SUCCESS) return fail("counter data size", r);
    d->data.assign(sz.counterDataSize, 0);
    if(!pm_reset(*d)) return pm_refuse("device " + std::to_string(d->device) + ": the counter data image");
    CUpti_PmSampling_Start_Params st = {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    st.pPmSamplingObject = d->obj;
    if((r = cuptiPmSamplingStart(&st)) != CUPTI_SUCCESS) return fail("cuptiPmSamplingStart", r);
    std::call_once(pm_exit_once, [] { std::atexit(pm_stop_all); });
    std::fprintf(stderr, "[ruppell gray-parrot] sampling device %d (%s) every %llu ns, %zu metrics\n", d->device,
                 chip_d.c_str(), static_cast<unsigned long long>(pm_interval_ns), pm_group.metrics.size());
    bool ok = true;
    while(!pm_stop && ok)
    {
        {
            std::unique_lock<std::mutex> lk(pm_wake_m);
            pm_wake.wait_for(lk, std::chrono::milliseconds(pm_decode_ms), [] { return pm_stop.load(); });
        }
        if(!pm_stop) ok = pm_drain(*d, false);
    }
    CUpti_PmSampling_Stop_Params sp = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
    sp.pPmSamplingObject = d->obj;
    if((r = cuptiPmSamplingStop(&sp)) == CUPTI_SUCCESS && ok) pm_drain(*d, true);
    CUpti_PmSampling_Disable_Params dp = {CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    dp.pPmSamplingObject = d->obj;
    cuptiPmSamplingDisable(&dp);
}

bool set_config(int g)
{
    // A configuration set after another kept collecting the first one's counters: a second group
    // read NaN except for the metrics the first group also had (B200, CUPTI 130001, plan 04 Part
    // B's bnan3, 2026-10-22). So switching groups disables the Range Profiler and enables it again.
    if(cur_config >= 0)
    {
        const auto a = std::chrono::steady_clock::now();
        CUpti_RangeProfiler_Disable_Params dp = {CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        dp.pRangeProfilerObject = obj;
        CUptiResult r = cuptiRangeProfilerDisable(&dp);
        CUpti_RangeProfiler_Enable_Params en = {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
        en.ctx = ctx;
        if(r == CUPTI_SUCCESS) r = cuptiRangeProfilerEnable(&en);
        if(r != CUPTI_SUCCESS)
        {
            last_error = "switching groups (Disable, Enable): " + result_name(r);
            refuse("the Range Profiler could not switch groups: " + result_name(r));
            return false;
        }
        obj        = en.pRangeProfilerObject;
        cur_config = -1;
        ++n_switches;
        switch_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count());
    }
    if(!data_image(gs[g]))
    {
        last_error = "the counter data image of group " + gs[g].name;
        return false;
    }
    CUpti_RangeProfiler_SetConfig_Params p = {CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    p.pRangeProfilerObject = obj;
    p.configSize           = gs[g].config.size();
    p.pConfig              = gs[g].config.data();
    p.counterDataImageSize = gs[g].data.size();
    p.pCounterDataImage    = gs[g].data.data();
    p.range                = CUPTI_UserRange;
    p.replayMode           = CUPTI_UserReplay;
    p.maxRangesPerPass     = 2;
    p.numNestingLevels     = 1;
    p.minNestingLevel      = 1;
    p.passIndex            = 0;
    p.targetNestingLevel   = 0;
    CUptiResult r = cuptiRangeProfilerSetConfig(&p);
    if(r != CUPTI_SUCCESS)
    {
        last_error = "SetConfig: " + result_name(r);
        return false;
    }
    cur_config = g;
    return true;
}
}  // namespace

// RUPPELL_DEVICE_COUNTERS=SET@RATE: one group, resolved by `ruppell run` into
// RUPPELL_DEVICE_COUNTER_GROUPS_SM_<CC> ("set/group=A,B"), or raw:A,B@RATE
void configure_device(const std::string& dev)
{
    pm_spec = dev;
    pm_on   = true;
    if(!spec.empty())
        return pm_refuse("per-step counters (RUPPELL_COUNTERS) are on: PM sampling cannot share a process with the "
                         "Range Profiler (runner plan 04, probe A7b)");
    const size_t at = dev.rfind('@');
    if(at == std::string::npos || !(pm_interval_ns = parse_rate(dev.substr(at + 1))))
        return pm_refuse("RUPPELL_DEVICE_COUNTERS=" + dev + ": want SET@RATE (activity@10kHz)");
    if(const int ms = std::atoi(env("RUPPELL_DEVICE_DECODE_MS").c_str()); ms > 0) pm_decode_ms = ms;
    if(const int n = std::atoi(env("RUPPELL_DEVICE_IMAGE_SAMPLES").c_str()); n > 0) pm_max_samples = static_cast<uint32_t>(n);
    pm_eval_after = env("RUPPELL_DEVICE_EVAL") == "after";
    const std::string what = dev.substr(0, at);
    std::string       list;
    if(what.rfind("raw:", 0) == 0)
    {
        pm_group.name = "raw/device";
        pm_group.set  = "raw";
        list          = what.substr(4);
    }
    else
    {
        std::string v;
        for(char** e = environ; e && *e; ++e)
            if(std::strncmp(*e, "RUPPELL_DEVICE_COUNTER_GROUPS_SM_", 33) == 0)
                if(const char* eq = std::strchr(*e, '=')) v = eq + 1;
        const size_t eq = v.find('=');
        if(eq == std::string::npos)
            return pm_refuse("RUPPELL_DEVICE_COUNTERS=" + dev + ": no RUPPELL_DEVICE_COUNTER_GROUPS_SM_* (run under "
                             "`ruppell run`, which resolves the set; or give raw:A,B@RATE)");
        pm_group.name = v.substr(0, eq);
        pm_group.set  = pm_group.name.substr(0, pm_group.name.find('/'));
        list          = v.substr(eq + 1);
    }
    pm_group.metrics = split(list, ',');
    for(uint16_t i = 0; i < pm_group.metrics.size(); ++i) pm_group.ids.push_back(i);
    if(pm_group.metrics.empty()) return pm_refuse("RUPPELL_DEVICE_COUNTERS=" + dev + ": no metric");
    if(pm_eval_after)
    {
        f_raw = std::fopen((dir + "/device_counters.raw").c_str(), "ab");
        if(!f_raw) pm_refuse("device_counters.raw could not be opened");
    }
    else if(!(f_pmdev = std::fopen((dir + "/device_counters.bin").c_str(), "ab")))
        pm_refuse("device_counters.bin could not be opened");
}

void configure(const std::string& out_dir)
{
    dir                   = out_dir;
    spec                  = env("RUPPELL_COUNTERS");
    const std::string dev = env("RUPPELL_DEVICE_COUNTERS");
    any_on                = !spec.empty() || !dev.empty();
    if(!any_on) return;
    if(!dev.empty()) configure_device(dev);
    if(spec.empty()) return;
    if(const int gsn = std::atoi(env("RUPPELL_COUNTER_GROUP_STEPS").c_str()); gsn > 0) group_steps = gsn;
    const std::string scope = env("RUPPELL_COUNTER_SCOPE");
    if(!scope.empty() && scope != "step")
    {
        refuse("RUPPELL_COUNTER_SCOPE=" + scope + ": CUDA counts per step; per kernel, CUPTI's auto range skipped "
               "CUDA graph launches (runner plan 04, probe A3)");
        return;
    }
    if(spec.rfind("raw:", 0) == 0)
    {
        if(!parse_groups(spec.substr(4), true)) refuse("RUPPELL_COUNTERS=" + spec + ": no group");
    }
    else
    {
        // the one sm_ architecture `ruppell run` resolved the sets for (sets/sm_<CC>.yaml)
        std::string var;
        for(char** e = environ; e && *e; ++e)
            if(std::strncmp(*e, "RUPPELL_COUNTER_GROUPS_SM_", 26) == 0)
            {
                const char* eq = std::strchr(*e, '=');
                if(!eq) continue;
                if(!var.empty())
                {
                    refuse("more than one RUPPELL_COUNTER_GROUPS_SM_* set");
                    return;
                }
                var  = std::string(*e, eq - *e);
                arch = "sm_" + var.substr(26);
                for(char& c : arch) c = static_cast<char>(std::tolower(c));
            }
        if(var.empty())
        {
            refuse("RUPPELL_COUNTERS=" + spec + ": no RUPPELL_COUNTER_GROUPS_SM_* (run under `ruppell run`, which "
                   "resolves the sets; or give raw:A,B;C,D)");
            return;
        }
        if(!parse_groups(env(var.c_str()), false)) refuse(var + ": unreadable");
    }
    uint16_t next = 0;
    for(Group& g : gs)
        for(const std::string& x : g.metrics)
        {
            auto it = ids.find(x);
            if(it == ids.end()) it = ids.emplace(x, next++).first;
            g.ids.push_back(it->second);
        }
}

int groups()
{
    if(!any_on || spec.empty() || !refused.empty()) return -1;
    return static_cast<int>(gs.size());
}

bool counting() { return is_counting; }

int start(int group, uint32_t step, uint64_t now)
{
    if(!any_on || !refused.empty()) return -1;
    std::lock_guard<std::mutex> lk(m);
    if(!built && !build()) return -1;
    if(!refused.empty()) return -1;
    const int ng = static_cast<int>(gs.size());
    if(group >= ng) return -3;
    if(is_counting) return cur_group;
    CUcontext c = nullptr;
    cuCtxGetCurrent(&c);
    if(c != ctx)
    {
        ++n_other_ctx;
        last_error = "a counter step on another context than the first";
        return -2;
    }
    if(group < 0 && rot_count >= group_steps)
    {
        rot_group = (rot_group + 1) % ng;
        rot_count = 0;
    }
    const int  g = group >= 0 ? group : rot_group;
    if(group < 0) ++rot_count;
    const auto a = std::chrono::steady_clock::now();
    if(g != cur_config && !set_config(g))
    {
        ++n_start_fail;
        return -2;
    }
    CUpti_RangeProfiler_Start_Params st = {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    st.pRangeProfilerObject             = obj;
    CUptiResult r                       = cuptiRangeProfilerStart(&st);
    if(r == CUPTI_SUCCESS)
    {
        CUpti_RangeProfiler_PushRange_Params pu = {CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
        pu.pRangeProfilerObject                 = obj;
        pu.pRangeName                           = "step";
        r                                       = cuptiRangeProfilerPushRange(&pu);
    }
    if(r != CUPTI_SUCCESS)
    {
        ++n_start_fail;
        last_error = "Start/PushRange: " + result_name(r);
        refuse("the Range Profiler did not start a counter step: " + result_name(r));
        return -2;
    }
    start_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count());
    is_counting = true;
    cur_group   = g;
    cur_step    = step;
    t0          = now;
    ++n_steps;
    return g;
}

int stop(uint64_t now)
{
    std::lock_guard<std::mutex> lk(m);
    if(!is_counting) return 0;
    is_counting  = false;
    const auto a = std::chrono::steady_clock::now();
    Group&     g = gs[cur_group];
    CUpti_RangeProfiler_PopRange_Params po = {CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
    po.pRangeProfilerObject                = obj;
    CUptiResult r                          = cuptiRangeProfilerPopRange(&po);
    CUpti_RangeProfiler_Stop_Params sp     = {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    sp.pRangeProfilerObject                = obj;
    if(r == CUPTI_SUCCESS) r = cuptiRangeProfilerStop(&sp);
    CUpti_RangeProfiler_DecodeData_Params dd = {CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    dd.pRangeProfilerObject                  = obj;
    if(r == CUPTI_SUCCESS) r = cuptiRangeProfilerDecodeData(&dd);
    if(r != CUPTI_SUCCESS)
    {
        ++n_stop_fail;
        last_error = "PopRange/Stop/DecodeData: " + result_name(r);
        refuse("the Range Profiler did not end a counter step: " + result_name(r));
        return -2;
    }
    CUpti_RangeProfiler_GetCounterDataInfo_Params info = {CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    info.pCounterDataImage                             = g.data.data();
    info.counterDataImageSize                          = g.data.size();
    r                                                  = cuptiRangeProfilerGetCounterDataInfo(&info);
    if(r != CUPTI_SUCCESS || info.numTotalRanges != 1 || !sp.isAllPassSubmitted)
    {
        // a broken session or a dropped range reports nothing itself (probes A5, A7b)
        ++n_bad;
        last_error = "a counter step came back with " + std::to_string(info.numTotalRanges) + " ranges";
    }
    else
    {
        std::vector<const char*> names;
        for(auto& x : g.metrics) names.push_back(x.c_str());
        std::vector<double>                            vals(names.size());
        CUpti_Profiler_Host_EvaluateToGpuValues_Params e = {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        e.pHostObject          = g.host;
        e.pCounterDataImage    = g.data.data();
        e.counterDataImageSize = g.data.size();
        e.ppMetricNames        = names.data();
        e.numMetrics           = names.size();
        e.rangeIndex           = 0;
        e.pMetricValues        = vals.data();
        if((r = cuptiProfilerHostEvaluateToGpuValues(&e)) != CUPTI_SUCCESS)
        {
            ++n_bad;
            last_error = "EvaluateToGpuValues: " + result_name(r);
        }
        else
            for(size_t i = 0; i < vals.size(); ++i)
            {
                ruppell_device_counter_rec rec{};
                rec.t0      = t0;
                rec.t1      = now;
                rec.agent   = static_cast<uint64_t>(device);
                rec.step    = cur_step;
                rec.group   = static_cast<uint16_t>(cur_group);
                rec.counter = g.ids[i];
                rec.value   = vals[i];
                std::fwrite(&rec, sizeof(rec), 1, f_dev);
                ++n_records;
            }
    }
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params in = {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    in.pRangeProfilerObject = obj;
    in.pCounterData         = g.data.data();
    in.counterDataSize      = g.data.size();
    if((r = cuptiRangeProfilerCounterDataImageInitialize(&in)) != CUPTI_SUCCESS)
    {
        ++n_stop_fail;
        refuse("the counter data image could not be reset: " + result_name(r));
        return -2;
    }
    stop_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count());
    return 0;
}

void context_created(int device)
{
    if(!pm_on || device < 0) return;
    std::lock_guard<std::mutex> lk(pm_m);
    if(!pm_refused.empty() || pm_stop) return;
    for(PmDev* d : pm_devs)
        if(d->device == device) return;
    PmDev* d  = new PmDev();
    d->device = device;
    pm_devs.push_back(d);
    d->th = std::thread(pm_run, d);
}

void flush()
{
    std::lock_guard<std::mutex> lk(m);
    if(f_dev) std::fflush(f_dev);
}

void fini()
{
    if(pm_on)
    {
        pm_stop_all();
        std::lock_guard<std::mutex> lk(pm_m);
        for(FILE** f : {&f_raw, &f_pmdev})
            if(*f)
            {
                std::fclose(*f);
                *f = nullptr;
            }
    }
    std::lock_guard<std::mutex> lk(m);
    if(f_dev)
    {
        std::fclose(f_dev);
        f_dev = nullptr;
    }
}

std::string device_json()
{
    if(!pm_on) return "null";
    std::string devs = "[";
    {
        std::lock_guard<std::mutex> lk(pm_m);
        for(size_t i = 0; i < pm_devs.size(); ++i) devs += std::string(i ? ", " : "") + std::to_string(pm_devs[i]->device);
    }
    devs += "]";
    std::string ms = "[";
    for(size_t k = 0; k < pm_group.metrics.size(); ++k) ms += std::string(k ? ", " : "") + jstr(pm_group.metrics[k]);
    ms += "]";
    return "{\"collector\": \"cupti PM sampling\", \"spec\": " + jstr(pm_spec) + ", \"group\": 0, \"group_name\": " + jstr(pm_group.name) +
           ", \"counters\": " + ms + ", \"interval_ns\": " + std::to_string(pm_interval_ns) + ", \"devices\": " + devs +
           ", \"samples\": " + std::to_string(pm_samples.load()) + ", \"chunks\": " + std::to_string(pm_chunks.load()) +
           ", \"overflows\": " + std::to_string(pm_overflows.load()) + ", \"decodes\": " + std::to_string(pm_decodes.load()) +
           ", \"decode_ms_total\": " + std::to_string(pm_decode_ns.load() / 1e6) + ", \"decode_errors\": " +
           std::to_string(pm_errors.load()) + ", \"evaluated\": " + jstr(pm_eval_after ? "after the run: ruppell_pm_eval" : "in the tool") +
           ", \"records\": " + std::to_string(pm_records.load()) + ", \"eval_ms_total\": " + std::to_string(pm_eval_ns.load() / 1e6) +
           ", \"eval_failures\": " + std::to_string(pm_eval_fail.load()) + ", \"refused\": " +
           (pm_refused.empty() ? "null" : jstr(pm_refused)) + ", \"last_error\": " + (pm_error.empty() ? "null" : jstr(pm_error)) + "}";
}

std::string meta_json()
{
    if(!any_on) return "\"off\"";
    auto        u     = [](const std::atomic<uint64_t>& v) { return std::to_string(v.load()); };
    const auto  steps = env("RUPPELL_COUNTER_STEPS");
    uint32_t    api   = 0;
    cuptiGetVersion(&api);
    std::string groups_j = "[";
    for(size_t i = 0; i < gs.size(); ++i)
    {
        groups_j += std::string(i ? ", " : "") + "{\"group\": " + std::to_string(i) + ", \"name\": " + jstr(gs[i].name) +
                    ", \"counters\": [";
        for(size_t k = 0; k < gs[i].metrics.size(); ++k) groups_j += std::string(k ? ", " : "") + jstr(gs[i].metrics[k]);
        groups_j += "], \"passes\": " + std::to_string(gs[i].passes) + "}";
    }
    groups_j += "]";
    std::string sets_j = "[";
    {
        std::vector<std::string> seen;
        for(const Group& g : gs)
            if(std::find(seen.begin(), seen.end(), g.set) == seen.end()) seen.push_back(g.set);
        for(size_t i = 0; i < seen.size(); ++i) sets_j += std::string(i ? ", " : "") + jstr(seen[i]);
    }
    sets_j += "]";
    const uint64_t ns = n_steps.load();
    return "{\n    \"version\": 1, \"platform\": \"cuda\", \"arch\": " + jstr(arch) + ", \"collector\": " +
           jstr("cupti " + std::to_string(api) + ", Range Profiler, user range") + ",\n    \"scope\": " + std::string(spec.empty() ? "\"device\"" : "\"step\"") + ", \"sets\": " +
           sets_j + ", \"steps\": " + (steps.empty() ? "\"1/1000\"" : jstr(steps)) + ", \"counter_steps\": " + u(n_steps) +
           ",\n    \"groups\": " + groups_j + ",\n    \"records\": " + u(n_records) +
           ", \"dropped\": {\"steps\": " + u(n_bad) +
           ", \"reason\": \"a counter step whose range count was not 1, or whose values did not evaluate\"}, \"refused\": " +
           (refused.empty() ? "null" : jstr(refused)) + ", \"device\": " + device_json() +
           ",\n    \"notes\": [" +
           jstr("per step: one Range Profiler user range per counter step; per kernel is not used on CUDA (auto range "
                "skipped CUDA graph launches, runner plan 04 probe A3)") +
           ", " + jstr("the profiler launches one WaitNs kernel per counter step, outside every step") + "]" +
           ",\n    \"cuda\": {\"chip\": " + jstr(chip) + ", \"device\": " + std::to_string(device) +
           ", \"setup_ms\": " + std::to_string(static_cast<long long>(setup_ms)) +
           ", \"start_failures\": " + u(n_start_fail) + ", \"stop_failures\": " + u(n_stop_fail) +
           ", \"other_context_steps\": " + u(n_other_ctx) + ", \"group_switches\": " + u(n_switches) +
           ", \"switch_ms_mean\": " + std::to_string(n_switches.load() ? switch_ns.load() / 1e6 / n_switches.load() : 0.0) +
           ", \"start_ms_mean\": " + std::to_string(ns ? start_ns.load() / 1e6 / ns : 0.0) +
           ", \"stop_ms_mean\": " + std::to_string(ns ? stop_ns.load() / 1e6 / ns : 0.0) +
           ", \"last_error\": " + (last_error.empty() ? "null" : jstr(last_error)) + "}\n  }";
}
}  // namespace gray_parrot_cuda
