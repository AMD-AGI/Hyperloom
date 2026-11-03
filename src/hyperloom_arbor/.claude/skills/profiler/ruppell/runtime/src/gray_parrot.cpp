// gray-parrot's collection on ROCm (docs/gray-parrot/README.md §1.4; the runner's
// ruppell-accuracy/plans/03_GRAY_PARROT_ROCM.md, B1).
//
// Per kernel: the profiler SDK's dispatch counting service, in a context of its own (ctx_ctr),
// configured at init and started only for counter steps (ruppell_counters). While it is started
// the SDK serializes every dispatch on the GPU, counted or not. Its callback runs on the
// launching thread before each dispatch is queued and returns the current group's configuration
// for the dispatch's agent, or none for a kernel RUPPELL_COUNTER_KERNELS leaves out (decided
// once per kernel, when its symbol registers). It never allocates, locks, or calls into Python.
// The values arrive in a buffer on the tool's callback thread, are summed over the counter's
// dimension instances, and go to counters.bin, joined to dispatch.bin by dispatch id.
//
// Over the run: the device counting service, in a context of its own (ctx_dev), read by a thread
// at RUPPELL_DEVICE_COUNTERS's rate, to device_counters.bin.
//
// Switches (unset: nothing here is created):
//   RUPPELL_COUNTERS          sets ("memory,compute"), which `ruppell run` resolves per
//                             architecture into RUPPELL_COUNTER_GROUPS_<ARCH> ("set/group=A,B;..."),
//                             or raw:A,B[;C,D]: groups in the vendor's names, ';' between groups
//   RUPPELL_COUNTER_KERNELS   a regex (ECMAScript) searched in kernel symbol names, as the SDK gives
//                             them (mangled): count those kernels only
//   RUPPELL_COUNTER_DIMS      "sum" (default) or "xcc": keep each XCC's value
//   RUPPELL_COUNTER_SCOPE     "kernel" (default); "step" is not built yet, and refused
//   RUPPELL_DEVICE_COUNTERS   SPEC@RATE: one group (as RUPPELL_COUNTERS; resolved into
//                             RUPPELL_DEVICE_COUNTER_GROUPS_<ARCH>) read at RATE (100Hz, 1kHz, 10ms)
//   RUPPELL_DEVICE_AGENTS     GPU indices (logical_node_type_id) the device service counts on;
//                             default the GPUs visible to HIP

#include "gray_parrot.h"

#include "records.h"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <stdio_ext.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" int ruppell_detail(int on);

namespace gray_parrot
{
namespace
{
// Leaked on purpose, as tool.cpp's globals are: the SDK can call tool_fini after this library's
// static destructors have run.

struct Group
{
    std::string              set, name;  // "memory", "m1"; raw groups are "raw", "r1", ...
    std::vector<std::string> counters;
};

struct Agent
{
    uint64_t    handle = 0;
    std::string arch;          // "gfx950"
    int32_t     index  = -1;   // logical_node_type_id: the GPU's index
    bool        hip_visible = false;
};

// what a counter id read back from a record stands for, per agent and group: its number in
// counters.tsv (docs/gray-parrot/records.md), or with RUPPELL_COUNTER_DIMS=xcc one per XCC
struct Target
{
    uint16_t              number  = 0;
    bool                  has_xcc = false;
    uint64_t              xcc_dim = 0;  // the XCC dimension's id, for rocprofiler_query_record_dimension_position
    std::vector<uint16_t> by_xcc;       // RUPPELL_COUNTER_DIMS=xcc: each XCC's number (NAME[xcc=i])

    uint16_t of(size_t xcc) const { return xcc < by_xcc.size() ? by_xcc[xcc] : number; }
};

struct Service  // a counting service's configuration: its groups, agents and configs
{
    std::vector<Group>                                  groups;
    std::vector<Agent>                                  agents;
    std::vector<rocprofiler_counter_config_id_t>        cfg;      // [group * agents + agent]
    std::vector<std::unordered_map<uint64_t, Target>>   targets;  // [group * agents + agent]
    std::vector<std::string>                            status;   // per group: "ok" or why not
    std::string                                         refused;  // empty when configured
    uint16_t                                            group_base = 0;  // the records' number of its first group
};

bool        any_on   = false;  // RUPPELL_COUNTERS or RUPPELL_DEVICE_COUNTERS set
std::string& out_dir = *new std::string();
std::string& meta_spec        = *new std::string();
std::string& meta_dev_spec    = *new std::string();
std::string& kernel_filter    = *new std::string();
Service&     ctr              = *new Service();
Service&     dev              = *new Service();
bool         per_xcc          = false;
std::mutex&  m_tsv            = *new std::mutex();
std::vector<std::string>& tsv_rows = *new std::vector<std::string>();

// Configurations are built at tool init when the SDK can, otherwise (DEFERRED) at the first
// counter step, or by the device sampler before it starts; a group that can never fit is
// refused either way. *_built says when, *_deferred why it waited.
std::vector<Agent>& ctr_want     = *new std::vector<Agent>();  // the agents to build for
std::vector<Agent>& dev_want     = *new std::vector<Agent>();
std::string&        ctr_built    = *new std::string("no");
std::string&        dev_built    = *new std::string("no");
std::string&        ctr_deferred = *new std::string();
std::string&        dev_deferred = *new std::string();
std::atomic<uint64_t> n_ctr_builds{0}, n_dev_builds{0};

// ---- dispatch counting ----------------------------------------------------------------------
bool                     ctr_service = false;  // its context, buffer and service exist (flush, fini)
bool                     ctr_ok   = false;  // ruppell_counters may start it (under m_switch after init)
rocprofiler_context_id_t ctx_ctr  = {0};
rocprofiler_buffer_id_t  buf_ctr  = {0};
FILE*                    f_ctr    = nullptr;
std::mutex&              m_ctr    = *new std::mutex();   // the buffer callback's writes
std::mutex&              m_switch = *new std::mutex();   // ruppell_counters
std::atomic<int>         cur_group{0};
bool                     counting       = false;   // under m_switch
bool                     turned_detail  = false;   // counting turned dispatch records on
int                      next_group     = 0;
std::atomic<bool>        finished{false};
std::atomic<uint64_t>    n_counter_steps{0}, n_start_fail{0}, n_stop_fail{0}, n_counted{0}, n_filtered{0},
    n_values{0}, n_headers{0}, n_recs{0}, n_unknown{0}, n_ctr_drops{0}, n_ctr_callbacks{0};
std::atomic<uint64_t>    ctr_cb_ns{0}, ctr_cb_max_ns{0};
std::string&             last_switch_error = *new std::string();

constexpr uint64_t     kFilterCap = 1u << 20;
std::atomic<uint8_t>*  excluded   = nullptr;   // per kernel id, 1: left out by RUPPELL_COUNTER_KERNELS
std::regex*            filter_re  = nullptr;
std::atomic<uint64_t>  n_kernels_in{0}, n_kernels_out{0};

// ---- device counting ------------------------------------------------------------------------
bool                     dev_ok   = false;
rocprofiler_context_id_t ctx_dev  = {0};
FILE*                    f_dev    = nullptr;
double                   dev_hz   = 0;
rocprofiler_counter_config_id_t* dev_cfg_slots = nullptr;   // stable addresses for the callback
std::thread*             dev_thread = nullptr;
std::mutex&              m_dev      = *new std::mutex();
std::condition_variable& cv_dev     = *new std::condition_variable();
bool                     stop_dev   = false;
std::atomic<uint64_t>    n_samples{0}, n_sample_fail{0}, sample_ns{0}, sample_max_ns{0}, n_late{0},
    n_start_tries{0}, dev_start_ts{0}, dev_first_ts{0}, dev_last_ts{0}, n_dev_recs{0}, n_dev_unknown{0};
std::atomic<int>         dev_set_status{-1};
std::string&             dev_start_error  = *new std::string();
std::string&             dev_spec         = *new std::string();   // RUPPELL_DEVICE_COUNTERS before the @
uint16_t                 dev_group_base   = 0;   // the device group's number: after the dispatch groups
std::string&             dev_sample_error = *new std::string();

const char*
status_name(rocprofiler_status_t s)
{
    const char* n = rocprofiler_get_status_name(s);
    return n ? n : "?";
}

std::string
jstr(const std::string& s)
{
    std::string o = "\"";
    for(char c : s)
    {
        if(c == '"' || c == '\\')
        {
            o += '\\';
            o += c;
        }
        else if(static_cast<unsigned char>(c) < 0x20)
        {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        }
        else
            o += c;
    }
    return o + "\"";
}

std::string
env(const char* k)
{
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

std::vector<std::string>
split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::string              cur;
    for(char c : s)
    {
        if(c == sep)
        {
            out.push_back(cur);
            cur.clear();
        }
        else if(c != ' ' && c != '\t')
            cur += c;
    }
    out.push_back(cur);
    out.erase(std::remove(out.begin(), out.end(), std::string()), out.end());
    return out;
}

// "raw:A,B;C,D", or the resolved form "set/group=A,B;set/group=C"
std::vector<Group>
parse_groups(const std::string& spec)
{
    std::vector<Group> out;
    if(spec.rfind("raw:", 0) == 0)
    {
        int n = 0;
        for(const auto& g : split(spec.substr(4), ';'))
            out.push_back({"raw", "r" + std::to_string(++n), split(g, ',')});
        return out;
    }
    for(const auto& item : split(spec, ';'))
    {
        auto eq = item.find('=');
        if(eq == std::string::npos) continue;
        std::string sg = item.substr(0, eq);
        auto        sl = sg.find('/');
        out.push_back({sl == std::string::npos ? sg : sg.substr(0, sl), sl == std::string::npos ? "g1" : sg.substr(sl + 1),
                       split(item.substr(eq + 1), ',')});
    }
    return out;
}

// The groups a switch asks for on one architecture: raw:... as written, otherwise what
// `ruppell run` resolved into <prefix><ARCH>.
std::vector<Group>
groups_for(const std::string& spec, const std::string& prefix, const std::string& arch, std::string& why)
{
    if(spec.rfind("raw:", 0) == 0) return parse_groups(spec);
    std::string up = arch;
    for(char& c : up)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::string r = env((prefix + up).c_str());
    if(r.empty())
    {
        why = "the sets '" + spec + "' are not resolved for " + arch + " (" + prefix + up +
              " is unset: start the program with `ruppell run`, or give raw:NAME,...)";
        return {};
    }
    return parse_groups(r);
}

rocprofiler_status_t
collect_agents(rocprofiler_agent_version_t, const void** agents, size_t n, void* user)
{
    auto* out = static_cast<std::vector<Agent>*>(user);
    for(size_t i = 0; i < n; ++i)
    {
        const auto* a = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if(a->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        Agent g;
        g.handle      = a->id.handle;
        g.arch        = a->name ? a->name : "";
        g.index       = a->logical_node_type_id;
        g.hip_visible = a->runtime_visibility.hip != 0;
        out->push_back(g);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
collect_counters(rocprofiler_agent_id_t, rocprofiler_counter_id_t* ids, size_t n, void* user)
{
    static_cast<std::vector<rocprofiler_counter_id_t>*>(user)->assign(ids, ids + n);
    return ROCPROFILER_STATUS_SUCCESS;
}

struct Info
{
    uint64_t    id      = 0;
    bool        derived = false;
    std::string what;   // block, or a derived counter's expression
    std::string dims;   // "DIMENSION_XCC[8],..."
    std::string description;
    bool        has_xcc = false;
    uint64_t    xcc_dim = 0;
    size_t      xcc_n   = 0;   // the XCC dimension's instances
};

std::unordered_map<std::string, Info>
agent_counters(const Agent& a)
{
    std::unordered_map<std::string, Info> out;
    std::vector<rocprofiler_counter_id_t> ids;
    rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t{a.handle}, collect_counters, &ids);
    for(auto id : ids)
    {
        rocprofiler_counter_info_v1_t info{};
        info.size = sizeof(info);
        if(rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info) != ROCPROFILER_STATUS_SUCCESS ||
           !info.name)
            continue;
        Info x;
        x.id      = id.handle;
        x.derived = info.is_derived;
        x.what    = info.is_derived ? (info.expression ? info.expression : "") : (info.block ? info.block : "");
        x.description = info.description ? info.description : "";
        for(uint64_t d = 0; d < info.dimensions_count; ++d)
        {
            const auto* dim = info.dimensions ? info.dimensions[d] : nullptr;
            if(!dim || !dim->name) continue;
            const std::string dn = dim->name;
            if(!x.dims.empty()) x.dims += ',';
            x.dims += dn + "[" + std::to_string(dim->instance_size) + "]";
            if(dn.find("XCC") != std::string::npos)
            {
                x.has_xcc = true;
                x.xcc_dim = dim->id;
                x.xcc_n   = dim->instance_size;
            }
        }
        out.emplace(info.name, x);
    }
    return out;
}

// counters.tsv's unit. The SDK gives none: a base counter counts events (some count cycles, as
// its description says); a derived counter's unit is the one its description names, or "-".
std::string
unit_of(const Info& x)
{
    std::string d = x.description;
    for(char& c : d)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(!x.derived) return "count";
    if(d.find("kilobyte") != std::string::npos) return "KB";
    if(d.find("percent") != std::string::npos) return "percent";
    if(d.find("bytes") != std::string::npos) return "bytes";
    return "-";
}

std::map<std::string, uint16_t>& numbers = *new std::map<std::string, uint16_t>();  // under m_tsv

// A counter's number, unique across groups and services: a counter in several groups has one
// (records.md). Called under m_tsv.
uint16_t
number_of(const std::string& name)
{
    auto it = numbers.find(name);
    if(it != numbers.end()) return it->second;
    const auto n = static_cast<uint16_t>(numbers.size());
    numbers.emplace(name, n);
    return n;
}

enum Built
{
    BUILT,
    REFUSED,   // a group can never be collected: an unknown counter, more than a block's slots
    DEFERRED,  // the SDK could not tell yet (HSA not initialized at tool init): try again later
};

// Whether a refusal of rocprofiler_create_counter_config is about the group itself, rather than
// the SDK not being ready.
bool
definitive(rocprofiler_status_t st)
{
    switch(st)
    {
        case ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT:
        case ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT:
        case ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND:
        case ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND:
        case ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED:
        case ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND:
        case ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS:
        case ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED:
        case ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND: return true;
        default: return false;
    }
}

// Configurations for every group on every agent, and counters.tsv's rows, into svc; its groups are
// numbered from group_base in the records. Anything but BUILT comes with `why`, naming the group
// and its counters by block; then svc gets only the groups and their checks (for meta.json), and
// the configurations made are destroyed.
Built
build(Service& svc, uint16_t group_base, const std::string& spec, const std::string& prefix,
      const std::vector<Agent>& agents, std::string& why)
{
    if(agents.empty())
    {
        why = "no GPU agent";
        return REFUSED;
    }
    Service                                             s;
    std::vector<std::unordered_map<std::string, Info>> known;
    for(const auto& a : agents)
    {
        std::string w;
        auto        gs = groups_for(spec, prefix, a.arch, w);
        if(gs.empty())
        {
            why = w.empty() ? "no groups in '" + spec + "'" : w;
            return REFUSED;
        }
        if(s.groups.empty())
            s.groups = gs;
        else if(gs.size() != s.groups.size())
        {
            why = "the agents' architectures resolve to different numbers of groups";
            return REFUSED;
        }
        known.push_back(agent_counters(a));
    }
    if(s.groups.size() > 0xFFFE)
    {
        why = "too many groups";
        return REFUSED;
    }
    const size_t na = agents.size(), ng = s.groups.size();
    s.agents        = agents;
    s.cfg.assign(ng * na, rocprofiler_counter_config_id_t{0});
    s.targets.assign(ng * na, {});
    s.status.assign(ng, "ok");
    auto fail = [&](Built b, size_t g, const std::string& st) {
        s.status[g] = st;
        why         = "group " + s.groups[g].set + "/" + s.groups[g].name + ": " + st;
        for(auto& c : s.cfg)
            if(c.handle) rocprofiler_destroy_counter_config(c);
        svc.groups     = s.groups;
        svc.group_base = group_base;
        svc.status = s.status;
        return b;
    };
    for(size_t ai = 0; ai < na; ++ai)
        if(known[ai].empty()) return fail(DEFERRED, 0, "the SDK lists no counters for " + agents[ai].arch + " yet");
    for(size_t g = 0; g < ng; ++g)
    {
        const Group& grp = s.groups[g];
        for(size_t ai = 0; ai < na; ++ai)
        {
            const auto&                           kn = known[ai];
            std::vector<rocprofiler_counter_id_t> ids;
            std::map<std::string, int>            blocks;
            for(const auto& c : grp.counters)
            {
                auto it = kn.find(c);
                if(it == kn.end()) return fail(REFUSED, g, "unknown counter " + c + " on " + agents[ai].arch);
                ids.push_back(rocprofiler_counter_id_t{it->second.id});
                ++blocks[it->second.derived ? std::string("derived") : it->second.what];
            }
            rocprofiler_counter_config_id_t cfg{};
            rocprofiler_status_t st = rocprofiler_create_counter_config(rocprofiler_agent_id_t{agents[ai].handle},
                                                                        ids.data(), ids.size(), &cfg);
            if(st != ROCPROFILER_STATUS_SUCCESS)
            {
                std::string by;
                for(const auto& [b, n] : blocks)
                    by += (by.empty() ? "" : ", ") + b + " " + std::to_string(n);
                std::string cs;
                for(const auto& c : grp.counters)
                    cs += (cs.empty() ? "" : ",") + c;
                return fail(definitive(st) ? REFUSED : DEFERRED, g,
                            std::string(status_name(st)) + " on " + agents[ai].arch + " (" + cs + "; by block: " + by + ")");
            }
            s.cfg[g * na + ai] = cfg;
        }
    }
    // counters.tsv: one line per group and counter (records.md); a counter split by XCC has a
    // number per instance, NAME[xcc=i], reduce "none"
    std::lock_guard<std::mutex> lk(m_tsv);
    s.group_base = group_base;
    for(size_t g = 0; g < ng; ++g)
    {
        const Group&      grp = s.groups[g];
        const std::string gn  = std::to_string(group_base + g);
        for(const auto& c : grp.counters)
        {
            const Info& x0    = known[0].at(c);
            const bool  split = per_xcc && x0.has_xcc && x0.xcc_n > 1;
            const auto  row   = [&](const std::string& name, uint16_t n) {
                tsv_rows.push_back(gn + "\t" + std::to_string(n) + "\t" + name + "\t" + grp.set + "\t" +
                                   (x0.derived ? "derived" : "base") + "\t" + x0.what + "\t" + (split ? "none" : "sum") +
                                   "\t" + unit_of(x0));
            };
            Target t;
            if(split)
                for(size_t i = 0; i < x0.xcc_n; ++i)
                {
                    const std::string name = c + "[xcc=" + std::to_string(i) + "]";
                    t.by_xcc.push_back(number_of(name));
                    row(name, t.by_xcc.back());
                }
            else
            {
                t.number = number_of(c);
                row(c, t.number);
            }
            for(size_t ai = 0; ai < na; ++ai)
            {
                const Info& x = known[ai].at(c);
                Target      ta = t;
                ta.has_xcc     = x.has_xcc;
                ta.xcc_dim     = x.xcc_dim;
                s.targets[g * na + ai][x.id] = ta;
            }
        }
    }
    svc = s;
    return BUILT;
}

void
write_tsv()
{
    std::lock_guard<std::mutex> lk(m_tsv);
    FILE*                       f = std::fopen((out_dir + "/counters.tsv").c_str(), "w");
    if(!f) return;
    std::fprintf(f, "group\tcounter\tname\tset\tkind\tsource\treduce\tunit\n");
    for(const auto& r : tsv_rows)
        std::fprintf(f, "%s\n", r.c_str());
    std::fclose(f);
}

int
agent_slot(const std::vector<Agent>& agents, uint64_t handle)
{
    for(size_t i = 0; i < agents.size(); ++i)
        if(agents[i].handle == handle) return static_cast<int>(i);
    return -1;
}

void
max_into(std::atomic<uint64_t>& m, uint64_t v)
{
    uint64_t prev = m.load();
    while(v > prev && !m.compare_exchange_weak(prev, v)) {}
}

// ---- dispatch counting: the callback and the buffer -----------------------------------------

// The group and step each counted dispatch was given, by dispatch id, for the buffer callback:
// the buffered service's records do not carry the callback's user data (it reaches only the
// callback service's record callback; SDK 1.3.2, runner job 90806 read it back as another group).
// A ring, written lock-free by the dispatch callback; a slot holds its dispatch id, so a slot
// reused by a dispatch 2^20 later reads as unknown rather than as the wrong group.
constexpr uint64_t kGroupRing = 1u << 20;
struct GroupSlot
{
    std::atomic<uint64_t> dispatch{0};
    std::atomic<uint32_t> group{0};
    std::atomic<uint32_t> step{0};
};
GroupSlot* group_ring = nullptr;   // allocated with the dispatch counting service
std::atomic<uint64_t> n_group_unknown{0};
std::atomic<uint32_t> cur_step{0};  // the counter step's id (ruppell_counters)

bool
slot_of(uint64_t dispatch_id, int& group, uint32_t& step)
{
    const GroupSlot& s = group_ring[dispatch_id & (kGroupRing - 1)];
    if(s.dispatch.load(std::memory_order_acquire) != dispatch_id) return false;
    group = static_cast<int>(s.group.load(std::memory_order_relaxed));
    step  = s.step.load(std::memory_order_relaxed);
    return true;
}

void
on_dispatch(rocprofiler_dispatch_counting_service_data_t d, rocprofiler_counter_config_id_t* config,
            rocprofiler_user_data_t*, void*)
{
    const uint64_t kid = d.dispatch_info.kernel_id;
    if(excluded && kid < kFilterCap && excluded[kid].load(std::memory_order_relaxed))
    {
        n_filtered.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int ai = agent_slot(ctr.agents, d.dispatch_info.agent_id.handle);
    if(ai < 0) return;
    const int g = cur_group.load(std::memory_order_relaxed);
    *config     = ctr.cfg[static_cast<size_t>(g) * ctr.agents.size() + ai];
    GroupSlot& s = group_ring[d.dispatch_info.dispatch_id & (kGroupRing - 1)];
    s.group.store(static_cast<uint32_t>(g), std::memory_order_relaxed);
    s.step.store(cur_step.load(std::memory_order_relaxed), std::memory_order_relaxed);
    s.dispatch.store(d.dispatch_info.dispatch_id, std::memory_order_release);
    n_counted.fetch_add(1, std::memory_order_relaxed);
}

// one dispatch's values, summed over dimension instances while its records stream past
struct Acc
{
    uint64_t dispatch = 0, agent = 0;
    int      group    = -1;   // in the service, -1 unknown
    uint32_t step     = 0;
    struct E
    {
        uint16_t number;
        double   v;
    };
    std::vector<E> e;

    void start(uint64_t d, uint64_t a)
    {
        dispatch = d;
        agent    = a;
        if(!slot_of(d, group, step)) group = -1;
    }
    void emit(FILE* f)
    {
        if(!dispatch) return;
        const auto g = static_cast<uint16_t>(ctr.group_base + (group < 0 ? 0 : group));
        for(const auto& x : e)
        {
            ruppell_counter_rec r{dispatch, 0, 0, agent, step, g, x.number, x.v};   // key0: the dispatch id (ROCm)
            if(f) std::fwrite(&r, sizeof(r), 1, f);
            ++n_recs;
        }
        dispatch = agent = 0;
        group = -1;
        step  = 0;
        e.clear();
    }
    void add(uint16_t number, double v)
    {
        for(auto& x : e)
            if(x.number == number)
            {
                x.v += v;
                return;
            }
        e.push_back({number, v});
    }
};
Acc& acc_ctr = *new Acc();   // under m_ctr

void
on_counter_buffer(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers,
                  size_t num_headers, void*, uint64_t drop_count)
{
    const uint64_t t0 = ruppell_tool::now();
    n_ctr_drops += drop_count;
    std::lock_guard<std::mutex> lk(m_ctr);
    Acc&                        acc = acc_ctr;
    const size_t   na    = ctr.agents.size();
    uint64_t       c_key = ~0ull, c_cid = 0;   // the last target looked up
    const Target*  c_t   = nullptr;
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* h = headers[i];
        if(h->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS) continue;
        if(h->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            // a counted dispatch; its values follow (counted in meta.json, not recorded)
            auto* r = static_cast<rocprofiler_dispatch_counting_service_record_t*>(h->payload);
            acc.emit(f_ctr);
            acc.start(r->dispatch_info.dispatch_id, r->dispatch_info.agent_id.handle);
            ++n_headers;
            if(acc.group < 0) ++n_group_unknown;
        }
        else if(h->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            auto* r = static_cast<rocprofiler_counter_record_t*>(h->payload);
            if(r->dispatch_id != acc.dispatch)
            {
                acc.emit(f_ctr);
                acc.start(r->dispatch_id, r->agent_id.handle);
            }
            rocprofiler_counter_id_t cid{};
            rocprofiler_query_record_counter_id(r->id, &cid);
            const int ai = agent_slot(ctr.agents, r->agent_id.handle);
            if(ai < 0 || acc.group < 0 || acc.group >= static_cast<int>(ctr.groups.size()))
            {
                ++n_unknown;
                continue;
            }
            const uint64_t key = static_cast<uint64_t>(acc.group) * na + static_cast<uint64_t>(ai);
            if(key != c_key || cid.handle != c_cid)
            {
                auto it = ctr.targets[key].find(cid.handle);
                c_t     = it == ctr.targets[key].end() ? nullptr : &it->second;
                c_key   = key;
                c_cid   = cid.handle;
            }
            if(!c_t)
            {
                ++n_unknown;
                continue;
            }
            size_t pos = 0;
            if(!c_t->by_xcc.empty() &&
               rocprofiler_query_record_dimension_position(r->id, c_t->xcc_dim, &pos) == ROCPROFILER_STATUS_SUCCESS)
                acc.add(c_t->of(pos), r->counter_value);
            else
                acc.add(c_t->number, r->counter_value);
            ++n_values;
        }
    }
    acc.emit(f_ctr);  // a dispatch continued in the next callback gets a second record: readers sum
    const uint64_t t1 = ruppell_tool::now();
    ++n_ctr_callbacks;
    ctr_cb_ns += t1 - t0;
    max_into(ctr_cb_max_ns, t1 - t0);
}

// ---- device counting: the configuration callback and the sampler ----------------------------

void
on_device_config(rocprofiler_context_id_t ctx, rocprofiler_agent_id_t, rocprofiler_device_counting_agent_cb_t set_config,
                 void* user)
{
    dev_set_status = static_cast<int>(set_config(ctx, *static_cast<rocprofiler_counter_config_id_t*>(user)));
}

void
sample_loop()
{
    // The context starts here rather than at init: a read needs HSA initialized, which it is not
    // yet when the tool initializes. A failed start is retried every 10 ms.
    std::vector<rocprofiler_counter_record_t> recs(1u << 16);
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / dev_hz));
    auto       next   = std::chrono::steady_clock::now();
    bool       started = false;
    // The SDK's values count from the context's start (probe A6: never decreasing, runner job
    // 90857); each record is the difference from the previous read, over [previous read, this read]
    struct Sum
    {
        uint64_t agent;
        uint16_t number;
        double   v;
    };
    std::vector<Sum>                    sums;
    std::map<std::pair<uint64_t, uint16_t>, double> last;   // (agent, counter): the previous read's value
    uint64_t                            t_last = 0;
    std::unique_lock<std::mutex> lk(m_dev);
    while(!stop_dev)
    {
        if(dev_built == "no")
        {
            // the configurations init could not build (DEFERRED): built here, before the start
            std::string why;
            ++n_dev_builds;
            const Built b = build(dev, dev_group_base, dev_spec, "RUPPELL_DEVICE_COUNTER_GROUPS_", dev_want, why);
            if(b == REFUSED)
            {
                dev.refused = why;
                std::fprintf(stderr, "[ruppell gray-parrot] device counters refused: %s\n", why.c_str());
                break;
            }
            if(b == DEFERRED)
            {
                dev_deferred = why;
                cv_dev.wait_for(lk, std::chrono::milliseconds(10), [] { return stop_dev; });
                continue;
            }
            for(size_t ai = 0; ai < dev.agents.size(); ++ai)
                dev_cfg_slots[ai] = dev.cfg[ai];
            dev_built = "by the sampler";
            write_tsv();
        }
        if(!started)
        {
            ++n_start_tries;
            rocprofiler_status_t st = rocprofiler_start_context(ctx_dev);
            if(st == ROCPROFILER_STATUS_SUCCESS)
            {
                started      = true;
                dev_start_ts = ruppell_tool::now();
                t_last       = dev_start_ts;
                next         = std::chrono::steady_clock::now();
            }
            else
            {
                if(dev_start_error.empty()) dev_start_error = status_name(st);
                cv_dev.wait_for(lk, std::chrono::milliseconds(10), [] { return stop_dev; });
                continue;
            }
        }
        size_t                  cnt = recs.size();
        const uint64_t          t0  = ruppell_tool::now();
        rocprofiler_status_t st = rocprofiler_sample_device_counting_service(ctx_dev, rocprofiler_user_data_t{},
                                                                            ROCPROFILER_COUNTER_FLAG_NONE, recs.data(), &cnt);
        const uint64_t t1 = ruppell_tool::now();
        if(st != ROCPROFILER_STATUS_SUCCESS)
        {
            ++n_sample_fail;
            if(dev_sample_error.empty()) dev_sample_error = status_name(st);
        }
        else
        {
            sums.clear();
            for(size_t i = 0; i < cnt; ++i)
            {
                const auto&              r = recs[i];
                rocprofiler_counter_id_t cid{};
                rocprofiler_query_record_counter_id(r.id, &cid);
                const int ai = agent_slot(dev.agents, r.agent_id.handle);
                if(ai < 0)
                {
                    ++n_dev_unknown;
                    continue;
                }
                auto it = dev.targets[static_cast<size_t>(ai)].find(cid.handle);   // one group: [0 * na + ai]
                if(it == dev.targets[static_cast<size_t>(ai)].end())
                {
                    ++n_dev_unknown;
                    continue;
                }
                size_t   pos    = 0;
                uint16_t number = it->second.number;
                if(!it->second.by_xcc.empty() &&
                   rocprofiler_query_record_dimension_position(r.id, it->second.xcc_dim, &pos) == ROCPROFILER_STATUS_SUCCESS)
                    number = it->second.of(pos);
                bool found = false;
                for(auto& s : sums)
                    if(s.agent == r.agent_id.handle && s.number == number)
                    {
                        s.v += r.counter_value;
                        found = true;
                        break;
                    }
                if(!found) sums.push_back({r.agent_id.handle, number, r.counter_value});
            }
            const auto g = static_cast<uint16_t>(dev.group_base);
            for(const auto& s : sums)
            {
                double& prev = last[{s.agent, s.number}];
                ruppell_device_counter_rec o{t_last, t1, s.agent, 0, g, s.number, s.v - prev};   // step 0: placed by time
                prev = s.v;
                if(f_dev) std::fwrite(&o, sizeof(o), 1, f_dev);
                ++n_dev_recs;
            }
            t_last = t1;
            ++n_samples;
            sample_ns += t1 - t0;
            max_into(sample_max_ns, t1 - t0);
            uint64_t zero = 0;
            dev_first_ts.compare_exchange_strong(zero, t0);
            dev_last_ts = t1;
        }
        next += period;
        const auto now = std::chrono::steady_clock::now();
        if(next < now)
        {
            ++n_late;   // the read took longer than the period: no catching up
            next = now;
        }
        cv_dev.wait_until(lk, next, [] { return stop_dev; });
    }
    if(started) rocprofiler_stop_context(ctx_dev);
}

double
parse_rate(const std::string& r)
{
    // "100Hz", "1kHz", "10ms", "500us", or a number of Hz
    std::size_t idx = 0;
    double      v   = 0;
    try
    {
        v = std::stod(r, &idx);
    } catch(...)
    {
        return 0;
    }
    std::string u = r.substr(idx);
    for(char& c : u)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if(u.empty() || u == "hz") return v;
    if(u == "khz") return v * 1e3;
    if(u == "s") return v > 0 ? 1.0 / v : 0;
    if(u == "ms") return v > 0 ? 1e3 / v : 0;
    if(u == "us") return v > 0 ? 1e6 / v : 0;
    return 0;
}

std::vector<Agent>
gpu_agents()
{
    std::vector<Agent> all;
    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, collect_agents, sizeof(rocprofiler_agent_v0_t),
                                       &all);
    return all;
}

std::string
agents_json(const std::vector<Agent>& a)
{
    std::string o = "[";
    for(size_t i = 0; i < a.size(); ++i)
        o += (i ? ", " : "") + std::string("{\"index\": ") + std::to_string(a[i].index) + ", \"handle\": " +
             std::to_string(a[i].handle) + ", \"arch\": " + jstr(a[i].arch) + "}";
    return o + "]";
}

std::string
groups_json(const Service& s)
{
    std::string o = "[";
    for(size_t g = 0; g < s.groups.size(); ++g)
    {
        std::string cs;
        for(const auto& c : s.groups[g].counters)
            cs += (cs.empty() ? "" : ", ") + jstr(c);
        // passes: 1, or the group was refused (gray-parrot never collects in several passes)
        o += (g ? ", " : "") + std::string("{\"group\": ") + std::to_string(s.group_base + g) + ", \"set\": " +
             jstr(s.groups[g].set) + ", \"name\": " + jstr(s.groups[g].name) + ", \"counters\": [" + cs +
             "], \"passes\": 1, \"check\": " + jstr(g < s.status.size() ? s.status[g] : "unchecked") + "}";
    }
    return o + "]";
}

std::string
sets_json(const std::string& spec)
{
    if(spec.rfind("raw:", 0) == 0) return "[\"raw\"]";
    std::string o;
    for(const auto& x : split(spec, ','))
        o += (o.empty() ? "" : ", ") + jstr(x);
    return "[" + o + "]";
}
}  // namespace

void
configure(const std::string& dir, rocprofiler_callback_thread_t cb_thread)
{
    meta_spec     = env("RUPPELL_COUNTERS");
    meta_dev_spec = env("RUPPELL_DEVICE_COUNTERS");
    if(meta_spec.empty() && meta_dev_spec.empty()) return;  // off: create nothing
    any_on  = true;
    out_dir = dir;
    per_xcc = env("RUPPELL_COUNTER_DIMS") == "xcc";
    auto all = gpu_agents();
    std::vector<Agent> visible;
    for(const auto& a : all)
        if(a.hip_visible) visible.push_back(a);
    if(visible.empty()) visible = all;   // the SDK's visibility is an estimate from the environment

    if(!meta_spec.empty())
    {
        const std::string scope = env("RUPPELL_COUNTER_SCOPE");
        std::string       why;
        Built             b = REFUSED;
        if(!scope.empty() && scope != "kernel")
            ctr.refused = "RUPPELL_COUNTER_SCOPE=" + scope + " is not built yet (plan 03, Part D)";
        else if((b = build(ctr, 0, meta_spec, "RUPPELL_COUNTER_GROUPS_", visible, why)) == REFUSED)
            ctr.refused = why;
        else
        {
            ++n_ctr_builds;
            ctr_want = visible;
            if(b == BUILT)
                ctr_built = "at init";
            else
                ctr_deferred = why;
            kernel_filter = env("RUPPELL_COUNTER_KERNELS");
            if(!kernel_filter.empty())
            {
                try
                {
                    filter_re = new std::regex(kernel_filter, std::regex::ECMAScript | std::regex::optimize);
                    excluded  = new std::atomic<uint8_t>[kFilterCap]();
                } catch(const std::regex_error& e)
                {
                    ctr.refused = "RUPPELL_COUNTER_KERNELS is not a regex: " + std::string(e.what());
                }
            }
        }
        if(ctr.refused.empty())
        {
            group_ring = new GroupSlot[kGroupRing];   // 16 MiB, only with counters
            constexpr size_t     bytes = 32u << 20;
            rocprofiler_status_t s1    = rocprofiler_create_context(&ctx_ctr);
            rocprofiler_status_t s2    = s1 == ROCPROFILER_STATUS_SUCCESS
                                             ? rocprofiler_create_buffer(ctx_ctr, bytes, bytes - bytes / 8,
                                                                         ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_counter_buffer,
                                                                         nullptr, &buf_ctr)
                                             : s1;
            rocprofiler_status_t s3 =
                s2 == ROCPROFILER_STATUS_SUCCESS
                    ? rocprofiler_configure_buffer_dispatch_counting_service(ctx_ctr, buf_ctr, on_dispatch, nullptr)
                    : s2;
            rocprofiler_status_t s4 =
                s3 == ROCPROFILER_STATUS_SUCCESS ? rocprofiler_assign_callback_thread(buf_ctr, cb_thread) : s3;
            if(s4 != ROCPROFILER_STATUS_SUCCESS)
                ctr.refused = std::string("the dispatch counting service: ") + status_name(s4);
            else if((f_ctr = std::fopen((out_dir + "/counters.bin").c_str(), "wb")) == nullptr)
                ctr.refused = "cannot open counters.bin";
            else
                ctr_service = ctr_ok = true;
        }
        if(!ctr_ok)
            std::fprintf(stderr, "[ruppell gray-parrot] counters refused, tracing goes on: %s\n", ctr.refused.c_str());
        else
            std::fprintf(stderr, "[ruppell gray-parrot] counters configured: %zu groups on %zu GPUs, %s, never started yet%s%s\n",
                         ctr.groups.size(), ctr_want.size(), ctr_built == "no" ? "configurations deferred" : "configurations built",
                         ctr_deferred.empty() ? "" : ": ", ctr_deferred.c_str());
    }

    if(!meta_dev_spec.empty())
    {
        const auto at = meta_dev_spec.rfind('@');
        dev_spec      = at == std::string::npos ? meta_dev_spec : meta_dev_spec.substr(0, at);
        dev_hz        = at == std::string::npos ? 0 : parse_rate(meta_dev_spec.substr(at + 1));
        const std::string want = env("RUPPELL_DEVICE_AGENTS");
        for(const auto& a : all)
        {
            bool take = a.hip_visible;
            if(!want.empty())
            {
                take = false;
                for(const auto& w : split(want, ','))
                    if(w == std::to_string(a.index)) take = true;
            }
            if(take) dev_want.push_back(a);
        }
        if(want.empty() && dev_want.empty()) dev_want = all;
        std::string why;
        Built       b = REFUSED;
        if(dev_hz <= 0)
            dev.refused = "RUPPELL_DEVICE_COUNTERS=" + meta_dev_spec + ": want SPEC@RATE (100Hz, 1kHz, 10ms)";
        else if((b = build(dev, dev_group_base = static_cast<uint16_t>(ctr.groups.size()), dev_spec,
                           "RUPPELL_DEVICE_COUNTER_GROUPS_", dev_want, why)) == REFUSED)
            dev.refused = why;
        else
        {
            ++n_dev_builds;
            if(b == BUILT)
                dev_built = "at init";
            else
                dev_deferred = why;
            if(dev.groups.size() != 1)
                dev.refused = "the device service counts one group; '" + dev_spec + "' has " + std::to_string(dev.groups.size());
            else
            {
                dev_cfg_slots = new rocprofiler_counter_config_id_t[dev_want.size()]();
                rocprofiler_status_t st = rocprofiler_create_context(&ctx_dev);
                for(size_t ai = 0; ai < dev_want.size() && st == ROCPROFILER_STATUS_SUCCESS; ++ai)
                {
                    if(b == BUILT) dev_cfg_slots[ai] = dev.cfg[ai];
                    st = rocprofiler_configure_device_counting_service(ctx_dev, rocprofiler_buffer_id_t{0},
                                                                       rocprofiler_agent_id_t{dev_want[ai].handle},
                                                                       on_device_config, &dev_cfg_slots[ai]);
                }
                if(st != ROCPROFILER_STATUS_SUCCESS)
                    dev.refused = std::string("the device counting service: ") + status_name(st);
                else if((f_dev = std::fopen((out_dir + "/device_counters.bin").c_str(), "wb")) == nullptr)
                    dev.refused = "cannot open device_counters.bin";
                else
                    dev_ok = true;
            }
        }
        if(!dev_ok)
            std::fprintf(stderr, "[ruppell gray-parrot] device counters refused, tracing goes on: %s\n", dev.refused.c_str());
    }
    write_tsv();
}

void
start()
{
    if(!dev_ok) return;
    dev_thread = new std::thread(sample_loop);
}

void
kernel_symbol(uint64_t kernel_id, const char* name)
{
    if(!excluded || !filter_re) return;
    const bool in = name && std::regex_search(name, *filter_re);
    if(in)
        ++n_kernels_in;
    else
        ++n_kernels_out;
    if(kernel_id < kFilterCap) excluded[kernel_id].store(in ? 0 : 1, std::memory_order_relaxed);
}

void
flush()
{
    if(ctr_service) rocprofiler_flush_buffer(buf_ctr);
    {
        std::lock_guard<std::mutex> lk(m_ctr);
        if(f_ctr) std::fflush(f_ctr);
    }
    // the sampler writes f_dev under m_dev, between reads
    if(f_dev)
    {
        std::lock_guard<std::mutex> lk(m_dev);
        std::fflush(f_dev);
    }
}

void
fini()
{
    if(!any_on || finished.exchange(true)) return;
    {
        std::lock_guard<std::mutex> lk(m_switch);
        if(counting && rocprofiler_stop_context(ctx_ctr) == ROCPROFILER_STATUS_SUCCESS) counting = false;
    }
    if(dev_thread)
    {
        {
            std::lock_guard<std::mutex> lk(m_dev);
            stop_dev = true;
        }
        cv_dev.notify_all();
        if(dev_thread->joinable()) dev_thread->join();
    }
    if(ctr_service) rocprofiler_flush_buffer(buf_ctr);
    {
        std::lock_guard<std::mutex> lk(m_ctr);
        if(f_ctr)
        {
            std::fclose(f_ctr);
            f_ctr = nullptr;
        }
    }
    if(f_dev)
    {
        std::fclose(f_dev);
        f_dev = nullptr;
    }
}

void
forget_in_child()
{
    finished = true;
    ctr_ok = ctr_service = dev_ok = false;
    dev_thread      = nullptr;  // a fork copies only the forking thread
    for(FILE** f : {&f_ctr, &f_dev})
        if(*f)
        {
            __fpurge(*f);
            std::fclose(*f);
            *f = nullptr;
        }
}

std::string
meta_json()
{
    if(!any_on) return "\"off\"";
    auto u = [](const std::atomic<uint64_t>& v) { return std::to_string(v.load()); };
    // docs/gray-parrot/records.md's keys first; "rocm": this tool's own diagnostics
    uint32_t maj = 0, min = 0, pat = 0;
    rocprofiler_get_version(&maj, &min, &pat);
    const std::vector<Agent>& ag = !ctr.agents.empty() ? ctr.agents : (!ctr_want.empty() ? ctr_want : dev_want);
    const std::string steps = env("RUPPELL_COUNTER_STEPS");
    std::string notes = jstr("ROCm's Peak TOPS Limiter: its state is unknown; amd-smi reports N/A in ruppell's MI350X "
                             "container (runner plan 03, probe A3)");
    notes += ", " + jstr("units: the SDK gives none; a base counter counts events (some count cycles, as its "
                         "description says), a derived counter's unit is the one its description names");
    if(!meta_spec.empty())
        notes += ", " + jstr("counting serializes every dispatch on the GPU while a counter step runs; counter steps "
                             "are left out of every time");
    if(!meta_dev_spec.empty())
        notes += ", " + jstr("device readings: differences of the SDK's values, which count from the context's start "
                             "(probe A6), over [previous read, this read]");
    std::string o = "{\n    \"version\": 1, \"platform\": \"rocm\", \"arch\": " + jstr(ag.empty() ? "" : ag[0].arch) +
                    ", \"collector\": " + jstr("rocprofiler-sdk " + std::to_string(maj) + "." + std::to_string(min) + "." +
                                               std::to_string(pat)) +
                    ",\n    \"scope\": " + (meta_spec.empty() ? "null" : "\"kernel\"") +
                    ", \"sets\": " + (meta_spec.empty() ? "[]" : sets_json(meta_spec)) + ", \"steps\": " +
                    (steps.empty() ? "\"1/100\"" : jstr(steps)) + ", \"counter_steps\": " + u(n_counter_steps) +
                    ",\n    \"groups\": " + groups_json(ctr) + ",\n    \"records\": " + u(n_recs) +
                    ", \"dropped\": {\"records\": " + u(n_ctr_drops) + ", \"reason\": \"the SDK's counter buffer\", "
                    "\"unattributed_values\": " + u(n_unknown) + "}, \"refused\": " +
                    (ctr.refused.empty() ? "null" : jstr(ctr.refused)) + ",\n    \"notes\": [" + notes + "]";
    if(meta_dev_spec.empty())
        o += ",\n    \"device\": null";
    else
        o += ",\n    \"device\": {\"set\": " + sets_json(dev_spec) + ", \"group\": " + std::to_string(dev.group_base) +
             ", \"rate_hz\": " + std::to_string(dev_hz) + ", \"samples\": " + u(n_samples) + ", \"records\": " +
             u(n_dev_recs) + ", \"refused\": " + (dev.refused.empty() ? "null" : jstr(dev.refused)) + "}";
    o += ",\n    \"rocm\": {\"dims\": " + jstr(per_xcc ? "xcc" : "sum");
    if(meta_spec.empty())
        o += ",\n    \"dispatch\": \"off\"";
    else
    {
        o += ",\n    \"dispatch\": {\"spec\": " + jstr(meta_spec) + ", \"configured\": " + (ctr_ok ? "true" : "false") +
             ", \"refused\": " + (ctr.refused.empty() ? "null" : jstr(ctr.refused)) + ", \"configs_built\": " +
             jstr(ctr_built) + ", \"deferred_because\": " + (ctr_deferred.empty() ? "null" : jstr(ctr_deferred)) +
             ", \"build_tries\": " + u(n_ctr_builds) + ", \"groups\": " + groups_json(ctr) +
             ", \"agents\": " + agents_json(ctr.agents) + ", \"kernel_filter\": " +
             (kernel_filter.empty() ? "null" : jstr(kernel_filter)) + ", \"kernels_in\": " + u(n_kernels_in) +
             ", \"kernels_out\": " + u(n_kernels_out) + ", \"counter_steps\": " + u(n_counter_steps) +
             ", \"start_failures\": " + u(n_start_fail) + ", \"stop_failures\": " + u(n_stop_fail) +
             ", \"last_switch_error\": " + (last_switch_error.empty() ? "null" : jstr(last_switch_error)) +
             ", \"dispatches_counted\": " + u(n_counted) + ", \"dispatches_filtered\": " + u(n_filtered) +
             ", \"dispatch_headers\": " + u(n_headers) + ", \"value_records\": " + u(n_values) +
             ", \"records_written\": " + u(n_recs) + ", \"unknown_records\": " + u(n_unknown) +
             ", \"dispatches_of_unknown_group\": " + u(n_group_unknown) +
             ", \"dropped_records\": " + u(n_ctr_drops) + ", \"buffer_callbacks\": " + u(n_ctr_callbacks) +
             ", \"callback_total_ns\": " + u(ctr_cb_ns) + ", \"callback_max_ns\": " + u(ctr_cb_max_ns) + "}";
    }
    if(meta_dev_spec.empty())
        o += ",\n    \"device\": \"off\"";
    else
    {
        o += ",\n    \"device\": {\"spec\": " + jstr(meta_dev_spec) + ", \"rate_hz\": " + std::to_string(dev_hz) +
             ", \"configured\": " + (dev_ok ? "true" : "false") + ", \"refused\": " +
             (dev.refused.empty() ? "null" : jstr(dev.refused)) + ", \"configs_built\": " + jstr(dev_built) +
             ", \"deferred_because\": " + (dev_deferred.empty() ? "null" : jstr(dev_deferred)) +
             ", \"build_tries\": " + u(n_dev_builds) + ", \"groups\": " + groups_json(dev) +
             ", \"agents\": " + agents_json(dev.agents) + ", \"start_tries\": " + u(n_start_tries) +
             ", \"start_error\": " + (dev_start_error.empty() ? "null" : jstr(dev_start_error)) +
             ", \"set_config_status\": " + std::to_string(dev_set_status.load()) + ", \"started_ts\": " + u(dev_start_ts) +
             ", \"samples\": " + u(n_samples) + ", \"sample_failures\": " + u(n_sample_fail) +
             ", \"sample_error\": " + (dev_sample_error.empty() ? "null" : jstr(dev_sample_error)) +
             ", \"read_total_ns\": " + u(sample_ns) + ", \"read_max_ns\": " + u(sample_max_ns) +
             ", \"late\": " + u(n_late) + ", \"first_ts\": " + u(dev_first_ts) + ", \"last_ts\": " + u(dev_last_ts) +
             ", \"records_written\": " + u(n_dev_recs) + ", \"unknown_records\": " + u(n_dev_unknown) + "}";
    }
    return o + "}\n  }";
}
}  // namespace gray_parrot

// ---- C API, beside ruppell_detail ---------------------------------------------------------------

extern "C" {
// Counter steps (gray-parrot GP1). on=1: count the kernels queued from now on, for counter step
// `step` (its id, as the program numbers its steps; the records carry it), with group `group`, or
// the next group in rotation when group < 0. Kernel dispatch records are turned on with it, so the
// counted kernels keep their identity, scope and graph join; on=0 stops counting and turns them
// back off if counting turned them on. While counting is on, the SDK serializes every dispatch on
// the GPU; stop it only once the device has finished the step's kernels (the Python API waits):
// stopping with counted kernels queued hung the GPU (runner plan 03, probe A2). Returns the group
// counted (on), 0 (off), -1 without counters (unset or refused), -2 when the SDK did not start or
// stop the context (meta.json counts it), -3 for a group out of range.
int
ruppell_counters(int on, int group, unsigned long long step)
{
    using namespace gray_parrot;
    if(!ctr_ok || finished) return -1;
    std::lock_guard<std::mutex> lk(m_switch);
    if(!ctr_ok) return -1;
    if(on && ctr_built == "no")
    {
        // the configurations init could not build (DEFERRED): built now, before the first start
        std::string why;
        ++n_ctr_builds;
        const Built b = build(ctr, 0, meta_spec, "RUPPELL_COUNTER_GROUPS_", ctr_want, why);
        if(b == REFUSED)
        {
            ctr.refused = why;
            ctr_ok      = false;
            std::fprintf(stderr, "[ruppell gray-parrot] counters refused at the first counter step: %s\n", why.c_str());
            return -1;
        }
        if(b == DEFERRED)
        {
            ctr_deferred      = why;
            last_switch_error = "build: " + why;
            return -2;
        }
        ctr_built = "at the first counter step";
        write_tsv();
    }
    if(on)
    {
        const int ng = static_cast<int>(ctr.groups.size());
        if(group >= ng) return -3;
        const int g = group >= 0 ? group : next_group++ % ng;
        cur_group   = g;
        cur_step    = static_cast<uint32_t>(step);
        if(counting) return g;   // a new group (and step) for the kernels queued from now on
        turned_detail = false;
        if(!ruppell_tool::detail_state())
        {
            if(ruppell_detail(1) != 0) return -2;
            turned_detail = true;
        }
        rocprofiler_status_t st = rocprofiler_start_context(ctx_ctr);
        if(st != ROCPROFILER_STATUS_SUCCESS)
        {
            ++n_start_fail;
            last_switch_error = std::string("start: ") + status_name(st);
            if(turned_detail) ruppell_detail(0);
            turned_detail = false;
            if(st == ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT)
            {
                // another tool counts in this process (rocprofv3 --pmc did, probe A5): the SDK allows
                // one counting context per process, so this run has no counters; tracing goes on
                ctr.refused = "another tool counts in this process (ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT at the "
                              "first counter step)";
                ctr_ok = false;
                std::fprintf(stderr, "[ruppell gray-parrot] counters refused, tracing goes on: %s\n", ctr.refused.c_str());
                return -1;
            }
            return -2;
        }
        counting = true;
        ++n_counter_steps;
        return g;
    }
    if(!counting) return 0;
    rocprofiler_status_t st = rocprofiler_stop_context(ctx_ctr);
    if(st != ROCPROFILER_STATUS_SUCCESS)
    {
        ++n_stop_fail;
        last_switch_error = std::string("stop: ") + status_name(st);
        return -2;
    }
    counting = false;
    if(turned_detail) ruppell_detail(0);
    turned_detail = false;
    return 0;
}

// The number of groups counter steps rotate over (0 without counters).
int
ruppell_counter_groups()
{
    return gray_parrot::ctr_ok ? static_cast<int>(gray_parrot::ctr.groups.size()) : 0;
}
}
