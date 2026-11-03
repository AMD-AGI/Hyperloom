// ruppell's NCCL profiler plugin (roadmap-v2 M8): NCCL's (and RCCL's) own view of each call.
//
// Loaded by NCCL or RCCL through NCCL_PROFILER_PLUGIN, which `ruppell run` sets to this library
// by default (RUPPELL_NCCL=0 turns it off). The library calls it per communicator (init), per
// event (startEvent, stopEvent, recordEventState) and at ncclCommDestroy (finalize). It exports
// two versions of the
// interface, and the library takes the newest it knows: NCCL 2.29 takes v5, RCCL (ROCm 7.2,
// interfaces v1 to v4) takes v4. It writes to $RUPPELL_OUT/<host>-<pid>/nccl.tsv, beside the
// tool's records of the same process, as it goes, so a process killed before ncclCommDestroy
// (vLLM's workers, often) keeps what it saw up to the last 100 ms.
//
// Off the library's threads: an event formats its line into a buffer under one lock, and a writer
// thread hands the buffer to the file every 100 ms (and at finalize and exit). The library's
// threads never wait on the file, which on a synchronous NFS mount cost each call 116-217 us
// when they wrote themselves (runner ruppell-accuracy/results/m8, step 5). Beyond 256 MB not yet
// written, lines are dropped and counted; the last line says how many ("drop"). The library
// unloads the plugin when its last communicator is destroyed, and the writer would then wake in
// unmapped code (every rank of a DDP run died of SIGSEGV after destroy_process_group, MI350X,
// 2026-10-21), so once it writes, the plugin pins itself for the life of the process.
//
// Lines (tab-separated; host times are CLOCK_REALTIME ns, which clock.tsv maps to the tool's
// timestamps; GPU times are the library's own GPU timer, as its kernels read it, written as given:
// NCCL's globaltimer in ns, RCCL's wall_clock64 in 10 ns ticks on gfx9. The engine measures the
// unit against the host times and scales to ns, engine._nccl_tick):
//   comm    comm  name  nNodes  nRanks  rank  t  version
//   api     comm  id  coll|p2p  func  count  datatype  root|peer  stream  graphCaptured  t   (v5)
//   launch  comm  id  stream  t                                                          (v5)
//   coll    comm  id  api  seq  func  count  datatype  root  nChannels  nWarps  algo  proto  t
//   p2p     comm  id  api  func  count  datatype  peer  nChannels  t
//   chs     comm  task  nTimed  gpuStart  gpuStop
//   ch      comm  task  channel  gpuStart  gpuStop    (only for a channel whose task is gone)
//   drop    lines                                     (only when lines were dropped)
//   chd     comm  task  channel  gpuStart  gpuStop    (RUPPELL_NCCL_CHANNELS=1: each channel event)
// `api` on coll and p2p lines is the id of what made the task: the API call (v5), or the group
// (v4, which has no API events). The tasks sharing it are one kernel launch: RCCL at 8 ranks
// (MI350X) runs an all-gather as 16 p2p tasks, a send and a receive per peer, in one kernel. 0
// when the library gave neither. A task's channels make one chs line, when the last of its
// nChannels stops (each task gets exactly that many, NCCL 2.30 and RCCL 2.27, 2026-10-21):
// how many had a stop time, the first start and the last stop. Tasks still short of their
// channels write theirs at exit. Ids are per process, from 1. RCCL at 8 ranks (MI350X,
// 2026-10-21) attaches channel events of later kernels to an earlier task: one all-gather's 99
// channels spread over 270-580 ms, so its span is not its kernel's. The chd lines show it.
//
// What the libraries never report as a task (DDP at 8 ranks, 2026-10-21): NCCL 2.29's
// broadcasts, which it schedules from its own broadcast queue while its task events walk only the
// collective and p2p queues (src/enqueue.cc, src/plugin/profiler.cc at v2.29.7-1); their API
// lines are written, and the engine takes each as its call's record. And RCCL's MSCCL kernels
// (mscclKernel_*), of which nothing is written: interface v4 has no API events.
//
// The library starts a channel's event before its task's work ends, and keeps handing back the
// task's handle after the task's own stopEvent, so tasks live in a ring of slots and are never
// freed; a channel keeps its task's slot and id, and finds the slot reused if the id changed.

#include "../third_party/nccl_profiler/err.h"
#include "../third_party/nccl_profiler/types.h"
#include "../third_party/nccl_profiler/profiler.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

namespace
{
constexpr int MASK_V5 = ncclProfileColl | ncclProfileP2p | ncclProfileCollApi | ncclProfileP2pApi |
                        ncclProfileKernelLaunch | ncclProfileKernelCh;
// v4: a task's parent is its group event, and RCCL (ROCm 7.2) starts no task without one
constexpr int    MASK_V4 = ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileKernelCh;
constexpr size_t SLOTS   = 1u << 16;   // events in flight at once, far more than a library keeps
constexpr size_t CAP     = 256u << 20;  // bytes not yet written, beyond which lines are dropped

struct ctx_t
{
    uint64_t comm;
};

// every event handle starts with its type, as NCCL's example does (ours, whatever the version)
struct api_t
{
    uint64_t type;
    uint64_t id;
};
struct task_t
{
    uint64_t type;
    uint64_t id;
    uint64_t comm;
    unsigned nch;     // channels the library said the task runs on
    unsigned done;    // channels stopped
    unsigned timed;   // of those, with a stop time
    bool     written; // its chs line is out
    uint64_t start, stop;
};
struct ch_t
{
    uint64_t type;
    task_t*  t;
    uint64_t task;
    uint64_t comm;
    uint64_t start;
    uint64_t stop;
    int      channel;
};

// all leaked: the library may finalize, and the writer run, after static destructors
std::mutex&              m       = *new std::mutex();  // the buffer, the slots, the counters
std::mutex&              wm      = *new std::mutex();  // the file: one writer at a time, in order
std::condition_variable& cv      = *new std::condition_variable();
std::string&             pending = *new std::string();
// what drain writes from: emptied after each write with its capacity kept, and swapped with
// pending, so neither grows (copying itself) under the lock once it has reached a 100 ms load
std::string& spare = *new std::string();
FILE*                    out     = nullptr;
bool                     tried   = false;
bool                     diag    = false;  // RUPPELL_NCCL_CHANNELS=1: every channel's times too (chd)
std::atomic<uint64_t>    next_id{1};
uint64_t                 n_api = 0, n_task = 0, n_ch = 0, dropped = 0;  // under m
api_t*                   apis  = nullptr;
task_t*                  tasks = nullptr;
ch_t*                    chans = nullptr;

uint64_t
now_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

// what the buffer holds, to the file; the caller holds neither lock
void
drain()
{
    std::lock_guard<std::mutex> wl(wm);
    {
        std::lock_guard<std::mutex> lk(m);
        pending.swap(spare);
    }
    if(out && !spare.empty())
    {
        std::fwrite(spare.data(), 1, spare.size(), out);
        std::fflush(out);
    }
    spare.clear();
}

void
writer()
{
    for(;;)
    {
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait_for(lk, std::chrono::milliseconds(100));
        }
        drain();
    }
}

// the tool's process directory: RUPPELL_OUT/<host>-<pid>, as the tools name it. Call under m.
bool
open_out()
{
    if(tried) return out != nullptr;
    tried          = true;
    const char* dg = std::getenv("RUPPELL_NCCL_CHANNELS");
    diag           = dg && dg[0] == '1';
    const char* od = std::getenv("RUPPELL_OUT");
    if(!od || !*od) return false;
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    for(char* p = host; *p; ++p)
        if(*p == '.')
        {
            *p = 0;
            break;
        }
    std::string     dir = std::string(od) + "/" + host + "-" + std::to_string((long long) getpid());
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    out = std::fopen((dir + "/nccl.tsv").c_str(), "a");
    if(!out) return false;
    // the writer outlives the library's dlclose: keep this library mapped until exit
    Dl_info self{};
    if(dladdr(reinterpret_cast<void*>(&open_out), &self) && self.dli_fname)
        dlopen(self.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
    apis  = new api_t[SLOTS]();
    tasks = new task_t[SLOTS]();
    chans = new ch_t[SLOTS]();
    pending.reserve(4u << 20);
    spare.reserve(4u << 20);
    std::thread(writer).detach();
    return true;
}

// a line into the buffer; call under m
void
line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void
line(const char* fmt, ...)
{
    if(!out) return;
    char    buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if(n <= 0) return;
    if(size_t(n) >= sizeof(buf))  // only a name could be that long: cut, and keep the newline
    {
        n          = sizeof(buf) - 1;
        buf[n - 1] = '\n';
    }
    if(pending.size() + n > CAP)
    {
        ++dropped;
        return;
    }
    pending.append(buf, n);
}

const char*
s(const char* x)
{
    return x ? x : "";
}

uint64_t
comm_of(void* context)
{
    return context ? static_cast<ctx_t*>(context)->comm : 0;
}

uint64_t
parent_api(void* parent)
{
    if(!parent) return 0;
    uint64_t t = *static_cast<uint64_t*>(parent);
    return (t == ncclProfileCollApi || t == ncclProfileP2pApi || t == ncclProfileGroup)
               ? static_cast<api_t*>(parent)->id
               : 0;
}

ncclResult_t
init_common(void** context, int* mask, int want, uint64_t comm, const char* name, int nNodes, int nranks, int rank,
            int version)
{
    std::lock_guard<std::mutex> lk(m);
    if(!open_out())
    {
        *mask    = 0;
        *context = nullptr;
        return ncclSuccess;
    }
    *mask    = want;
    *context = new ctx_t{comm};
    line("comm\t%llu\t%s\t%d\t%d\t%d\t%llu\tv%d\n", (unsigned long long) comm, s(name), nNodes, nranks, rank,
         (unsigned long long) now_ns(), version);
    return ncclSuccess;
}

// a task's channels as one line; call under m
void
chs(task_t* k)
{
    k->written = true;
    line("chs\t%llu\t%llu\t%u\t%llu\t%llu\n", (unsigned long long) k->comm, (unsigned long long) k->id, k->timed,
         (unsigned long long) k->start, (unsigned long long) k->stop);
}

// a new task in the ring; a task it displaces that is still short of channels writes what it has
task_t*
task(uint64_t type, uint64_t comm, unsigned nch)
{
    task_t* k = &tasks[n_task++ % SLOTS];
    if(k->id && !k->written && k->done) chs(k);
    *k = {type, next_id++, comm, nch, 0, 0, false, 0, 0};
    return k;
}

// the events both versions have: tasks (coll, p2p) and kernel channels. Call under m.
void*
coll(uint64_t comm, uint64_t api, uint64_t seq, const char* func, size_t count, const char* dtype, int root,
     unsigned nch, unsigned nwarps, const char* algo, const char* proto)
{
    task_t* k = task(ncclProfileColl, comm, nch);
    line("coll\t%llu\t%llu\t%llu\t%llu\t%s\t%zu\t%s\t%d\t%u\t%u\t%s\t%s\t%llu\n", (unsigned long long) comm,
         (unsigned long long) k->id, (unsigned long long) api, (unsigned long long) seq, s(func), count, s(dtype), root,
         nch, nwarps, s(algo), s(proto), (unsigned long long) now_ns());
    return k;
}

void*
p2p(uint64_t comm, uint64_t api, const char* func, size_t count, const char* dtype, int peer, unsigned nch)
{
    task_t* k = task(ncclProfileP2p, comm, nch);
    line("p2p\t%llu\t%llu\t%llu\t%s\t%zu\t%s\t%d\t%u\t%llu\n", (unsigned long long) comm, (unsigned long long) k->id,
         (unsigned long long) api, s(func), count, s(dtype), peer, nch, (unsigned long long) now_ns());
    return k;
}

void*
channel(void* parent, uint64_t comm, int ch, uint64_t start)
{
    auto* t = static_cast<task_t*>(parent);
    if(t && t->type != ncclProfileColl && t->type != ncclProfileP2p) t = nullptr;
    ch_t* c = &chans[n_ch++ % SLOTS];
    *c      = {ncclProfileKernelCh, t, t ? t->id : 0, t ? t->comm : comm, start, 0, ch};
    return c;
}

// ---- interface v5 (NCCL 2.29) -------------------------------------------------------------------

ncclResult_t
init_v5(void** context, uint64_t commId, int* mask, const char* commName, int nNodes, int nranks, int rank,
        ncclDebugLogger_t)
{
    return init_common(context, mask, MASK_V5, commId, commName, nNodes, nranks, rank, 5);
}

ncclResult_t
start_v5(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* d)
{
    *eHandle = nullptr;
    if(!out || !d) return ncclSuccess;
    const uint64_t comm = comm_of(context);
    std::lock_guard<std::mutex> lk(m);
    switch(d->type)
    {
        case ncclProfileCollApi:
        case ncclProfileP2pApi:
        {
            api_t* a = &apis[n_api++ % SLOTS];
            *a       = {d->type, next_id++};
            *eHandle = a;
            if(d->type == ncclProfileCollApi)
                line("api\t%llu\t%llu\tcoll\t%s\t%zu\t%s\t%d\t%p\t%d\t%llu\n", (unsigned long long) comm,
                     (unsigned long long) a->id, s(d->collApi.func), d->collApi.count, s(d->collApi.datatype),
                     d->collApi.root, d->collApi.stream, d->collApi.graphCaptured ? 1 : 0,
                     (unsigned long long) now_ns());
            else
                line("api\t%llu\t%llu\tp2p\t%s\t%zu\t%s\t-1\t%p\t%d\t%llu\n", (unsigned long long) comm,
                     (unsigned long long) a->id, s(d->p2pApi.func), d->p2pApi.count, s(d->p2pApi.datatype),
                     d->p2pApi.stream, d->p2pApi.graphCaptured ? 1 : 0, (unsigned long long) now_ns());
            return ncclSuccess;
        }
        case ncclProfileKernelLaunch:
            line("launch\t%llu\t%llu\t%p\t%llu\n", (unsigned long long) comm, (unsigned long long) parent_api(d->parentObj),
                 d->kernelLaunch.stream, (unsigned long long) now_ns());
            return ncclSuccess;
        case ncclProfileColl:
            *eHandle = coll(comm, parent_api(d->parentObj), d->coll.seqNumber, d->coll.func, d->coll.count,
                            d->coll.datatype, d->coll.root, d->coll.nChannels, d->coll.nWarps, d->coll.algo,
                            d->coll.proto);
            return ncclSuccess;
        case ncclProfileP2p:
            *eHandle = p2p(comm, parent_api(d->parentObj), d->p2p.func, d->p2p.count, d->p2p.datatype, d->p2p.peer,
                           d->p2p.nChannels);
            return ncclSuccess;
        case ncclProfileKernelCh:
            *eHandle = channel(d->parentObj, comm, d->kernelCh.channelId, d->kernelCh.pTimer);
            return ncclSuccess;
        default: return ncclSuccess;
    }
}

// ---- interface v4 (RCCL in ROCm 7.2; NCCL 2.25-2.26) ------------------------------------------------

ncclResult_t
init_v4(void** context, int* mask, const char* commName, uint64_t commHash, int nNodes, int nranks, int rank,
        ncclDebugLogger_t)
{
    return init_common(context, mask, MASK_V4, commHash, commName, nNodes, nranks, rank, 4);
}

ncclResult_t
start_v4(void* context, void** eHandle, ncclProfilerEventDescr_v4_t* d)
{
    *eHandle = nullptr;
    if(!out || !d) return ncclSuccess;
    const uint64_t comm = comm_of(context);
    std::lock_guard<std::mutex> lk(m);
    switch(d->type)
    {
        case ncclProfileGroup:  // a handle for its tasks to hang off; no line
        {
            api_t* g = &apis[n_api++ % SLOTS];
            *g       = {ncclProfileGroup, next_id++};
            *eHandle = g;
            return ncclSuccess;
        }
        case ncclProfileColl:  // its group's id, as v5's tasks carry their API call's
            *eHandle = coll(comm, parent_api(d->parentObj), d->coll.seqNumber, d->coll.func, d->coll.count,
                            d->coll.datatype, d->coll.root, d->coll.nChannels, d->coll.nWarps, d->coll.algo,
                            d->coll.proto);
            return ncclSuccess;
        case ncclProfileP2p:
            *eHandle = p2p(comm, parent_api(d->parentObj), d->p2p.func, d->p2p.count, d->p2p.datatype, d->p2p.peer,
                           d->p2p.nChannels);
            return ncclSuccess;
        case ncclProfileKernelCh:
            *eHandle = channel(d->parentObj, comm, d->kernelCh.channelId, d->kernelCh.pTimer);
            return ncclSuccess;
        default: return ncclSuccess;
    }
}

// ---- shared by both: a channel's stop, the event's end, finalize ----------------------------------

ncclResult_t
record_state(void* eHandle, ncclProfilerEventState_t state, uint64_t ptimer)
{
    if(!eHandle) return ncclSuccess;
    auto* c = static_cast<ch_t*>(eHandle);
    if(c->type == ncclProfileKernelCh && state == ncclProfilerKernelChStop) c->stop = ptimer;
    return ncclSuccess;
}

ncclResult_t
record_state_v5(void* eHandle, ncclProfilerEventState_v5_t state, ncclProfilerEventStateArgs_v5_t* args)
{
    return record_state(eHandle, state, args ? args->kernelCh.pTimer : 0);
}

ncclResult_t
record_state_v4(void* eHandle, ncclProfilerEventState_v4_t state, ncclProfilerEventStateArgs_v4_t* args)
{
    return record_state(eHandle, state, args ? args->kernelCh.pTimer : 0);
}

// a channel's end folds into its task's line, written when the task's last channel stops; a
// channel whose task's slot was reused writes its own
ncclResult_t
stop_event(void* eHandle)
{
    if(!eHandle) return ncclSuccess;
    auto* c = static_cast<ch_t*>(eHandle);
    if(c->type != ncclProfileKernelCh) return ncclSuccess;
    std::lock_guard<std::mutex> lk(m);
    if(diag)
        line("chd\t%llu\t%llu\t%d\t%llu\t%llu\n", (unsigned long long) c->comm, (unsigned long long) c->task,
             c->channel, (unsigned long long) c->start, (unsigned long long) c->stop);
    task_t* k = c->t;
    if(!k || k->id != c->task || k->written)
    {
        line("ch\t%llu\t%llu\t%d\t%llu\t%llu\n", (unsigned long long) c->comm, (unsigned long long) c->task, c->channel,
             (unsigned long long) c->start, (unsigned long long) c->stop);
        return ncclSuccess;
    }
    ++k->done;
    if(c->stop)
    {
        if(!k->timed || c->start < k->start) k->start = c->start;
        if(c->stop > k->stop) k->stop = c->stop;
        ++k->timed;
    }
    if(k->nch && k->done >= k->nch) chs(k);
    return ncclSuccess;
}

ncclResult_t
finalize(void* context)
{
    delete static_cast<ctx_t*>(context);
    drain();
    return ncclSuccess;
}

// at exit: tasks still short of channels write what they have, then the buffer and the drops
struct flush_at_exit
{
    ~flush_at_exit()
    {
        {
            std::lock_guard<std::mutex> lk(m);
            if(!out) return;
            for(size_t i = 0; tasks && i < SLOTS; ++i)
                if(tasks[i].id && !tasks[i].written && tasks[i].done) chs(&tasks[i]);
        }
        drain();
        std::lock_guard<std::mutex> wl(wm);  // the file's lock, then the buffer's, as drain takes them
        std::lock_guard<std::mutex> lk(m);
        if(dropped) pending += "drop\t" + std::to_string(dropped) + "\n";  // past the cap: it is the last
        if(!pending.empty() && out)
        {
            std::fwrite(pending.data(), 1, pending.size(), out);
            std::fflush(out);
        }
    }
} flusher;
}  // namespace

extern "C" {
__attribute__((visibility("default"))) ncclProfiler_v5_t ncclProfiler_v5 = {
    "ruppell", init_v5, start_v5, stop_event, record_state_v5, finalize};
__attribute__((visibility("default"))) ncclProfiler_v4_t ncclProfiler_v4 = {
    "ruppell", init_v4, start_v4, stop_event, record_state_v4, finalize};
}
