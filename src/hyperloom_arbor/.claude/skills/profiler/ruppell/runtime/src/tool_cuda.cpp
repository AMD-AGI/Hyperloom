// ruppell runtime on CUDA: a CUPTI injection library (the twin of tool.cpp).
//
// Loaded into the traced process by the CUDA driver at cuInit, through
// CUDA_INJECTION64_PATH (see python/ruppell/cli.py). Records kernels, runtime
// and driver API calls and external correlation ids through CUPTI activity
// buffers; graph nodes as they are created (during stream capture, with the
// scope on top of the creating thread's stack), their clones at
// instantiation, and capture begin, end and graph launches through CUPTI
// callbacks. Kernel names come with the kernel records. Everything goes to
// compact binary files under $RUPPELL_OUT; nothing is serialized to JSON in
// the traced process.
//
// Exports the same C API (ruppell_*) as tool.cpp, called through ctypes.
// What CUPTI reports, and why the tool is built this way:
// docs/feasibility-cuda.md.

#include "calls.h"
#include "cpu_tree.h"
#include "gray_parrot_cuda.h"
#include "ranges.h"
#include "records.h"

#include <cuda.h>
#include <cupti.h>

#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
// Supported CUPTI API versions (cuptiGetVersion). Kernel records are read as
// CUpti_ActivityKernel10, which is what these versions deliver; any other
// version fails at load time rather than being read with the wrong layout.
constexpr uint32_t SUPPORTED_CUPTI[] = {130001};

constexpr CUpti_ExternalCorrelationKind EXT_KIND = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2;
constexpr size_t                        BUF_BYTES = 8u << 20;
constexpr uint32_t                      DRIVER_OP = 1u << 16;  // api op = cbid, | DRIVER_OP for the driver domain

// Leaked on purpose, as in tool.cpp: the exit handler may run after this
// library's static destructors.
std::string& out_dir = *new std::string();
FILE*        f_dispatch = nullptr;
FILE*        f_copies   = nullptr;  // copies.bin; stays null with RUPPELL_COPIES=0
FILE*        f_api      = nullptr;
FILE*        f_ext      = nullptr;
FILE*        f_cap      = nullptr;
FILE*        f_kernels  = nullptr;
FILE*        f_defs     = nullptr;
FILE*        f_steps    = nullptr;
FILE*        f_clock    = nullptr;
FILE*        f_self     = nullptr;

std::mutex& m_buf   = *new std::mutex();
std::mutex& m_cap   = *new std::mutex();
std::mutex& m_defs  = *new std::mutex();
std::mutex& m_steps = *new std::mutex();
std::mutex& m_meta  = *new std::mutex();
// Definitions wait here (under m_defs) and are written by the clock thread every second, by
// flush_all and by the finalizer, never by the thread that defines: a write and flush per
// definition on Cluster A's NFS cost each new collective call 1-5 ms (runner
// ruppell-accuracy/results/m8, step 8). m_defs_w keeps one writer at a time, in order.
std::string& defs_pending = *new std::string();
std::string& defs_spare   = *new std::string();
std::mutex&  m_defs_w     = *new std::mutex();

std::unordered_map<std::string, uint64_t>& kernel_ids = *new std::unordered_map<std::string, uint64_t>();

CUpti_SubscriberHandle subscriber = nullptr;
uint32_t               cupti_version = 0;
int                    driver_version = 0;

std::atomic<bool>     active{false};
std::atomic<uint64_t> n_dispatch{0}, n_api{0}, n_ext{0}, n_cap{0}, n_kernels{0}, n_drops{0}, n_self{0}, n_copies{0};
// copies.bin's CUPTI kinds: copies, peer copies, sets
const CUpti_ActivityKind kCopyKinds[] = {CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMCPY2,
                                         CUPTI_ACTIVITY_KIND_MEMSET};
bool
copies_off()
{
    const char* v = std::getenv("RUPPELL_COPIES");
    return v && *v == '0';
}
std::atomic<uint64_t> n_incomplete{0}, n_push_fail{0};
// detail sampling (ruppell_detail_on_stream): requests refused, kernel records switched
std::atomic<uint64_t> n_detail_refused{0}, n_detail_toggles{0};
std::mutex&           m_detail = *new std::mutex();  // kernel records on or off, one switch at a time
bool                  kernels_on = true;
std::atomic<uint64_t> cb_total_ns{0}, cb_max_ns{0}, first_ts{0}, last_ts{0};

std::thread*             clock_thread = nullptr;
std::mutex&              m_clock      = *new std::mutex();
std::condition_variable& cv_clock     = *new std::condition_variable();
bool                     stop_clock   = false;
std::atomic<bool>        finalized{false};

thread_local std::vector<uint64_t> tl_ext;           // external ids pushed on this thread, innermost last
thread_local uint64_t              tl_pending_key = 0;
thread_local int                   tl_depth       = 0;  // nesting of the traced graph API calls

uint64_t
sdk_now()
{
    uint64_t t = 0;
    cuptiGetTimestamp(&t);
    return t;
}

uint64_t
clock_ns(clockid_t id)
{
    timespec ts{};
    clock_gettime(id, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t
this_tid()
{
    return static_cast<uint64_t>(syscall(SYS_gettid));
}

bool
check(CUptiResult r, const char* what)
{
    if(r == CUPTI_SUCCESS) return true;
    const char* s = nullptr;
    cuptiGetResultString(r, &s);
    std::fprintf(stderr, "[ruppell] %s failed: %s\n", what, s ? s : "?");
    return false;
}

void
write_cap(uint64_t kind, uint64_t tid, uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
    ruppell_cap_rec r{kind, tid, sdk_now(), a, b, c, d};
    std::lock_guard<std::mutex> lk(m_cap);
    if(f_cap) std::fwrite(&r, sizeof(r), 1, f_cap);
    ++n_cap;
}

FILE*
open_out(const char* name, const char* mode)
{
    auto  path = out_dir + "/" + name;
    FILE* f    = std::fopen(path.c_str(), mode);
    if(!f) std::fprintf(stderr, "[ruppell] cannot open %s\n", path.c_str());
    return f;
}

// ---- activity buffers: kernels, API calls, external ids ------------------------------------

void CUPTIAPI
on_buffer_request(uint8_t** buffer, size_t* size, size_t* max_records)
{
    *size        = BUF_BYTES;
    *buffer      = static_cast<uint8_t*>(std::aligned_alloc(8, BUF_BYTES));
    *max_records = 0;
}

uint64_t
kernel_id_of(const char* name)
{
    // under m_buf
    std::string key = name ? name : "";
    auto        it  = kernel_ids.find(key);
    if(it != kernel_ids.end()) return it->second;
    uint64_t id = kernel_ids.size() + 1;
    kernel_ids.emplace(key, id);
    if(f_kernels) std::fprintf(f_kernels, "%llu\t%s\n", (unsigned long long) id, key.c_str());
    ++n_kernels;
    return id;
}

void CUPTIAPI
on_buffer_complete(CUcontext ctx, uint32_t stream_id, uint8_t* buffer, size_t, size_t valid)
{
    const uint64_t t0 = sdk_now();
    uint64_t       nrec = 0;
    {
        std::lock_guard<std::mutex> lk(m_buf);
        CUpti_Activity*             rec = nullptr;
        while(cuptiActivityGetNextRecord(buffer, valid, &rec) == CUPTI_SUCCESS)
        {
            ++nrec;
            switch(rec->kind)
            {
                case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                {
                    auto* k = reinterpret_cast<const CUpti_ActivityKernel10*>(rec);
                    if(k->start == 0 && k->end == 0)
                    {
                        ++n_incomplete;  // flushed before it finished
                        break;
                    }
                    ruppell_kernel_rec o{};
                    o.corr          = k->correlationId;
                    o.start         = k->start;
                    o.end           = k->end;
                    o.kernel_id     = kernel_id_of(k->name);
                    o.graph_node_id = k->graphNodeId;
                    o.stream_id     = k->streamId;
                    o.device_id     = k->deviceId;
                    o.graph_id      = k->graphId;
                    o.grid[0]       = static_cast<uint32_t>(k->gridX);
                    o.grid[1]       = static_cast<uint32_t>(k->gridY);
                    o.grid[2]       = static_cast<uint32_t>(k->gridZ);
                    o.block[0]      = static_cast<uint32_t>(k->blockX);
                    o.block[1]      = static_cast<uint32_t>(k->blockY);
                    o.block[2]      = static_cast<uint32_t>(k->blockZ);
                    if(f_dispatch) std::fwrite(&o, sizeof(o), 1, f_dispatch);
                    ++n_dispatch;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMCPY:
                {
                    auto* m = reinterpret_cast<const CUpti_ActivityMemcpy6*>(rec);
                    if(m->start == 0 && m->end == 0) break;
                    ruppell_copy_rec o{};
                    o.corr          = m->correlationId;
                    o.start         = m->start;
                    o.end           = m->end;
                    o.bytes         = m->bytes;
                    o.stream_id     = m->streamId;
                    o.dst_device    = m->deviceId;
                    o.src_device    = m->deviceId;
                    o.graph_node_id = m->graphNodeId;
                    o.graph_id      = m->graphId;
                    o.kind          = RUPPELL_COPY_MEMCPY;
                    o.direction     = m->copyKind;
                    if(f_copies) std::fwrite(&o, sizeof(o), 1, f_copies);
                    ++n_copies;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMCPY2:
                {
                    auto* m = reinterpret_cast<const CUpti_ActivityMemcpyPtoP4*>(rec);
                    if(m->start == 0 && m->end == 0) break;
                    ruppell_copy_rec o{};
                    o.corr          = m->correlationId;
                    o.start         = m->start;
                    o.end           = m->end;
                    o.bytes         = m->bytes;
                    o.stream_id     = m->streamId;
                    o.dst_device    = m->dstDeviceId;
                    o.src_device    = m->srcDeviceId;
                    o.graph_node_id = m->graphNodeId;
                    o.graph_id      = m->graphId;
                    o.kind          = RUPPELL_COPY_MEMCPY;
                    o.direction     = m->copyKind;
                    if(f_copies) std::fwrite(&o, sizeof(o), 1, f_copies);
                    ++n_copies;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMSET:
                {
                    auto* m = reinterpret_cast<const CUpti_ActivityMemset4*>(rec);
                    if(m->start == 0 && m->end == 0) break;
                    ruppell_copy_rec o{};
                    o.corr          = m->correlationId;
                    o.start         = m->start;
                    o.end           = m->end;
                    o.bytes         = m->bytes;
                    o.stream_id     = m->streamId;
                    o.dst_device    = m->deviceId;
                    o.src_device    = m->deviceId;
                    o.graph_node_id = m->graphNodeId;
                    o.graph_id      = m->graphId;
                    o.kind          = RUPPELL_COPY_MEMSET;
                    o.value         = m->value;
                    if(f_copies) std::fwrite(&o, sizeof(o), 1, f_copies);
                    ++n_copies;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_RUNTIME:
                case CUPTI_ACTIVITY_KIND_DRIVER:
                {
                    auto*           a = reinterpret_cast<const CUpti_ActivityAPI*>(rec);
                    ruppell_api_rec o{};
                    o.corr      = a->correlationId;
                    o.start     = a->start;
                    o.end       = a->end;
                    o.thread_id = a->threadId;
                    o.kind      = rec->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? 1 : 2;
                    o.op        = a->cbid | (rec->kind == CUPTI_ACTIVITY_KIND_DRIVER ? DRIVER_OP : 0);
                    if(f_api) std::fwrite(&o, sizeof(o), 1, f_api);
                    ++n_api;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
                {
                    auto* e = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(rec);
                    if(e->externalKind != EXT_KIND) break;
                    ruppell_ext_rec o{e->correlationId, e->externalId};
                    if(f_ext) std::fwrite(&o, sizeof(o), 1, f_ext);
                    ++n_ext;
                    break;
                }
                default: break;
            }
        }
        size_t dropped = 0;
        if(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped) == CUPTI_SUCCESS) n_drops += dropped;
        const uint64_t   t1 = sdk_now();
        ruppell_self_rec s{t0, t1, nrec};
        if(f_self) std::fwrite(&s, sizeof(s), 1, f_self);
    }
    std::free(buffer);
    const uint64_t t1 = sdk_now();
    ++n_self;
    cb_total_ns += t1 - t0;
    uint64_t prev = cb_max_ns.load();
    while(t1 - t0 > prev && !cb_max_ns.compare_exchange_weak(prev, t1 - t0)) {}
    uint64_t zero = 0;
    first_ts.compare_exchange_strong(zero, t0);
    last_ts = t1;
}

// ---- callbacks: graph nodes, clones, capture begin and end, graph launches ------------------

bool
is_capture_begin(CUpti_CallbackDomain dom, CUpti_CallbackId id)
{
    if(dom == CUPTI_CB_DOMAIN_RUNTIME_API)
        return id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000 ||
               id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_ptsz_v10000 ||
               id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_v12030 ||
               id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_ptsz_v12030;
    return id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2 ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz;
}

bool
is_capture_end(CUpti_CallbackDomain dom, CUpti_CallbackId id)
{
    if(dom == CUPTI_CB_DOMAIN_RUNTIME_API)
        return id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_v10000 ||
               id == CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_ptsz_v10000;
    return id == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture ||
           id == CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz;
}

bool
is_graph_launch(CUpti_CallbackDomain dom, CUpti_CallbackId id)
{
    if(dom == CUPTI_CB_DOMAIN_RUNTIME_API)
        return id == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000 ||
               id == CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000;
    return id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz;
}

void
on_api(CUpti_CallbackDomain dom, CUpti_CallbackId id, const CUpti_CallbackData* ci)
{
    const bool enter = ci->callbackSite == CUPTI_API_ENTER;
    if(enter) ++tl_depth;
    const int  depth = tl_depth;
    if(!enter) --tl_depth;
    const uint64_t tid = this_tid();

    if(is_graph_launch(dom, id))
    {
        // a replay's kernels carry one of these correlation ids; both are kept
        if(enter) write_cap(RUPPELL_CAP_LAUNCH, tid, 0, 0, ci->correlationId, depth);
        return;
    }
    if(enter || depth != 1) return;  // capture begin and end: the outermost call, on exit
    const bool ok = ci->functionReturnValue && *static_cast<const int*>(ci->functionReturnValue) == 0;
    if(is_capture_begin(dom, id) && ok)
    {
        write_cap(RUPPELL_CAP_BEGIN, tid, 0, tl_pending_key, 0, 0);
        tl_pending_key = 0;
    }
    else if(is_capture_end(dom, id))
        write_cap(RUPPELL_CAP_END, tid, 0, 0, ok ? 1 : 0, 0);
}

void
on_resource(CUpti_CallbackId id, const CUpti_ResourceData* rd)
{
    if(id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
    {
        // gray-parrot's device sampling (RUPPELL_DEVICE_COUNTERS) starts per device here
        uint32_t dev = 0;
        if(rd->context && cuptiGetDeviceId(rd->context, &dev) == CUPTI_SUCCESS)
            gray_parrot_cuda::context_created(static_cast<int>(dev));
        return;
    }
    auto* g = static_cast<const CUpti_GraphData*>(rd->resourceDescriptor);
    if(!g) return;
    const uint64_t tid = this_tid();
    uint64_t       node = 0;
    uint32_t       graph = 0;
    switch(id)
    {
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED:
            cuptiGetGraphNodeId(g->node, &node);
            if(g->graph) cuptiGetGraphId(g->graph, &graph);
            write_cap(RUPPELL_CAP_NODE, tid, node, tl_ext.empty() ? 0 : tl_ext.back(), (uint64_t) g->nodeType, graph);
            return;
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED:
        {
            uint64_t orig = 0;
            uint32_t ograph = 0;
            cuptiGetGraphNodeId(g->node, &node);
            cuptiGetGraphNodeId(g->originalNode, &orig);
            if(g->graph) cuptiGetGraphId(g->graph, &graph);
            if(g->originalGraph) cuptiGetGraphId(g->originalGraph, &ograph);
            write_cap(RUPPELL_CAP_CLONE, tid, node, orig, graph, ograph);
            return;
        }
        case CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED:
        case CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS:
            if(g->node) cuptiGetGraphNodeId(g->node, &node);
            write_cap(RUPPELL_CAP_NODE_UPDATE, tid, node, 0, 0, id);
            return;
        default: return;
    }
}

// ---- NVTX (RUPPELL_NVTX=1): the program's ranges, through CUPTI's NVTX callbacks -------------
//
// The NVTX headers are not part of CUPTI, nor always installed with CUDA, so the few layouts the
// callbacks carry are mirrored here. NVTX fixes them (its ABI is versioned through the
// attribute struct's version and size); CUPTI's parameter structs are in generated_nvtx_meta.h.

struct nvtx_attr  // nvtxEventAttributes_v2
{
    uint16_t    version;
    uint16_t    size;
    uint32_t    category;
    int32_t     color_type;
    uint32_t    color;
    int32_t     payload_type;
    int32_t     reserved0;
    uint64_t    payload;       // the union's 64 bits
    int32_t     message_type;  // 1 ASCII, 2 wide, 3 a registered string handle
    const void* message;
};
static_assert(sizeof(nvtx_attr) == 48, "nvtxEventAttributes_v2 layout");

struct nvtx_text_params { const char* message; };                    // MarkA, RangeStartA, RangePushA
struct nvtx_wtext_params { const wchar_t* message; };                // the W forms, DomainCreateW
struct nvtx_ex_params { const nvtx_attr* attr; };                    // MarkEx, RangeStartEx, RangePushEx
struct nvtx_end_params { uint64_t id; };                             // RangeEnd
struct nvtx_domain_ex_params { void* domain; const nvtx_attr* attr; };  // DomainMarkEx, ...StartEx, ...PushEx
struct nvtx_domain_end_params { void* domain; uint64_t id; };        // DomainRangeEnd
struct nvtx_domain_params { void* domain; };                         // DomainRangePop
struct nvtx_domain_string_params { void* domain; const char* string; };  // DomainRegisterStringA
struct nvtx_create_params { const char* name; };                     // DomainCreateA

std::atomic<bool> nvtx_injected{false};  // NVTX initialized through this library (InitializeInjectionNvtx2)

std::string
nvtx_text(const nvtx_attr* a, ranges::attrs* out)
{
    if(!a) return {};
    if(a->size >= 32)  // up to and with the payload
    {
        out->category     = a->category;
        out->payload_type = (uint32_t) a->payload_type;  // as given: extension types (NCCL's) are negative
        out->payload      = a->payload;
    }
    if(a->size < sizeof(nvtx_attr) || !a->message) return {};
    switch(a->message_type)
    {
        case 1: return static_cast<const char*>(a->message);
        case 2: return ranges::from_wide(static_cast<const wchar_t*>(a->message));
        case 3: return ranges::registered((uint64_t) a->message);
        default: return {};
    }
}

// what an NVTX call returned (a range id, a domain or string handle), 0 when CUPTI gives none
template <typename T>
uint64_t
nvtx_returned(const CUpti_NvtxData* d)
{
    return d->functionReturnValue ? (uint64_t) * static_cast<const T*>(d->functionReturnValue) : 0;
}

const char*
or_empty(const char* s)
{
    return s ? s : "";
}

void
on_nvtx(CUpti_CallbackId id, const CUpti_NvtxData* d)
{
    if(!ranges::enabled || !d || !d->functionParams) return;
    const void*    p   = d->functionParams;
    const uint64_t tid = this_tid();
    const uint64_t now = sdk_now();
    ranges::attrs  a;
    switch(id)
    {
        case CUPTI_CBID_NVTX_nvtxRangePushA:
            ranges::push(0, ranges::intern(0, or_empty(static_cast<const nvtx_text_params*>(p)->message)), a, now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxRangePushW:
            ranges::push(0, ranges::intern(0, ranges::from_wide(static_cast<const nvtx_wtext_params*>(p)->message)), a,
                         now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxRangePushEx:
        {
            std::string t = nvtx_text(static_cast<const nvtx_ex_params*>(p)->attr, &a);
            ranges::push(0, ranges::intern(0, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx:
        {
            auto*       q   = static_cast<const nvtx_domain_ex_params*>(p);
            uint64_t    dom = (uint64_t) q->domain;
            std::string t   = nvtx_text(q->attr, &a);
            ranges::push(dom, ranges::intern(dom, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxRangePop: ranges::pop(0, now, tid); return;
        case CUPTI_CBID_NVTX_nvtxDomainRangePop:
            ranges::pop((uint64_t) static_cast<const nvtx_domain_params*>(p)->domain, now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxRangeStartA:
            ranges::start(nvtx_returned<uint64_t>(d),
                          ranges::intern(0, or_empty(static_cast<const nvtx_text_params*>(p)->message)), a, now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxRangeStartW:
            ranges::start(nvtx_returned<uint64_t>(d),
                          ranges::intern(0, ranges::from_wide(static_cast<const nvtx_wtext_params*>(p)->message)), a,
                          now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxRangeStartEx:
        {
            std::string t = nvtx_text(static_cast<const nvtx_ex_params*>(p)->attr, &a);
            ranges::start(nvtx_returned<uint64_t>(d), ranges::intern(0, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangeStartEx:
        {
            auto*       q   = static_cast<const nvtx_domain_ex_params*>(p);
            std::string t   = nvtx_text(q->attr, &a);
            ranges::start(nvtx_returned<uint64_t>(d), ranges::intern((uint64_t) q->domain, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxRangeEnd: ranges::end(static_cast<const nvtx_end_params*>(p)->id, now, tid); return;
        case CUPTI_CBID_NVTX_nvtxDomainRangeEnd:
            ranges::end(static_cast<const nvtx_domain_end_params*>(p)->id, now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxMarkA:
            ranges::mark(ranges::intern(0, or_empty(static_cast<const nvtx_text_params*>(p)->message)), a, now, tid);
            return;
        case CUPTI_CBID_NVTX_nvtxMarkW:
            ranges::mark(ranges::intern(0, ranges::from_wide(static_cast<const nvtx_wtext_params*>(p)->message)), a, now,
                         tid);
            return;
        case CUPTI_CBID_NVTX_nvtxMarkEx:
        {
            std::string t = nvtx_text(static_cast<const nvtx_ex_params*>(p)->attr, &a);
            ranges::mark(ranges::intern(0, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx:
        {
            auto*       q = static_cast<const nvtx_domain_ex_params*>(p);
            std::string t = nvtx_text(q->attr, &a);
            ranges::mark(ranges::intern((uint64_t) q->domain, t), a, now, tid);
            return;
        }
        case CUPTI_CBID_NVTX_nvtxDomainCreateA:
            ranges::domain_created(nvtx_returned<void*>(d), or_empty(static_cast<const nvtx_create_params*>(p)->name));
            return;
        case CUPTI_CBID_NVTX_nvtxDomainCreateW:
            ranges::domain_created(nvtx_returned<void*>(d),
                                   ranges::from_wide(static_cast<const nvtx_wtext_params*>(p)->message));
            return;
        case CUPTI_CBID_NVTX_nvtxDomainRegisterStringA:
            ranges::string_registered(nvtx_returned<void*>(d),
                                      or_empty(static_cast<const nvtx_domain_string_params*>(p)->string));
            return;
        default: return;
    }
}

void CUPTIAPI
on_callback(void*, CUpti_CallbackDomain dom, CUpti_CallbackId id, const void* data)
{
    if(!active) return;
    if(dom == CUPTI_CB_DOMAIN_RESOURCE)
        on_resource(id, static_cast<const CUpti_ResourceData*>(data));
    else if(dom == CUPTI_CB_DOMAIN_RUNTIME_API || dom == CUPTI_CB_DOMAIN_DRIVER_API)
        on_api(dom, id, static_cast<const CUpti_CallbackData*>(data));
    else if(dom == CUPTI_CB_DOMAIN_NVTX)
        on_nvtx(id, static_cast<const CUpti_NvtxData*>(data));
}

// ---- clock pairs, flushing --------------------------------------------------------------------

void
write_clock_pair()
{
    const uint64_t a = sdk_now();
    const uint64_t r = clock_ns(CLOCK_REALTIME);
    const uint64_t m = clock_ns(CLOCK_MONOTONIC);
    const uint64_t b = sdk_now();
    if(f_clock)
    {
        std::fprintf(f_clock, "%llu\t%llu\t%llu\t%llu\n", (unsigned long long) a, (unsigned long long) b,
                     (unsigned long long) r, (unsigned long long) m);
        std::fflush(f_clock);
    }
}

void write_meta(bool final);

// The definitions waiting, to defs.tsv
void
drain_defs()
{
    std::lock_guard<std::mutex> wl(m_defs_w);
    {
        std::lock_guard<std::mutex> lk(m_defs);
        defs_pending.swap(defs_spare);
    }
    if(f_defs && !defs_spare.empty())
    {
        std::fwrite(defs_spare.data(), 1, defs_spare.size(), f_defs);
        std::fflush(f_defs);
    }
    defs_spare.clear();
}

// Completed records and the definitions to the files, the files' stdio buffers out,
// meta.json not final.
void
flush_all()
{
    drain_defs();
    check(cuptiActivityFlushAll(0), "flush activity");
    {
        std::lock_guard<std::mutex> lk(m_buf);
        for(FILE* f : {f_dispatch, f_copies, f_api, f_ext, f_kernels, f_self})
            if(f) std::fflush(f);
    }
    {
        std::lock_guard<std::mutex> lk(m_cap);
        if(f_cap) std::fflush(f_cap);
    }
    for(FILE* f : {f_defs, f_steps, f_clock})
        if(f) std::fflush(f);
    cpu_tree::flush();
    ranges::flush();
    calls::flush();
    gray_parrot_cuda::flush();
    write_meta(false);
}

// A clock pair and the definitions every second, a flush every RUPPELL_FLUSH_S seconds
// (default 5, 0 for never): a served process is often killed rather than let exit.
void
clock_loop()
{
    const char* fs      = std::getenv("RUPPELL_FLUSH_S");
    const long  flush_s = fs && *fs ? std::atol(fs) : 5;
    long        n       = 0;
    std::unique_lock<std::mutex> lk(m_clock);
    while(!stop_clock)
    {
        write_clock_pair();
        if(active)
        {
            lk.unlock();
            drain_defs();
            lk.lock();
        }
        if(flush_s > 0 && ++n % flush_s == 0 && active)
        {
            lk.unlock();
            flush_all();
            lk.lock();
        }
        cv_clock.wait_for(lk, std::chrono::seconds(1), [] { return stop_clock; });
    }
}

// ---- lifecycle --------------------------------------------------------------------------------

void
write_api_op_names()
{
    FILE* f = open_out("api_ops.tsv", "w");
    if(!f) return;
    for(uint32_t id = 1; id < CUPTI_RUNTIME_TRACE_CBID_SIZE; ++id)
    {
        const char* name = nullptr;
        if(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, id, &name) == CUPTI_SUCCESS && name)
            std::fprintf(f, "%u\t%s\n", id, name);
    }
    for(uint32_t id = 1; id < CUPTI_DRIVER_TRACE_CBID_SIZE; ++id)
    {
        const char* name = nullptr;
        if(cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, id, &name) == CUPTI_SUCCESS && name)
            std::fprintf(f, "%u\t%s\n", id | DRIVER_OP, name);
    }
    std::fclose(f);
}

void
write_meta(bool final)
{
    std::lock_guard<std::mutex> lk(m_meta);
    // written whole, then renamed: a process killed mid-write (vLLM kills its
    // workers at shutdown) keeps the last complete meta.json, never an empty one
    if(FILE* f = open_out("meta.json.tmp", "w"))
    {
        std::fprintf(f,
                     "{\n  \"record_version\": %d,\n  \"platform\": \"cuda\",\n  \"final\": %s,\n  \"pid\": %d,\n"
                     "  \"sdk_version\": \"cupti %u\",\n  \"runtime_version\": \"driver %d\",\n"
                     "  \"buffer_policy\": \"cupti activity buffers\",\n"
                     "  \"records\": {\"dispatch\": %llu, \"copy\": %llu, \"api\": %llu, \"ext\": %llu, \"capture\": %llu, "
                     "\"kernel_symbols\": %llu, \"buffer_callbacks\": %llu},\n"
                     "  \"dropped_records\": %llu,\n  \"incomplete_kernel_records\": %llu,\n"
                     "  \"external_correlation_push_failures\": %llu,\n"
                     "  \"nvtx\": %s,\n  \"collective_calls\": %s,\n  \"counters\": %s,\n"
                     // detail_toggles > 0 marks a sampled run (witness.py)
                     "  \"detail_toggles\": %llu,\n  \"detail_refused\": %llu,\n"
                     "  \"buffer\": {\"bytes\": %llu, \"callback_total_ns\": %llu, \"callback_max_ns\": %llu, "
                     "\"first_callback_ts\": %llu, \"last_callback_ts\": %llu}\n}\n",
                     RUPPELL_RECORD_VERSION_CUDA, final ? "true" : "false", (int) getpid(), cupti_version,
                     driver_version, (unsigned long long) n_dispatch, (unsigned long long) n_copies, (unsigned long long) n_api,
                     (unsigned long long) n_ext, (unsigned long long) n_cap, (unsigned long long) n_kernels,
                     (unsigned long long) n_self, (unsigned long long) n_drops,
                     (unsigned long long) n_incomplete, (unsigned long long) n_push_fail,
                     ranges::meta_json(nvtx_injected ? "nvtx, through CUPTI" : "nvtx, not injected").c_str(), calls::meta_json().c_str(),
                     gray_parrot_cuda::meta_json().c_str(),
                     (unsigned long long) n_detail_toggles, (unsigned long long) n_detail_refused,
                     (unsigned long long) BUF_BYTES, (unsigned long long) cb_total_ns,
                     (unsigned long long) cb_max_ns, (unsigned long long) first_ts,
                     (unsigned long long) last_ts);
        if(std::fclose(f) == 0)
            std::rename((out_dir + "/meta.json.tmp").c_str(), (out_dir + "/meta.json").c_str());
    }
}

void
tool_fini()
{
    if(finalized.exchange(true) || !active) return;
    // every record, finished or not (unfinished kernels are counted, not written)
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "final flush");
    active = false;
    {
        std::lock_guard<std::mutex> lk(m_clock);
        stop_clock = true;
    }
    cv_clock.notify_all();
    if(clock_thread && clock_thread->joinable()) clock_thread->join();
    write_clock_pair();
    drain_defs();
    // gray-parrot first: its device sampling stops and writes its last samples, so meta.json counts them
    gray_parrot_cuda::fini();
    write_meta(true);
    cpu_tree::close();
    ranges::close();
    calls::close();
    std::lock_guard<std::mutex> lb(m_buf);
    std::lock_guard<std::mutex> lc(m_cap);
    for(FILE** f : {&f_dispatch, &f_copies, &f_api, &f_ext, &f_cap, &f_kernels, &f_defs, &f_steps, &f_clock, &f_self})
        if(*f)
        {
            std::fclose(*f);
            *f = nullptr;
        }
}

bool
version_supported()
{
    cuptiGetVersion(&cupti_version);
    cuDriverGetVersion(&driver_version);
    for(uint32_t v : SUPPORTED_CUPTI)
        if(v == cupti_version && v == CUPTI_API_VERSION) return true;
    std::fprintf(stderr,
                 "[ruppell] unsupported CUPTI API version %u (built against %u; supported: 130001); "
                 "not tracing\n",
                 cupti_version, (unsigned) CUPTI_API_VERSION);
    return false;
}

// A forked child (DLRMv4's DataLoader workers) inherits this process's streams,
// their unflushed buffers and its tracing state, and its RecordFunction observer
// still fires. Its writes land in this process's files in buffer-sized chunks,
// not whole records, and the records after one are read out of alignment (job
// 57923's op records carried thread ids such as 1 and 776). So the child stops
// tracing and drops the streams unflushed; the SDK's exit handler is not its to run.
void
child_after_fork()
{
    active    = false;
    finalized = true;
    cpu_tree::forget_in_child();
    ranges::forget_in_child();
    calls::forget_in_child();
    for(FILE** f : {&f_dispatch, &f_copies, &f_api, &f_ext, &f_cap, &f_kernels, &f_defs, &f_steps, &f_clock, &f_self})
        cpu_tree::drop_inherited(*f);
}

int
tool_init()
{
    const char* od = std::getenv("RUPPELL_OUT");
    if(!od || !*od)
    {
        std::fprintf(stderr, "[ruppell] RUPPELL_OUT is not set; not tracing\n");
        return 0;
    }
    if(!version_supported()) return 0;

    char                   old[CUPTI_OLD_SUBSCRIBER_NAME_MIN_LEN + 1] = {0};
    CUpti_SubscriberParams sp{};
    sp.structSize        = CUpti_SubscriberParams_STRUCT_SIZE;
    sp.subscriberName    = "ruppell";
    sp.oldSubscriberName = old;
    sp.oldSubscriberSize = sizeof(old);
    CUptiResult sr       = cuptiSubscribe_v2(&subscriber, (CUpti_CallbackFunc) on_callback, nullptr, &sp);
    if(sr != CUPTI_SUCCESS)
    {
        const char* s = nullptr;
        cuptiGetResultString(sr, &s);
        std::fprintf(stderr, "[ruppell] cannot subscribe to CUPTI (%s; already held by: %s); not tracing\n",
                     s ? s : "?", old[0] ? old : "unknown");
        return 0;
    }

    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    for(char* p = host; *p; ++p)
        if(*p == '.')
        {
            *p = 0;
            break;
        }
    out_dir = std::string(od) + "/" + host + "-" + std::to_string((long long) getpid());
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    f_dispatch = open_out("dispatch.bin", "wb");
    f_api      = open_out("api.bin", "wb");
    f_ext      = open_out("ext.bin", "wb");
    f_cap      = open_out("capture.bin", "wb");
    f_kernels  = open_out("kernels.tsv", "w");
    f_defs     = open_out("defs.tsv", "w");
    f_steps    = open_out("steps.tsv", "w");
    f_clock    = open_out("clock.tsv", "w");
    f_self     = open_out("self.bin", "wb");
    if(!f_dispatch || !f_api || !f_ext || !f_cap || !f_kernels || !f_defs || !f_steps || !f_clock || !f_self)
        return 0;
    write_api_op_names();
    cpu_tree::open(out_dir);
    ranges::open(out_dir);
    calls::open(out_dir);
    // gray-parrot (hardware counters): reads its switches; nothing more unless they are set
    gray_parrot_cuda::configure(out_dir);
    pthread_atfork(nullptr, nullptr, child_after_fork);

    bool ok = true;
    // NVTX ranges (RUPPELL_NVTX=1): CUPTI calls back only if NVTX was injected with it
    // (NVTX_INJECTION64_PATH, which ruppell run sets to this library; InitializeInjectionNvtx2)
    if(ranges::enabled) ok &= check(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX), "nvtx callbacks");
    for(CUpti_CallbackId id : {CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
                               CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED, CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS,
                               CUPTI_CBID_RESOURCE_CONTEXT_CREATED})
        ok &= check(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, id), "resource callback");
    for(CUpti_CallbackId id = 1; id < CUPTI_RUNTIME_TRACE_CBID_SIZE; ++id)
        if(is_capture_begin(CUPTI_CB_DOMAIN_RUNTIME_API, id) || is_capture_end(CUPTI_CB_DOMAIN_RUNTIME_API, id) ||
           is_graph_launch(CUPTI_CB_DOMAIN_RUNTIME_API, id))
            ok &= check(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, id), "runtime callback");
    for(CUpti_CallbackId id = 1; id < CUPTI_DRIVER_TRACE_CBID_SIZE; ++id)
        if(is_capture_begin(CUPTI_CB_DOMAIN_DRIVER_API, id) || is_capture_end(CUPTI_CB_DOMAIN_DRIVER_API, id) ||
           is_graph_launch(CUPTI_CB_DOMAIN_DRIVER_API, id))
            ok &= check(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id), "driver callback");

    ok &= check(cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM), "thread id type");
    ok &= check(cuptiActivityRegisterCallbacks(on_buffer_request, on_buffer_complete), "activity buffers");
    for(CUpti_ActivityKind k : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_RUNTIME,
                                CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION})
        ok &= check(cuptiActivityEnable(k), "activity kind");
    // memory copies and sets (copies.bin), switched with kernel detail
    if(!copies_off() && (f_copies = open_out("copies.bin", "wb")) != nullptr)
        for(CUpti_ActivityKind k : kCopyKinds) ok &= check(cuptiActivityEnable(k), "copy activity kind");
    if(!ok)
    {
        std::fprintf(stderr, "[ruppell] CUPTI configuration failed; not tracing\n");
        return 0;
    }

    active       = true;
    clock_thread = new std::thread(clock_loop);
    std::atexit(tool_fini);
    std::fprintf(stderr, "[ruppell] tracing to %s (CUPTI %u, driver %d)\n", out_dir.c_str(), cupti_version,
                 driver_version);
    return 1;
}

int
push_both(uint64_t value)
{
    cpu_tree::scope_push(value, sdk_now());
    tl_ext.push_back(value);
    if(cuptiActivityPushExternalCorrelationId(EXT_KIND, value) != CUPTI_SUCCESS)
    {
        ++n_push_fail;
        return -2;
    }
    return 0;
}

int
pop_both()
{
    cpu_tree::scope_pop(sdk_now(), this_tid());
    if(!tl_ext.empty()) tl_ext.pop_back();
    uint64_t last = 0;
    return cuptiActivityPopExternalCorrelationId(EXT_KIND, &last) == CUPTI_SUCCESS ? 0 : -2;
}
}  // namespace

// ---- C API used by python/ruppell (ctypes); the same as tool.cpp's ------------------------------

extern "C" {
// Called by the CUDA driver at cuInit when CUDA_INJECTION64_PATH names this library.
int
InitializeInjection(void)
{
    tool_init();
    return 1;  // the process runs either way; ruppell_active() says whether it is traced
}

// Called by NVTX at the process's first NVTX call when NVTX_INJECTION64_PATH names this library
// (ruppell run sets it with RUPPELL_NVTX=1). CUPTI implements NVTX and calls its subscriber back
// (CUPTI_CB_DOMAIN_NVTX), so the call goes on to CUPTI's own entry, in the libcupti this library
// is linked against. NVTX takes one injection library per process: under another tool's (nsys)
// this is never called, and meta.json says "not injected".
int
InitializeInjectionNvtx2(const void* get_export_table)
{
    Dl_info info{};
    void*   h = nullptr;
    if(dladdr(reinterpret_cast<void*>(&cuptiGetVersion), &info) && info.dli_fname)
        h = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    using init_fn = int (*)(const void*);
    auto cupti_init = h ? reinterpret_cast<init_fn>(dlsym(h, "InitializeInjectionNvtx2")) : nullptr;
    if(!cupti_init)
    {
        std::fprintf(stderr, "[ruppell] NVTX: no InitializeInjectionNvtx2 in CUPTI (%s); NVTX ranges not recorded\n",
                     info.dli_fname ? info.dli_fname : "libcupti not found");
        return 0;
    }
    const int ok  = cupti_init(get_export_table);
    nvtx_injected = ok != 0;
    return ok;
}

int
ruppell_active()
{
    return active ? 1 : 0;
}

int
ruppell_push(uint64_t value)
{
    if(!active) return -1;
    return push_both(value);
}

int
ruppell_pop()
{
    if(!active) return -1;
    return pop_both();
}

int
ruppell_define(uint64_t id, const char* kind, const char* name, const char* attrs_json)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_defs);  // the line waits for drain_defs
    defs_pending += std::to_string(id);
    defs_pending += '\t';
    defs_pending += kind ? kind : "";
    defs_pending += '\t';
    defs_pending += name ? name : "";
    defs_pending += '\t';
    defs_pending += attrs_json && *attrs_json ? attrs_json : "{}";
    defs_pending += '\n';
    return 0;
}

// Push a collective call (the vLLM adapter, roadmap-v2 M8) by its kind, a "collective" definition
// with no scope. The tool pairs the kind with the scope it is called from, and pushes the pair's
// own id, defined once as "collective_at" {collective, parent}.
// - The scope: the innermost value on this thread's stack below RUPPELL_EXT_COLL_AT_BASE, so not
//   a marker, and not an enclosing collective call (a dispatcher around its backend).
// - Why here: defined in Python, each new element count made a kind per layer scope, and 45-48%
//   of a served model's eager calls were first uses at 6 us each (runner
//   ruppell-accuracy/plans/02_ADAPTER_COST.md, B1).
int
ruppell_push_collective(uint64_t kind)
{
    if(!active) return -1;
    uint64_t parent = 0;
    for(auto it = tl_ext.rbegin(); it != tl_ext.rend(); ++it)
        if(*it < RUPPELL_EXT_COLL_AT_BASE)
        {
            parent = *it;
            break;
        }
    static std::mutex                                                             m;
    static std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> ids;  // kind -> parent -> id
    static uint64_t                                                               next = RUPPELL_EXT_COLL_AT_BASE;
    uint64_t                                                                      id   = 0;
    bool                                                                          fresh = false;
    {
        std::lock_guard<std::mutex> lk(m);
        auto&                       by_parent = ids[kind];
        auto                        f         = by_parent.find(parent);
        if(f == by_parent.end())
        {
            id = next++;
            by_parent.emplace(parent, id);
            fresh = true;
        }
        else
            id = f->second;
    }
    if(fresh)
    {
        char attrs[96];
        std::snprintf(attrs, sizeof(attrs), "{\"collective\": %llu, \"parent\": %llu}", (unsigned long long) kind,
                      (unsigned long long) parent);
        ruppell_define(id, "collective_at", "-", attrs);
    }
    return push_both(id);
}

// A collective call's entry (its kind, a "collective" definition) and exit, written to
// calls.bin as a range when it exits (calls.h): the alternative to pushing it, which costs no
// external correlation (runner ruppell-accuracy/plans/02_ADAPTER_COST.md, B3).
int
ruppell_call_enter(uint64_t kind)
{
    if(!active) return -1;
    calls::enter(kind, sdk_now());
    return 0;
}

int
ruppell_call_exit()
{
    if(!active) return -1;
    calls::exit(sdk_now(), this_tid());
    return 0;
}

int
ruppell_capture_key(uint64_t key_id)
{
    if(!active) return -1;
    tl_pending_key = key_id;
    write_cap(RUPPELL_CAP_KEY, this_tid(), key_id, 0, 0, 0);
    return 0;
}

int
ruppell_step_cpu(uint64_t step, uint64_t kind)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_steps);
    std::fprintf(f_steps, "%llu\t%llu\t{\"_cpu\": %llu}\n", (unsigned long long) step, (unsigned long long) sdk_now(),
                 (unsigned long long) kind);
    return 0;
}

int
ruppell_flush()
{
    if(!active) return -1;
    flush_all();
    return 0;
}

// Detail sampling (tool.cpp's ruppell_detail), kernel records on (1) or off
// (0), between steps. Two ways of turning them off failed on CUPTI 130001
// (B200):
// - at once, while the sampled step's kernels were still in flight (vLLM's
//   async scheduling turns detail off for the next step while the GPU runs the
//   sampled one): their end times came out milliseconds late, every sampled
//   step's kernels a median 3.7-9.8 ms (2026-09-30);
// - from a host function queued on the stream, once the GPU had finished the
//   sampled step: CUPTI decides at launch whether a kernel is recorded, so the
//   next steps' kernels, launched before the GPU got there, were recorded
//   outside every step (177,282 at 2/10, 2026-10-20).
// So "off" waits for the GPU to finish everything enqueued (cuCtxSynchronize:
// the sampled step on every stream), then turns kernel records off, before the
// next step's kernels are launched. It costs one drained step per switch, one
// in N steps at RUPPELL_DETAIL=K/N. "On" takes effect at once: kernels launched
// from then are recorded, those already in flight are not. Returns 0, -1 when
// not tracing, -2 when CUPTI refused, -3 when "off" is refused (the stream is
// capturing, or the wait failed): the caller then keeps every step traced.
static int
refuse_detail(const char* why)
{
    if(n_detail_refused++ == 0)
        std::fprintf(stderr, "[ruppell] detail off refused: %s; tracing every step\n", why);
    return -3;
}

// What detail switches: kernel records and the runtime and driver API records under them
// (RUPPELL_DETAIL_API=0 keeps the API records on in every step). With kernel records alone
// switched, every launch's API call was still recorded; switching both, at 5/1000 on B200
// (Qwen3-8B TP1, three rounds against plain), C32's mean TTFT went from +25-70 % to +18-30 %,
// and sampled decode steps read like full detail (98.8 % of kernel time within 3 %, against
// 90.6 % with kernel records alone and 99.1 % between two full-detail runs).
static bool
switch_detail(bool on)
{
    static const bool api = [] {
        const char* v = std::getenv("RUPPELL_DETAIL_API");
        return !(v && *v == '0');
    }();
    bool ok = check(on ? cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
                       : cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), on ? "detail on" : "detail off");
    if(api)
        for(CUpti_ActivityKind k : {CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER})
            ok &= check(on ? cuptiActivityEnable(k) : cuptiActivityDisable(k), on ? "detail on (api)" : "detail off (api)");
    if(f_copies)
        for(CUpti_ActivityKind k : kCopyKinds)
            ok &= check(on ? cuptiActivityEnable(k) : cuptiActivityDisable(k), on ? "detail on (copies)" : "detail off (copies)");
    return ok;
}

int
ruppell_detail_on_stream(int on, void* stream)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_detail);
    if(on)
    {
        if(!kernels_on)
        {
            if(!switch_detail(true)) return -2;
            kernels_on = true;
            ++n_detail_toggles;
        }
        return 0;
    }
    if(!kernels_on) return 0;
    // stream 0 is the default stream, where torch's current stream and ruppell's markers often are
    CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
    if(cuStreamIsCapturing(static_cast<CUstream>(stream), &cs) != CUDA_SUCCESS || cs != CU_STREAM_CAPTURE_STATUS_NONE)
        return refuse_detail("the stream is capturing");
    if(cuCtxSynchronize() != CUDA_SUCCESS) return refuse_detail("cuCtxSynchronize failed");
    check(cuptiActivityFlushAll(0), "flush before detail off");
    if(!switch_detail(false)) return -2;
    kernels_on = false;
    ++n_detail_toggles;
    return 0;
}

int
ruppell_detail(int on)
{
    return ruppell_detail_on_stream(on, nullptr);
}

// gray-parrot's counter steps (tool.cpp's twin; gray_parrot_cuda.cpp): on=1 before a counter
// step's first kernel, with `step` its id and `group` the group to count (< 0: the next in
// rotation); on=0 after its last, once the caller has drained the device (the Python API does).
// Counting on turns kernel records on too, so the step stays on the timeline; off turns them back
// off if counting turned them on. On CUDA a counter step is counted as one Range Profiler user
// range. Returns the group counted (on), 0 (off), -1 without counters (unset, refused, not
// tracing), -2 when CUPTI did not start or stop (meta.json says why), -3 for a group out of range.
static bool counters_turned_detail = false;

int
ruppell_counters(int on, int group, unsigned long long step)
{
    if(!active || gray_parrot_cuda::groups() < 0) return -1;
    if(on)
    {
        if(gray_parrot_cuda::counting()) return gray_parrot_cuda::start(group, static_cast<uint32_t>(step), sdk_now());
        CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
        if(cuStreamIsCapturing(nullptr, &cs) != CUDA_SUCCESS || cs != CU_STREAM_CAPTURE_STATUS_NONE) return -2;
        counters_turned_detail = false;
        {
            std::lock_guard<std::mutex> lk(m_detail);
            if(!kernels_on)
            {
                if(!switch_detail(true)) return -2;
                kernels_on             = true;
                counters_turned_detail = true;
                ++n_detail_toggles;
            }
        }
        const int g = gray_parrot_cuda::start(group, static_cast<uint32_t>(step), sdk_now());
        if(g < 0 && counters_turned_detail) ruppell_detail_on_stream(0, nullptr);
        return g;
    }
    if(!gray_parrot_cuda::counting()) return 0;
    const int rc = gray_parrot_cuda::stop(sdk_now());
    if(counters_turned_detail)
    {
        counters_turned_detail = false;
        ruppell_detail_on_stream(0, nullptr);
    }
    return rc;
}

// the groups a counter step can count, or -1 without counters
int
ruppell_counter_groups()
{
    return active ? gray_parrot_cuda::groups() : -1;
}

int
ruppell_step_fields(uint64_t step, const char* json)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_steps);
    std::fprintf(f_steps, "%llu\t%llu\t%s\n", (unsigned long long) step, (unsigned long long) sdk_now(),
                 json && *json ? json : "{}");
    return 0;
}

// ---- the CPU op tree (cpu_tree.h; RUPPELL_CPU_OPS=1), for the RecordFunction library ----------

// 1 when the tool records the CPU op tree in this process.
int
ruppell_cpu_ops_enabled()
{
    return active && cpu_tree::enabled ? 1 : 0;
}

// An op signature (name, input dims and types, as JSON) under the id the library chose.
int
ruppell_op_define(uint64_t sig, const char* text)
{
    if(!active || !cpu_tree::enabled) return -1;
    cpu_tree::define(sig, text);
    return 0;
}

// An op begins on this thread: its start time, and its depth through *depth.
uint64_t
ruppell_op_enter(uint32_t* depth)
{
    uint32_t d = cpu_tree::op_enter();
    if(depth) *depth = d;
    return sdk_now();
}

// The op begun at t0 (ruppell_op_enter) ends now.
void
ruppell_op_exit(uint64_t sig, uint64_t t0, uint32_t depth)
{
    cpu_tree::op_exit(sig, t0, sdk_now(), this_tid(), depth);
}

// The op begun at t0 carries autograd sequence number seq (see tool.cpp).
void
ruppell_op_seq(uint64_t t0, uint64_t seq, uint32_t backward, uint64_t fwd_tid)
{
    cpu_tree::op_seq(t0, seq, this_tid(), backward, fwd_tid);
}
}
