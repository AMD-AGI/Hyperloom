// ruppell runtime: a ROCm profiler SDK tool library.
//
// Loaded into the traced process by the SDK (see python/ruppell/cli.py, which
// sets the environment). Records kernel dispatches and HIP API calls through a
// lossless SDK buffer, graph capture/instantiate/launch and captured launches
// through HIP API callbacks, and kernel names from code-object callbacks.
// Everything goes to compact binary files under $RUPPELL_OUT; nothing is
// serialized to JSON in the traced process. Hardware counters (gray-parrot, off
// unless RUPPELL_COUNTERS or RUPPELL_DEVICE_COUNTERS is set) are gray_parrot.cpp's;
// this file only calls its hooks.
//
// Exports a small C API (ruppell_*) that the Python API calls through ctypes.

#include "calls.h"
#include "cpu_tree.h"
#include "gray_parrot.h"
#include "ranges.h"
#include "records.h"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
// Every non-trivial global is leaked on purpose: the SDK calls tool_fini from its
// own exit handler, which can run after this library's static destructors. An
// object destroyed before tool_fini would make the finalizer write nothing (a
// joinable std::thread destroyed at exit also calls std::terminate).
std::string& out_dir = *new std::string();
FILE*       f_dispatch = nullptr;
FILE*       f_copies   = nullptr;  // copies.bin; stays null with RUPPELL_COPIES=0 or without dispatch records
FILE*       f_api      = nullptr;
FILE*       f_cap      = nullptr;
FILE*       f_kernels  = nullptr;
FILE*       f_defs     = nullptr;
FILE*       f_steps    = nullptr;
FILE*       f_clock    = nullptr;
FILE*       f_self     = nullptr;

std::mutex& m_buf   = *new std::mutex();
std::mutex& m_cap   = *new std::mutex();
std::mutex& m_kern  = *new std::mutex();
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

rocprofiler_context_id_t      ctx_cb  = {0};
rocprofiler_context_id_t      ctx_buf = {0};
rocprofiler_buffer_id_t       buffer  = {0};
// kernel dispatch tracing has its own context, so it can be switched on and off
// at run time (detail sampling): the SDK's per-dispatch cost is the tracer's
// main host cost on a served model (M3, job 49326)
rocprofiler_context_id_t      ctx_disp    = {0};
rocprofiler_buffer_id_t       buffer_disp = {0};
bool                          disp_enabled = false;
std::atomic<int>              detail_on{0};
std::atomic<uint64_t>         n_detail_toggles{0};
rocprofiler_client_finalize_t fini_fn = nullptr;
std::string&                  sdk_runtime_version = *new std::string();
std::string&                  hip_api_filter      = *new std::string("analysis");

std::atomic<bool>     active{false};
std::atomic<uint64_t> n_dispatch{0}, n_api{0}, n_cap{0}, n_kernels{0}, n_drops{0}, n_self{0}, n_copies{0};
// buffer pressure: time the SDK's callback thread spends in our callback, its
// longest call, and the first and last record times (for records per second)
std::atomic<uint64_t> cb_total_ns{0}, cb_max_ns{0}, first_ts{0}, last_ts{0};
std::atomic<uint64_t> n_push_fail{0};

std::thread*             clock_thread = nullptr;
std::mutex&              m_clock      = *new std::mutex();
std::condition_variable& cv_clock     = *new std::condition_variable();
bool                     stop_clock   = false;
std::atomic<bool>        finalized{false};

thread_local int      tl_capturing   = 0;
thread_local uint64_t tl_pending_key = 0;

uint64_t
sdk_now()
{
    rocprofiler_timestamp_t t = 0;
    rocprofiler_get_timestamp(&t);
    return t;
}

uint64_t
clock_ns(clockid_t id)
{
    timespec ts{};
    clock_gettime(id, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

void
check(rocprofiler_status_t s, const char* what)
{
    if(s != ROCPROFILER_STATUS_SUCCESS)
        std::fprintf(stderr, "[ruppell] %s failed: %s\n", what, rocprofiler_get_status_string(s));
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

// ---- buffer: kernel dispatches and HIP API calls -------------------------------------------

void
on_buffer(rocprofiler_context_id_t,
          rocprofiler_buffer_id_t,
          rocprofiler_record_header_t** headers,
          size_t                        num_headers,
          void*,
          uint64_t drop_count)
{
    const uint64_t t0 = sdk_now();
    n_drops += drop_count;
    std::lock_guard<std::mutex> lk(m_buf);
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* h = headers[i];
        if(h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
        if(h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* r = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload);
            ruppell_dispatch_rec o{};
            o.corr         = r->correlation_id.internal;
            o.ext          = r->correlation_id.external.value;
            o.start        = r->start_timestamp;
            o.end          = r->end_timestamp;
            o.kernel_id    = r->dispatch_info.kernel_id;
            o.dispatch_id  = r->dispatch_info.dispatch_id;
            o.queue_id     = r->dispatch_info.queue_id.handle;
            o.agent_id     = r->dispatch_info.agent_id.handle;
            o.thread_id    = r->thread_id;
            o.grid[0]      = r->dispatch_info.grid_size.x;
            o.grid[1]      = r->dispatch_info.grid_size.y;
            o.grid[2]      = r->dispatch_info.grid_size.z;
            o.workgroup[0] = r->dispatch_info.workgroup_size.x;
            o.workgroup[1] = r->dispatch_info.workgroup_size.y;
            o.workgroup[2] = r->dispatch_info.workgroup_size.z;
            if(f_dispatch) std::fwrite(&o, sizeof(o), 1, f_dispatch);
            ++n_dispatch;
        }
        else if(h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            auto* r = static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(h->payload);
            ruppell_copy_rec o{};
            o.corr       = r->correlation_id.internal;
            o.ext        = r->correlation_id.external.value;
            o.start      = r->start_timestamp;
            o.end        = r->end_timestamp;
            o.bytes      = r->bytes;
            o.dst_device = r->dst_agent_id.handle;
            o.src_device = r->src_agent_id.handle;
            o.thread_id  = r->thread_id;
            o.kind       = RUPPELL_COPY_MEMCPY;
            o.direction  = static_cast<uint32_t>(r->operation);
            if(f_copies) std::fwrite(&o, sizeof(o), 1, f_copies);
            ++n_copies;
        }
        else if(h->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* r = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(h->payload);
            ruppell_api_rec o{};
            o.corr      = r->correlation_id.internal;
            o.ext       = r->correlation_id.external.value;
            o.start     = r->start_timestamp;
            o.end       = r->end_timestamp;
            o.thread_id = r->thread_id;
            o.kind      = static_cast<uint32_t>(r->kind);
            o.op        = static_cast<uint32_t>(r->operation);
            if(f_api) std::fwrite(&o, sizeof(o), 1, f_api);
            ++n_api;
        }
    }
    const uint64_t t1 = sdk_now();
    ruppell_self_rec s{t0, t1, num_headers};
    if(f_self) std::fwrite(&s, sizeof(s), 1, f_self);
    ++n_self;
    cb_total_ns += t1 - t0;
    uint64_t prev = cb_max_ns.load();
    while(t1 - t0 > prev && !cb_max_ns.compare_exchange_weak(prev, t1 - t0)) {}
    uint64_t zero = 0;
    first_ts.compare_exchange_strong(zero, t0);
    last_ts = t1;
}

// ---- callbacks: kernel names, graph lifecycle, captured launches -----------------------------

void
on_code_object(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;
    auto* d =
        static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
    // the third column is when the symbol was registered (its code object loaded):
    // a code object loaded inside a measured phase is a lazy load (runner Plan 26 §4.4)
    const uint64_t t = sdk_now();
    std::lock_guard<std::mutex> lk(m_kern);
    if(f_kernels)
        std::fprintf(f_kernels, "%llu\t%s\t%llu\n", (unsigned long long) d->kernel_id,
                     d->kernel_name ? d->kernel_name : "", (unsigned long long) t);
    ++n_kernels;
    gray_parrot::kernel_symbol(d->kernel_id, d->kernel_name);
}

bool
is_kernel_launch(uint32_t op)
{
    switch(op)
    {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvLaunchKernelEx:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchByPtr: return true;
        default: return false;
    }
}

bool
is_memory_op(uint32_t op)
{
    switch(op)
    {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync:
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3DAsync: return true;
        default: return false;
    }
}

void
on_hip(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) return;
    auto*          d     = static_cast<rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
    const uint32_t op    = record.operation;
    const bool     enter = record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER;
    const uint64_t tid   = record.thread_id;
    const bool     ok    = !enter && d->retval.hipError_t_retval == hipSuccess;

    switch(op)
    {
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture:
            if(ok)
            {
                tl_capturing = 1;
                write_cap(RUPPELL_CAP_BEGIN, tid, (uint64_t) d->args.hipStreamBeginCapture.stream,
                          tl_pending_key, 0, 0);
                tl_pending_key = 0;
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt:
            if(ok)
            {
                tl_capturing = 1;
                write_cap(RUPPELL_CAP_BEGIN, tid,
                          (uint64_t) d->args.hipStreamBeginCapture_spt.stream, tl_pending_key, 0, 0);
                tl_pending_key = 0;
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture:
            if(!enter)
            {
                tl_capturing = 0;
                auto* pg     = d->args.hipStreamEndCapture.pGraph;
                write_cap(RUPPELL_CAP_END, tid, (uint64_t) d->args.hipStreamEndCapture.stream,
                          (ok && pg) ? (uint64_t) *pg : 0, ok ? 1 : 0, 0);
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt:
            if(!enter)
            {
                tl_capturing = 0;
                auto* pg     = d->args.hipStreamEndCapture_spt.pGraph;
                write_cap(RUPPELL_CAP_END, tid, (uint64_t) d->args.hipStreamEndCapture_spt.stream,
                          (ok && pg) ? (uint64_t) *pg : 0, ok ? 1 : 0, 0);
            }
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate:
            if(ok && d->args.hipGraphInstantiate.pGraphExec)
                write_cap(RUPPELL_CAP_INSTANTIATE, tid,
                          (uint64_t) *d->args.hipGraphInstantiate.pGraphExec,
                          (uint64_t) d->args.hipGraphInstantiate.graph, 0, 0);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags:
            if(ok && d->args.hipGraphInstantiateWithFlags.pGraphExec)
                write_cap(RUPPELL_CAP_INSTANTIATE, tid,
                          (uint64_t) *d->args.hipGraphInstantiateWithFlags.pGraphExec,
                          (uint64_t) d->args.hipGraphInstantiateWithFlags.graph, 0, 0);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams:
            if(ok && d->args.hipGraphInstantiateWithParams.pGraphExec)
                write_cap(RUPPELL_CAP_INSTANTIATE, tid,
                          (uint64_t) *d->args.hipGraphInstantiateWithParams.pGraphExec,
                          (uint64_t) d->args.hipGraphInstantiateWithParams.graph, 0, 0);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
            if(enter)
                write_cap(RUPPELL_CAP_LAUNCH, tid, (uint64_t) d->args.hipGraphLaunch.graphExec,
                          (uint64_t) d->args.hipGraphLaunch.stream, record.correlation_id.internal,
                          0);
            return;
        case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
            if(enter)
                write_cap(RUPPELL_CAP_LAUNCH, tid, (uint64_t) d->args.hipGraphLaunch_spt.graphExec,
                          (uint64_t) d->args.hipGraphLaunch_spt.stream,
                          record.correlation_id.internal, 0);
            return;
        default: break;
    }

    if(enter && tl_capturing)
    {
        if(is_kernel_launch(op))
            write_cap(RUPPELL_CAP_OP, tid, op, record.correlation_id.internal,
                      record.correlation_id.external.value, 1);
        else if(is_memory_op(op))
            write_cap(RUPPELL_CAP_OP, tid, op, record.correlation_id.internal,
                      record.correlation_id.external.value, 2);
    }
}

// ---- ROCTX (RUPPELL_NVTX=1): the program's ranges, through the SDK's marker callbacks ---------
// ROCTX has no domains: every range is in "roctx". PyTorch's torch.cuda.nvtx and emit_nvtx call
// ROCTX on ROCm (through libroctx64, which reaches the SDK).

void
on_marker(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API || !ranges::enabled) return;
    auto*          d     = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    const bool     enter = record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER;
    const uint64_t tid   = record.thread_id;
    auto           text  = [](const char* s) { return std::string(s ? s : ""); };
    ranges::attrs  a;
    switch(record.operation)
    {
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
            if(enter) ranges::push(0, ranges::intern(0, text(d->args.roctxRangePushA.message), "roctx"), a, sdk_now(), tid);
            return;
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop:
            if(enter) ranges::pop(0, sdk_now(), tid);
            return;
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA:
            // the range's id is what the call returns
            if(!enter)
                ranges::start(d->retval.roctx_range_id_t_retval,
                              ranges::intern(0, text(d->args.roctxRangeStartA.message), "roctx"), a, sdk_now(), tid);
            return;
        case ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop:
            if(enter) ranges::end(d->args.roctxRangeStop.id, sdk_now(), tid);
            return;
        case ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA:
            if(enter) ranges::mark(ranges::intern(0, text(d->args.roctxMarkA.message), "roctx"), a, sdk_now(), tid);
            return;
        default: return;
    }
}

// ---- clock pairs ------------------------------------------------------------------------------

void
write_clock_pair()
{
    const uint64_t a = sdk_now();
    const uint64_t r = clock_ns(CLOCK_REALTIME);
    const uint64_t m = clock_ns(CLOCK_MONOTONIC);
    const uint64_t b = sdk_now();
    if(f_clock)
    {
        std::fprintf(f_clock, "%llu\t%llu\t%llu\t%llu\n", (unsigned long long) a,
                     (unsigned long long) b, (unsigned long long) r, (unsigned long long) m);
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

// Write out everything recorded so far: the SDK's buffer (through on_buffer),
// the definitions, the files' stdio buffers, and meta.json marked not final.
void
flush_all()
{
    drain_defs();
    check(rocprofiler_flush_buffer(buffer), "flush buffer");
    if(disp_enabled) check(rocprofiler_flush_buffer(buffer_disp), "flush dispatch buffer");
    for(FILE* f : {f_dispatch, f_copies, f_api, f_cap, f_kernels, f_defs, f_steps, f_clock, f_self})
        if(f) std::fflush(f);
    cpu_tree::flush();
    ranges::flush();
    calls::flush();
    gray_parrot::flush();
    write_meta(false);
}

// A clock pair and the definitions every second, and a flush every RUPPELL_FLUSH_S
// seconds (default 5, 0 for never): a served process is often killed rather than let
// finalize (a worker at shutdown, anything under rocprofv3's signal handler), and then
// loses only what it recorded since the last flush.
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
        if(active) drain_defs();
        if(flush_s > 0 && ++n % flush_s == 0 && active) flush_all();
        cv_clock.wait_for(lk, std::chrono::seconds(1), [] { return stop_clock; });
    }
}

// ---- lifecycle --------------------------------------------------------------------------------

void
write_api_op_names()
{
    FILE* f = open_out("api_ops.tsv", "w");
    if(!f) return;
    for(uint32_t op = 0; op < ROCPROFILER_HIP_RUNTIME_API_ID_LAST; ++op)
    {
        const char* name = nullptr;
        uint64_t    len  = 0;
        if(rocprofiler_query_buffer_tracing_kind_operation_name(
               ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, op, &name, &len) ==
               ROCPROFILER_STATUS_SUCCESS &&
           name)
            std::fprintf(f, "%u\t%.*s\n", op, (int) len, name);
    }
    std::fclose(f);
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
    gray_parrot::forget_in_child();
    for(FILE** f : {&f_dispatch, &f_copies, &f_api, &f_cap, &f_kernels, &f_defs, &f_steps, &f_clock, &f_self})
        cpu_tree::drop_inherited(*f);
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void*)
{
    fini_fn        = fini_func;
    const char* od = std::getenv("RUPPELL_OUT");
    if(!od || !*od)
    {
        std::fprintf(stderr, "[ruppell] RUPPELL_OUT is not set; not tracing\n");
        return -1;
    }
    // one directory per process: several processes (vLLM's workers) share one
    // RUPPELL_OUT, and each must write its own files
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
    f_cap      = open_out("capture.bin", "wb");
    f_kernels  = open_out("kernels.tsv", "w");
    f_defs     = open_out("defs.tsv", "w");
    f_steps    = open_out("steps.tsv", "w");
    f_clock    = open_out("clock.tsv", "w");
    f_self     = open_out("self.bin", "wb");
    if(!f_dispatch || !f_api || !f_cap || !f_kernels || !f_defs || !f_steps || !f_clock || !f_self)
        return -1;
    write_api_op_names();
    cpu_tree::open(out_dir);
    ranges::open(out_dir);
    calls::open(out_dir);
    pthread_atfork(nullptr, nullptr, child_after_fork);

    check(rocprofiler_create_context(&ctx_cb), "create callback context");
    check(rocprofiler_create_context(&ctx_buf), "create buffer context");
    check(rocprofiler_create_context(&ctx_disp), "create dispatch context");

    rocprofiler_tracing_operation_t co_ops[] = {ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
    check(rocprofiler_configure_callback_tracing_service(ctx_cb, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                         co_ops, 1, on_code_object, nullptr),
          "code object callbacks");

    std::vector<rocprofiler_tracing_operation_t> hip_ops = {
        ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt,
    };
    // Diagnostic switches, to find where the tracer's host cost goes (M3): each
    // turns off a part the analysis needs, so a run with any of them set is
    // not a valid trace. RUPPELL_DISPATCH=0 (no kernel dispatch records),
    // RUPPELL_HIP_API=none (no HIP API records), RUPPELL_LAUNCH_CALLBACKS=0 (no
    // per-launch callbacks: captures lose their registry).
    auto env_off = [](const char* k) {
        const char* v = std::getenv(k);
        return v && (std::string(v) == "0" || std::string(v) == "none");
    };
    if(!env_off("RUPPELL_LAUNCH_CALLBACKS"))
        for(uint32_t op = 0; op < ROCPROFILER_HIP_RUNTIME_API_ID_LAST; ++op)
            if(is_kernel_launch(op) || is_memory_op(op)) hip_ops.push_back(op);
    check(rocprofiler_configure_callback_tracing_service(ctx_cb, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                         hip_ops.data(), hip_ops.size(), on_hip, nullptr),
          "hip api callbacks");
    // ROCTX ranges (RUPPELL_NVTX=1, roadmap-v2 M7)
    if(ranges::enabled)
    {
        rocprofiler_tracing_operation_t mk_ops[] = {
            ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA, ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop, ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA,
            ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop};
        check(rocprofiler_configure_callback_tracing_service(ctx_cb, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                             mk_ops, 5, on_marker, nullptr),
              "roctx callbacks");
    }

    constexpr size_t buf_bytes = 16u << 20;
    check(rocprofiler_create_buffer(ctx_buf, buf_bytes, buf_bytes - buf_bytes / 8,
                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_buffer, nullptr, &buffer),
          "create buffer");
    disp_enabled = !env_off("RUPPELL_DISPATCH");
    if(disp_enabled)
    {
        check(rocprofiler_create_buffer(ctx_disp, buf_bytes, buf_bytes - buf_bytes / 8,
                                        ROCPROFILER_BUFFER_POLICY_LOSSLESS, on_buffer, nullptr, &buffer_disp),
              "create dispatch buffer");
        check(rocprofiler_configure_buffer_tracing_service(ctx_disp, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                           nullptr, 0, buffer_disp),
              "kernel dispatch tracing");
        // memory copies on the DMA engines, in the dispatch context: switched with kernel
        // detail, and tagged with the same external correlation (scopes, markers)
        if(!env_off("RUPPELL_COPIES") && (f_copies = open_out("copies.bin", "wb")) != nullptr)
            check(rocprofiler_configure_buffer_tracing_service(ctx_disp, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                                                               nullptr, 0, buffer_disp),
                  "memory copy tracing");
    }
    // HIP API records: by default only the calls the analysis uses (launches for
    // side-stream ownership and flows; copies, sets, synchronizations and
    // allocations for host blocks; graph calls). PyTorch makes about ten times
    // as many other calls (hipStreamIsCapturing, hipGetDevice, ...), and each
    // traced call costs host time on the step's critical path.
    // RUPPELL_HIP_API=all records every call.
    const char*                                  api_env = std::getenv("RUPPELL_HIP_API");
    const bool                                   api_all = api_env && std::string(api_env) == "all";
    if(api_all) hip_api_filter = "all";
    std::vector<rocprofiler_tracing_operation_t> api_ops;
    if(!api_all)
    {
        // ModuleLoad: hipModuleLoad*, rare, each a code object loaded (a lazy load
        // inside a measured phase shows as one; runner Plan 26 §4.4)
        const char* keep[] = {"Launch", "Memcpy", "Memset", "Synchronize", "WaitEvent", "Graph", "Malloc", "Free", "ModuleLoad"};
        for(uint32_t op = 0; op < ROCPROFILER_HIP_RUNTIME_API_ID_LAST; ++op)
        {
            const char* name = nullptr;
            uint64_t    len  = 0;
            if(rocprofiler_query_buffer_tracing_kind_operation_name(
                   ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, op, &name, &len) != ROCPROFILER_STATUS_SUCCESS ||
               !name)
                continue;
            const std::string n(name, len);
            for(const char* k : keep)
                if(n.find(k) != std::string::npos)
                {
                    api_ops.push_back(op);
                    break;
                }
        }
    }
    if(!env_off("RUPPELL_HIP_API"))
        check(rocprofiler_configure_buffer_tracing_service(ctx_buf, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
                                                           api_all ? nullptr : api_ops.data(),
                                                           api_all ? 0 : api_ops.size(), buffer),
              "hip api tracing");
    else
        hip_api_filter = "none";

    rocprofiler_callback_thread_t cb_thread{};
    check(rocprofiler_create_callback_thread(&cb_thread), "create callback thread");
    check(rocprofiler_assign_callback_thread(buffer, cb_thread), "assign callback thread");
    if(disp_enabled) check(rocprofiler_assign_callback_thread(buffer_disp, cb_thread), "assign callback thread");
    // hardware counters: services can be configured only now; nothing is created when unset
    gray_parrot::configure(out_dir, cb_thread);

    for(auto* c : {&ctx_cb, &ctx_buf, &ctx_disp})
    {
        if(c == &ctx_disp && !disp_enabled) continue;
        int valid = 0;
        rocprofiler_context_is_valid(*c, &valid);
        if(!valid)
        {
            std::fprintf(stderr, "[ruppell] invalid context; not tracing\n");
            return -1;
        }
    }
    check(rocprofiler_start_context(ctx_cb), "start callback context");
    check(rocprofiler_start_context(ctx_buf), "start buffer context");
    // detail (kernel dispatches) starts on: startup, capture and warmup are
    // always traced; an adapter may sample served steps (ruppell_detail)
    if(disp_enabled)
    {
        check(rocprofiler_start_context(ctx_disp), "start dispatch context");
        detail_on = 1;
    }

    clock_thread = new std::thread(clock_loop);
    active       = true;
    gray_parrot::start();
    std::fprintf(stderr, "[ruppell] tracing to %s\n", out_dir.c_str());
    return 0;
}

void
write_meta(bool final)
{
    std::lock_guard<std::mutex> lk(m_meta);
    uint32_t maj = 0, min = 0, pat = 0;
    rocprofiler_get_version(&maj, &min, &pat);
    // written whole, then renamed: a process killed mid-write (vLLM kills its
    // workers at shutdown) keeps the last complete meta.json, never an empty one
    if(FILE* f = open_out("meta.json.tmp", "w"))
    {
        std::fprintf(f,
                     "{\n  \"record_version\": %d,\n  \"final\": %s,\n  \"pid\": %d,\n  \"sdk_version\": \"%u.%u.%u\",\n"
                     "  \"runtime_version\": \"%s\",\n  \"buffer_policy\": \"lossless\",\n  \"hip_api_filter\": \"%s\",\n"
                     "  \"records\": {\"dispatch\": %llu, \"copy\": %llu, \"api\": %llu, \"capture\": %llu, "
                     "\"kernel_symbols\": %llu, \"buffer_callbacks\": %llu},\n"
                     "  \"dropped_records\": %llu,\n  \"external_correlation_push_failures\": %llu,\n"
                     "  \"nvtx\": %s,\n  \"collective_calls\": %s,\n"
                     "  \"counters\": %s,\n"
                     "  \"detail_toggles\": %llu,\n"
                     "  \"buffer\": {\"bytes\": %llu, \"callback_total_ns\": %llu, \"callback_max_ns\": %llu, "
                     "\"first_callback_ts\": %llu, \"last_callback_ts\": %llu}\n}\n",
                     RUPPELL_RECORD_VERSION, final ? "true" : "false", (int) getpid(), maj, min, pat, sdk_runtime_version.c_str(), hip_api_filter.c_str(),
                     (unsigned long long) n_dispatch, (unsigned long long) n_copies, (unsigned long long) n_api,
                     (unsigned long long) n_cap, (unsigned long long) n_kernels,
                     (unsigned long long) n_self, (unsigned long long) n_drops,
                     (unsigned long long) n_push_fail, ranges::meta_json("roctx").c_str(), calls::meta_json().c_str(),
                     gray_parrot::meta_json().c_str(), (unsigned long long) n_detail_toggles,
                     (unsigned long long) (16u << 20),
                     (unsigned long long) cb_total_ns, (unsigned long long) cb_max_ns,
                     (unsigned long long) first_ts, (unsigned long long) last_ts);
        if(std::fclose(f) == 0)
            std::rename((out_dir + "/meta.json.tmp").c_str(), (out_dir + "/meta.json").c_str());
    }
}

void
tool_fini(void*)
{
    if(finalized.exchange(true)) return;
    active = false;
    gray_parrot::fini();  // counting stopped and its buffer drained, before meta.json
    check(rocprofiler_flush_buffer(buffer), "flush buffer");
    if(disp_enabled) check(rocprofiler_flush_buffer(buffer_disp), "flush dispatch buffer");
    {
        std::lock_guard<std::mutex> lk(m_clock);
        stop_clock = true;
    }
    cv_clock.notify_all();
    if(clock_thread && clock_thread->joinable()) clock_thread->join();
    write_clock_pair();
    drain_defs();

    write_meta(true);
    cpu_tree::close();
    ranges::close();
    calls::close();
    for(FILE** f : {&f_dispatch, &f_copies, &f_api, &f_cap, &f_kernels, &f_defs, &f_steps, &f_clock, &f_self})
        if(*f)
        {
            std::fclose(*f);
            *f = nullptr;
        }
}

thread_local std::vector<uint64_t> tl_ext;  // external ids pushed on this thread, innermost last

int
push_both(uint64_t value)
{
    cpu_tree::scope_push(value, sdk_now());
    tl_ext.push_back(value);
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    rocprofiler_user_data_t u{};
    u.value = value;
    int rc  = 0;
    if(rocprofiler_push_external_correlation_id(ctx_cb, tid, u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    if(rocprofiler_push_external_correlation_id(ctx_buf, tid, u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    if(disp_enabled && rocprofiler_push_external_correlation_id(ctx_disp, tid, u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    if(rc) ++n_push_fail;
    return rc;
}

int
pop_both()
{
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    cpu_tree::scope_pop(sdk_now(), tid);
    if(!tl_ext.empty()) tl_ext.pop_back();
    rocprofiler_user_data_t u{};
    int                     rc = 0;
    if(rocprofiler_pop_external_correlation_id(ctx_cb, tid, &u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    if(rocprofiler_pop_external_correlation_id(ctx_buf, tid, &u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    if(disp_enabled && rocprofiler_pop_external_correlation_id(ctx_disp, tid, &u) != ROCPROFILER_STATUS_SUCCESS) rc = -2;
    return rc;
}
}  // namespace

// for gray_parrot.cpp
namespace ruppell_tool
{
int
detail_state()
{
    return detail_on.load();
}

uint64_t
now()
{
    return sdk_now();
}
}  // namespace ruppell_tool

// ---- C API used by python/ruppell (ctypes) ------------------------------------------------------

extern "C" {
int
ruppell_active()
{
    return active ? 1 : 0;
}

// Detail sampling: kernel dispatch records on (1) or off (0). Returns 0, or -1
// when not tracing or dispatch tracing is disabled.
int
ruppell_detail(int on)
{
    if(!active || !disp_enabled) return -1;
    if((on ? 1 : 0) == detail_on.load()) return 0;
    rocprofiler_status_t st = on ? rocprofiler_start_context(ctx_disp) : rocprofiler_stop_context(ctx_disp);
    if(st != ROCPROFILER_STATUS_SUCCESS) return -2;
    detail_on = on ? 1 : 0;
    ++n_detail_toggles;
    return 0;
}

// Push an external correlation value (scope id or marker code) for this thread.
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

// Define a scope or graph key: id, kind ("scope" or "graph"), name, JSON attributes.
// Names and attributes must not contain tabs or newlines (the Python API sanitizes).
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

// Push a collective call by its kind, paired here with the scope it is called from: as
// tool_cuda.cpp's ruppell_push_collective, which says why.
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

// A collective call's entry and exit, written to calls.bin as a range: as tool_cuda.cpp's.
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
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    calls::exit(sdk_now(), tid);
    return 0;
}

// The next stream capture on this thread belongs to graph key `key_id`.
int
ruppell_capture_key(uint64_t key_id)
{
    if(!active) return -1;
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    tl_pending_key = key_id;
    write_cap(RUPPELL_CAP_KEY, tid, key_id, 0, 0, 0);
    return 0;
}

// CPU time of a step boundary on this thread (kind 1 begin, 2 end), for charging
// side-stream work to the step that launched it (docs/plan-m2.md, 9.2).
int
ruppell_step_cpu(uint64_t step, uint64_t kind)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_steps);
    std::fprintf(f_steps, "%llu\t%llu\t{\"_cpu\": %llu}\n", (unsigned long long) step,
                 (unsigned long long) sdk_now(), (unsigned long long) kind);
    return 0;
}

// Flush everything recorded so far and write meta.json (marked not final), for a
// process that may be killed before the SDK finalizes (a server's worker at
// shutdown). The files stay open; the finalizer rewrites meta.json.
int
ruppell_flush()
{
    if(!active) return -1;
    flush_all();
    return 0;
}

// Per-step workload fields as JSON (the schema layer's input).
int
ruppell_step_fields(uint64_t step, const char* json)
{
    if(!active) return -1;
    std::lock_guard<std::mutex> lk(m_steps);
    std::fprintf(f_steps, "%llu\t%llu\t%s\n", (unsigned long long) step, (unsigned long long) sdk_now(),
                 json && *json ? json : "{}");
    return 0;
}
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority,
                      rocprofiler_client_id_t* id)
{
    (void) version;
    (void) priority;
    id->name            = "ruppell";
    sdk_runtime_version = runtime_version ? runtime_version : "";
    static auto cfg     = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                                          &tool_init, &tool_fini, nullptr};
    return &cfg;
}

// ---- the CPU op tree (cpu_tree.h; RUPPELL_CPU_OPS=1), for the RecordFunction library ----------
// The same functions as tool_cuda.cpp's; timestamps and thread ids in this SDK's domain.

extern "C" {
int
ruppell_cpu_ops_enabled()
{
    return active && cpu_tree::enabled ? 1 : 0;
}

int
ruppell_op_define(uint64_t sig, const char* text)
{
    if(!active || !cpu_tree::enabled) return -1;
    cpu_tree::define(sig, text);
    return 0;
}

uint64_t
ruppell_op_enter(uint32_t* depth)
{
    uint32_t d = cpu_tree::op_enter();
    if(depth) *depth = d;
    return sdk_now();
}

void
ruppell_op_exit(uint64_t sig, uint64_t t0, uint32_t depth)
{
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    cpu_tree::op_exit(sig, t0, sdk_now(), tid, depth);
}

// The op begun at t0 carries autograd sequence number seq (backward: 1 for a
// backward function, whose forward ran on fwd_tid).
void
ruppell_op_seq(uint64_t t0, uint64_t seq, uint32_t backward, uint64_t fwd_tid)
{
    rocprofiler_thread_id_t tid = 0;
    rocprofiler_get_thread_id(&tid);
    cpu_tree::op_seq(t0, seq, tid, backward, fwd_tid);
}
}
