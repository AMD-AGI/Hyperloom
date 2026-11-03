// Binary record layouts written by the ruppell tool library.
// Little-endian, packed. python/ruppell/records.py mirrors these exactly;
// change both together and bump RUPPELL_RECORD_VERSION.
#pragma once
#include <cstdint>

// Version 1: ROCm (tool.cpp). Version 2: CUDA (tool_cuda.cpp), which writes
// ruppell_kernel_rec to dispatch.bin and adds ext.bin; the other files keep
// version 1's layouts. meta.json names the platform.
#define RUPPELL_RECORD_VERSION 1
#define RUPPELL_RECORD_VERSION_CUDA 2

#pragma pack(push, 1)

// dispatch.bin: one per kernel dispatch (ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
struct ruppell_dispatch_rec
{
    uint64_t corr;         // correlation_id.internal: the API call that launched it
    uint64_t ext;          // correlation_id.external: scope id, marker code, or 0
    uint64_t start;        // ns, rocprofiler timestamp domain
    uint64_t end;          // ns
    uint64_t kernel_id;    // joins kernels.tsv
    uint64_t dispatch_id;  // unique, increasing in enqueue order
    uint64_t queue_id;
    uint64_t agent_id;
    uint64_t thread_id;    // thread that launched it
    uint32_t grid[3];
    uint32_t workgroup[3];
};
static_assert(sizeof(ruppell_dispatch_rec) == 96, "dispatch record layout");

// api.bin: one per HIP runtime API call (ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
struct ruppell_api_rec
{
    uint64_t corr;
    uint64_t ext;
    uint64_t start;
    uint64_t end;
    uint64_t thread_id;
    uint32_t kind;
    uint32_t op;  // operation id; names in api_ops.tsv
};
static_assert(sizeof(ruppell_api_rec) == 48, "api record layout");

// capture.bin: graph lifecycle and captured launches, from HIP API callbacks
enum ruppell_cap_kind : uint64_t
{
    RUPPELL_CAP_BEGIN       = 1,  // a = stream
    RUPPELL_CAP_END         = 2,  // a = stream, b = graph
    RUPPELL_CAP_INSTANTIATE = 3,  // a = graph exec, b = graph
    RUPPELL_CAP_LAUNCH      = 4,  // a = graph exec, b = stream, c = corr of the hipGraphLaunch call
    RUPPELL_CAP_OP          = 5,  // a = api op, b = corr, c = ext (scope), d = class (1 kernel launch, 2 memory op)
    RUPPELL_CAP_KEY         = 6,  // a = graph key id (names in scopes.tsv, kind "graph"); applies to the next BEGIN on this thread
    // CUDA only (CUPTI resource callbacks; ids are CUPTI's, unique in the process)
    RUPPELL_CAP_NODE        = 7,  // a = node id, b = ext on top of this thread's stack, c = node type, d = graph id
    RUPPELL_CAP_CLONE       = 8,  // a = clone's node id, b = original node id, c = clone's graph id, d = original graph id
    RUPPELL_CAP_NODE_UPDATE = 9,  // a = node id (parameters changed after capture)
};
// On CUDA, BEGIN and END carry no handles (a = b = 0; END c = 1 on success),
// and LAUNCH has a = b = 0, c = corr, d = nesting depth (1: the outermost of a
// runtime call and the driver call inside it).

struct ruppell_cap_rec
{
    uint64_t kind;
    uint64_t thread_id;
    uint64_t ts;
    uint64_t a, b, c, d;
};
static_assert(sizeof(ruppell_cap_rec) == 56, "capture record layout");

// self.bin: time the tool spent handling its own buffers
struct ruppell_self_rec
{
    uint64_t start;
    uint64_t end;
    uint64_t records;
};
static_assert(sizeof(ruppell_self_rec) == 24, "self record layout");

// CUDA: dispatch.bin holds one per kernel (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
struct ruppell_kernel_rec
{
    uint64_t corr;           // correlationId: the API call that launched it (the graph launch for a replay)
    uint64_t ext;            // 0; the external id comes from ext.bin, joined by corr
    uint64_t start;          // ns, CUPTI timestamp domain
    uint64_t end;            // ns
    uint64_t kernel_id;      // joins kernels.tsv (interned by the tool)
    uint64_t graph_node_id;  // 0 unless launched by a graph; the instantiated node, see RUPPELL_CAP_CLONE
    uint64_t stream_id;
    uint64_t device_id;
    uint64_t graph_id;       // 0 unless launched by a graph
    uint32_t grid[3];
    uint32_t block[3];
};
static_assert(sizeof(ruppell_kernel_rec) == 96, "kernel record layout");

// copies.bin (both platforms): one per memory copy, and on CUDA per memset.
// ROCm: ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, the copies the DMA engines run
// (hipMemset, and the copies HIP runs as blit kernels, are kernel dispatches in
// dispatch.bin). CUDA: CUPTI_ACTIVITY_KIND_MEMCPY, MEMCPY2 (peer) and MEMSET.
// Recorded with kernel detail (sampled with it); RUPPELL_COPIES=0 turns it off.
// A run without the file predates it. The record versions are unchanged.
enum ruppell_copy_kind : uint32_t
{
    RUPPELL_COPY_MEMCPY = 1,
    RUPPELL_COPY_MEMSET = 2,
};

struct ruppell_copy_rec
{
    uint64_t corr;           // the API call that issued it (the graph launch for a replay)
    uint64_t ext;            // ROCm: correlation_id.external (scope or 0); CUDA: 0, from ext.bin by corr
    uint64_t start;          // ns, the tool's timestamp domain
    uint64_t end;            // ns
    uint64_t bytes;
    uint64_t stream_id;      // CUDA: stream id; ROCm: 0 (the SDK gives none)
    uint64_t dst_device;     // CUDA: device id (a peer copy's destination); ROCm: destination agent handle
    uint64_t src_device;     // CUDA: a peer copy's source device, else = dst_device; ROCm: source agent handle
    uint64_t graph_node_id;  // CUDA: 0 unless issued by a graph (the instantiated node); ROCm: 0
    uint64_t thread_id;      // ROCm: the issuing thread; CUDA: 0 (from api.bin by corr)
    uint32_t graph_id;       // CUDA: 0 unless issued by a graph
    uint32_t kind;           // ruppell_copy_kind
    uint32_t direction;      // the platform's own code: CUDA CUpti_ActivityMemcpyKind,
                             // ROCm rocprofiler_memory_copy_operation_t; memset: 0
    uint32_t value;          // memset: the value set; memcpy: 0
};
static_assert(sizeof(ruppell_copy_rec) == 96, "copy record layout");

// CUDA: ext.bin, one per API call made while an external id was pushed
// (CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, kind CUSTOM2)
struct ruppell_ext_rec
{
    uint64_t corr;
    uint64_t ext;
};
static_assert(sizeof(ruppell_ext_rec) == 16, "ext record layout");

// ops.bin (both platforms, only with RUPPELL_CPU_OPS=1): one interval on a host
// thread, in the tool's timestamp domain, written when it closes (cpu_tree.h)
enum ruppell_op_kind : uint32_t
{
    RUPPELL_OP_ATEN  = 1,  // id: signature in ops.tsv (op name, input dims and types)
    RUPPELL_OP_SCOPE = 2,  // id: scope id in defs.tsv (a ruppell scope, push to pop)
    // an op's autograd sequence number, written when the op begins: t0 is the
    // op's own start (it joins the op's aten record by thread and t0), t1 the
    // sequence number; id: for a forward op the innermost ruppell scope open on
    // its thread (0 when none), for a backward function its forward thread id;
    // depth: 1 for a backward function, 0 for a forward op
    RUPPELL_OP_SEQ = 3,
};

struct ruppell_op_rec
{
    uint64_t t0;
    uint64_t t1;
    uint64_t thread_id;
    uint64_t id;
    uint32_t kind;
    uint32_t depth;  // open intervals of either kind on the thread when it began
};
static_assert(sizeof(ruppell_op_rec) == 40, "op record layout");

// ranges.bin (both platforms, only with RUPPELL_NVTX=1; roadmap-v2 M7): the program's own
// annotations, NVTX on CUDA and ROCTX on ROCm (ranges.h), one record per range when it
// closes and per mark. Names in ranges.tsv. A run without the file predates it or ran without
// RUPPELL_NVTX; the record versions are unchanged.
enum ruppell_range_kind : uint32_t
{
    RUPPELL_RANGE_PUSH  = 1,  // push to pop, on one thread, nested (depth)
    RUPPELL_RANGE_START = 2,  // start to end, which may be on another thread
    RUPPELL_RANGE_MARK  = 3,  // an instant: t1 = t0
};

struct ruppell_range_rec
{
    uint64_t t0;             // ns, the tool's timestamp domain: the push, start or mark
    uint64_t t1;             // the pop or end
    uint64_t thread_id;      // the thread that pushed, started or marked
    uint64_t end_thread_id;  // the thread that popped or ended
    uint64_t name;           // ranges.tsv: id, domain, text
    uint64_t payload;        // the payload's 64 bits as given (NVTX); payload_type says how to read them
    uint32_t kind;           // ruppell_range_kind
    uint32_t depth;          // ranges of any domain open on the thread when it began
    uint32_t payload_type;   // NVTX's nvtxPayloadType_t: 1 uint64, 2 int64, 3 double, 4 uint32,
                             // 5 int32, 6 float; 0 none; others (extension payloads) unread
    uint32_t category;       // NVTX category; 0 none
};
static_assert(sizeof(ruppell_range_rec) == 64, "range record layout");

// gray-parrot, hardware counters (off by default; written only with RUPPELL_COUNTERS or
// RUPPELL_DEVICE_COUNTERS set). Both platforms; the contract is docs/gray-parrot/records.md,
// and tests/test_gp_records_offline.py checks the engine's dtypes against these structs.
// counters.bin: one per counted kernel and counter, on counter steps.
struct ruppell_counter_rec
{
    uint64_t key0;     // ROCm: dispatch_id (joins dispatch.bin); CUDA: correlation id (a replay's: the graph launch's)
    uint64_t key1;     // ROCm: 0; CUDA: the instantiated graph node, as in ruppell_kernel_rec; 0 for an eager launch
    uint64_t seq;      // ROCm: 0; CUDA: the range's index within the counter step
    uint64_t agent;    // ROCm: the agent handle, as ruppell_dispatch_rec.agent_id; CUDA: the device id
    uint32_t step;     // the counter step's id, as the adapter numbered it (steps.tsv)
    uint16_t group;    // counters.tsv: the group this step collected
    uint16_t counter;  // counters.tsv: the counter
    double   value;    // reduced over dimension instances as counters.tsv says
};
static_assert(sizeof(ruppell_counter_rec) == 48, "counter record layout");

// device_counters.bin: device-wide readings, one per interval and counter: periodic samples
// (RUPPELL_DEVICE_COUNTERS), or one reading per counter step (RUPPELL_COUNTER_SCOPE=step).
struct ruppell_device_counter_rec
{
    uint64_t t0;       // ns, the tool's timestamp domain: the start of the interval the value covers
    uint64_t t1;       // its end
    uint64_t agent;    // ROCm: the agent handle; CUDA: the device id
    uint32_t step;     // a step reading: the counter step's id; a periodic sample: 0 (placed by time)
    uint16_t group;    // counters.tsv
    uint16_t counter;  // counters.tsv
    double   value;    // over [t0, t1]
};
static_assert(sizeof(ruppell_device_counter_rec) == 40, "device counter record layout");

#pragma pack(pop)

// External correlation encoding, shared with the Python API.
// 0: none. Ids the Python API defines (scopes, graph keys, collective kinds): 1 .. 2^56-1 (the vLLM
// adapter's named scopes from 2^40). A collective call paired with the scope it was called from,
// an id the tool defines ("collective_at", ruppell_push_collective): 2^56 .. 2^63-1. Markers: bit
// 63 set, (step << 2) | kind.
#define RUPPELL_EXT_MARKER_BIT (1ULL << 63)
#define RUPPELL_EXT_COLL_AT_BASE (1ULL << 56)
#define RUPPELL_MARKER_BEGIN 1ULL
#define RUPPELL_MARKER_END 2ULL
