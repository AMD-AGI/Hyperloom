// ruppell CPU ops: aten ops as intervals on host threads, from PyTorch's
// RecordFunction callbacks (RUPPELL_CPU_OPS=1; off by default).
//
// Built against the traced process's libtorch (runtime/build_cpu_ops.sh), the
// same on CUDA and ROCm: it knows nothing of the platform. The platform tool
// owns the clock and the files; this library hands it each op through the
// three functions passed to ruppell_cpu_ops_start (cpu_tree.h):
//   ruppell_op_define(sig, json)  once per signature (name, input dims, types)
//   ruppell_op_enter(&depth)      when an op begins: the tool's time
//   ruppell_op_exit(sig, t0, d)   when it ends
// Nothing of torch.profiler or Kineto runs; no GPU tracer is involved.
#include <ATen/record_function.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/Tensor.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace
{
using enter_fn  = uint64_t (*)(uint32_t*);
using exit_fn   = void (*)(uint64_t, uint64_t, uint32_t);
using define_fn = int (*)(uint64_t, const char*);
using seq_fn    = void (*)(uint64_t, uint64_t, uint32_t, uint64_t);

enter_fn  op_enter  = nullptr;
exit_fn   op_exit   = nullptr;
define_fn op_define = nullptr;
seq_fn    op_seq    = nullptr;  // ruppell_cpu_ops_seq: autograd sequence numbers (training)
bool      with_inputs = true;

std::mutex&                                m_sigs = *new std::mutex();
std::unordered_map<std::string, uint64_t>& sigs   = *new std::unordered_map<std::string, uint64_t>();
std::atomic<uint64_t>                      next_sig{1};
thread_local std::unordered_map<std::string, uint64_t>* tl_sigs = nullptr;
at::CallbackHandle                         handle = 0;
std::atomic<uint64_t>                      n_ops{0};

struct Ctx : at::ObserverContext
{
    uint64_t sig = 0, t0 = 0;
    uint32_t depth = 0;
};

void
json_str(std::string& out, const char* s)
{
    out += '"';
    for(; s && *s; ++s)
    {
        if(*s == '"' || *s == '\\') out += '\\';
        if(*s == '\t' || *s == '\n') out += ' ';
        else out += *s;
    }
    out += '"';
}

void
dims_of(std::string& out, c10::IntArrayRef sizes)
{
    out += '[';
    for(size_t i = 0; i < sizes.size(); ++i)
    {
        if(i) out += ',';
        out += std::to_string(sizes[i]);
    }
    out += ']';
}

// A tensor whose sizes live in Python (a __torch_dispatch__ subclass): asking
// for them dispatches into Python from inside the callback, which PyTorch
// asserts on while the subclass is being made (THPVariable_make_subclass calls
// aten::detach; DeepSeek V4.1 on ROCm, job 49756). Recorded without dims.
bool
python_sized(const at::Tensor& t)
{
    return t.key_set().has(c10::DispatchKey::Python);
}

// PyTorch profiler's "Input Dims" / "Input type" for one input
void
input(std::string& dims, std::string& types, const c10::IValue& v)
{
    if(v.isTensor())
    {
        const at::Tensor& t = v.toTensor();
        if(t.defined() && python_sized(t))
        {
            dims += "[]";
            json_str(types, std::string(t.dtype().name()).c_str());
        }
        else if(t.defined())
        {
            dims_of(dims, t.sizes());
            json_str(types, std::string(t.dtype().name()).c_str());
        }
        else
        {
            dims += "[]";
            json_str(types, "");
        }
    }
    else if(v.isTensorList())
    {
        dims += '[';
        bool first = true;
        for(const at::Tensor& t : v.toTensorList().vec())
        {
            if(!first) dims += ',';
            first = false;
            if(t.defined() && !python_sized(t)) dims_of(dims, t.sizes());
            else dims += "[]";
        }
        dims += ']';
        json_str(types, "TensorList");
    }
    else
    {
        dims += "[]";
        json_str(types, v.isScalar() ? "Scalar" : (v.isNone() ? "" : v.tagKind().c_str()));
    }
}

uint64_t
signature(const at::RecordFunction& fn, bool inputs)
{
    std::string key = fn.name();
    std::string dims, types;
    if(inputs)
    {
        dims += '[';
        types += '[';
        bool first = true;
        for(const c10::IValue& v : fn.inputs())
        {
            if(!first)
            {
                dims += ',';
                types += ',';
            }
            first = false;
            input(dims, types, v);
        }
        dims += ']';
        types += ']';
        key += '\t';
        key += dims;
        key += '\t';
        key += types;
    }
    if(!tl_sigs) tl_sigs = new std::unordered_map<std::string, uint64_t>();
    auto it = tl_sigs->find(key);
    if(it != tl_sigs->end()) return it->second;
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lk(m_sigs);
        auto g = sigs.find(key);
        if(g != sigs.end()) id = g->second;
        else
        {
            id = next_sig++;
            sigs.emplace(key, id);
            std::string text = "{\"name\": ";
            json_str(text, fn.name());
            text += ", \"scope\": " + std::to_string(static_cast<int>(fn.scope()));
            if(inputs) text += ", \"dims\": " + dims + ", \"types\": " + types;
            text += "}";
            op_define(id, text.c_str());
        }
    }
    tl_sigs->emplace(std::move(key), id);
    return id;
}

std::unique_ptr<at::ObserverContext>
on_start(const at::RecordFunction& fn)
{
    auto ctx = std::make_unique<Ctx>();
    // An input that cannot be read throws out of the callback; RecordFunction
    // catches it and calls on_end with no context. Record the op by name instead.
    try
    {
        ctx->sig = signature(fn, with_inputs);
    } catch(...)
    {
        ctx->sig = signature(fn, false);
    }
    ctx->t0 = op_enter(&ctx->depth);
    if(op_seq)
    {
        // a forward op and the backward function that differentiates it share
        // a sequence number (the link Kineto draws as forward-backward flows)
        int64_t seq = fn.seqNr();
        if(seq >= 0)
            op_seq(ctx->t0, static_cast<uint64_t>(seq),
                   fn.scope() == at::RecordScope::BACKWARD_FUNCTION ? 1u : 0u, fn.forwardThreadId());
    }
    return ctx;
}

void
on_end(const at::RecordFunction&, at::ObserverContext* c)
{
    auto* ctx = static_cast<Ctx*>(c);
    if(!ctx) return;  // on_start failed: no enter to close
    op_exit(ctx->sig, ctx->t0, ctx->depth);
    ++n_ops;
}
}  // namespace

extern "C" {
// Register the callback for aten ops (and user record_function scopes) on every
// thread. inputs = 0 records names only (cheaper: no input capture).
int
ruppell_cpu_ops_start(void* enter, void* exit, void* define, int inputs)
{
    if(handle) return 0;
    if(!enter || !exit || !define) return -1;
    op_enter    = reinterpret_cast<enter_fn>(enter);
    op_exit     = reinterpret_cast<exit_fn>(exit);
    op_define   = reinterpret_cast<define_fn>(define);
    with_inputs = inputs != 0;
    handle      = at::addGlobalCallback(at::RecordFunctionCallback(on_start, on_end)
                                            .needsInputs(with_inputs)
                                            .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION,
                                                     at::RecordScope::USER_SCOPE}));
    return handle ? 0 : -2;
}

// Record autograd sequence numbers through fn (the tool's ruppell_op_seq), so
// backward kernels can be attributed to the forward scope they differentiate.
// Call before ruppell_cpu_ops_start.
int
ruppell_cpu_ops_seq(void* fn)
{
    op_seq = reinterpret_cast<seq_fn>(fn);
    return 0;
}

// Pause (1) or resume (0) the callback on every thread, for detail sampling:
// a step without kernel records has no use for the ops that launched them, and
// the callback (with its input capture) is the op tree's cost. Ops already
// running when it pauses still get their end (RecordFunction keeps the
// callbacks it started with).
int
ruppell_cpu_ops_pause(int paused)
{
    if(!handle) return -1;
    if(paused)
        at::disableCallback(handle);
    else
        at::reenableCallback(handle);
    return 0;
}

int
ruppell_cpu_ops_stop()
{
    if(!handle) return 0;
    at::removeCallback(handle);
    handle = 0;
    return 0;
}

uint64_t
ruppell_cpu_ops_count()
{
    return n_ops.load();
}
}
