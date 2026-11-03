// The CPU op and module tree (RUPPELL_CPU_OPS=1; off by default), shared by the
// platform tools (tool.cpp, tool_cuda.cpp). Header-only, included once by each.
//
// Two kinds of intervals on host threads, timestamped by the tool (so they
// nest exactly with its API and graph-node records):
// - ruppell scopes, from ruppell_push to ruppell_pop (markers excluded): the
//   module tree the adapters already push;
// - aten ops, reported by the RecordFunction library (cpu_ops.cpp) through
//   ruppell_op_define, ruppell_op_enter and ruppell_op_exit (the tool reads
//   the clock and the thread at both ends).
// Records go to ops.bin (ruppell_op_rec), op signatures to ops.tsv.
#pragma once

#include "records.h"

#include <stdio_ext.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace cpu_tree
{
inline bool                  enabled = false;
inline FILE*                 f_ops   = nullptr;
inline FILE*                 f_sigs  = nullptr;
inline std::mutex&           m_ops   = *new std::mutex();
inline std::atomic<uint64_t> n_ops{0};

struct open_scope
{
    uint64_t id, t0;
    uint32_t depth;
};
inline thread_local std::vector<open_scope> tl_scopes;
inline thread_local uint32_t                tl_depth = 0;  // open scopes and ops on this thread

inline bool
env_on()
{
    const char* v = std::getenv("RUPPELL_CPU_OPS");
    return v && *v && std::string(v) != "0";
}

inline void
open(const std::string& dir)
{
    if(!env_on()) return;
    f_ops   = std::fopen((dir + "/ops.bin").c_str(), "wb");
    f_sigs  = std::fopen((dir + "/ops.tsv").c_str(), "w");
    enabled = f_ops && f_sigs;
}

// Drop a stream inherited across fork() without writing it: its buffer holds
// the parent's unflushed records, which the child's exit would flush again.
inline void
drop_inherited(FILE*& f)
{
    if(!f) return;
    __fpurge(f);
    std::fclose(f);
    f = nullptr;
}

// In a forked child: stop recording and drop the parent's streams (the tools'
// child_after_fork).
inline void
forget_in_child()
{
    enabled = false;
    drop_inherited(f_ops);
    drop_inherited(f_sigs);
}

inline void
write(const ruppell_op_rec& r)
{
    std::lock_guard<std::mutex> lk(m_ops);
    if(f_ops) std::fwrite(&r, sizeof(r), 1, f_ops);
    ++n_ops;
}

inline void
scope_push(uint64_t id, uint64_t now)
{
    if(!enabled || (id & RUPPELL_EXT_MARKER_BIT)) return;
    tl_scopes.push_back({id, now, tl_depth++});
}

inline void
scope_pop(uint64_t now, uint64_t tid)
{
    if(!enabled || tl_scopes.empty()) return;
    open_scope s = tl_scopes.back();
    tl_scopes.pop_back();
    if(tl_depth) --tl_depth;
    write({s.t0, now, tid, s.id, RUPPELL_OP_SCOPE, s.depth});
}

// an aten op: the library calls ruppell_op_enter when it begins (the tool reads
// the time and the depth) and ruppell_op_exit when it ends, on the same thread
inline uint32_t
op_enter()
{
    return tl_depth++;
}

inline void
op_exit(uint64_t sig, uint64_t t0, uint64_t t1, uint64_t tid, uint32_t depth)
{
    if(!enabled) return;
    if(tl_depth) --tl_depth;
    write({t0, t1, tid, sig, RUPPELL_OP_ATEN, depth});
}

// an op's autograd sequence number (RUPPELL_OP_SEQ): a forward op records the
// scope it ran in, so a backward function with the same number (on autograd's
// thread, where no module hook runs) can be attributed to that scope
inline void
op_seq(uint64_t t0, uint64_t seq, uint64_t tid, uint32_t backward, uint64_t fwd_tid)
{
    if(!enabled) return;
    uint64_t id = backward ? fwd_tid : (tl_scopes.empty() ? 0 : tl_scopes.back().id);
    write({t0, seq, tid, id, RUPPELL_OP_SEQ, backward});
}

inline void
define(uint64_t sig, const char* text)
{
    std::lock_guard<std::mutex> lk(m_ops);
    if(f_sigs) std::fprintf(f_sigs, "%llu\t%s\n", (unsigned long long) sig, text ? text : "");
}

inline void
flush()
{
    std::lock_guard<std::mutex> lk(m_ops);
    for(FILE* f : {f_ops, f_sigs})
        if(f) std::fflush(f);
}

inline void
close()
{
    std::lock_guard<std::mutex> lk(m_ops);
    for(FILE** f : {&f_ops, &f_sigs})
        if(*f)
        {
            std::fclose(*f);
            *f = nullptr;
        }
    enabled = false;
}
}  // namespace cpu_tree
