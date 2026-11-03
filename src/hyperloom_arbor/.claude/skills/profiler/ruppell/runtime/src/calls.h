// A framework adapter's collective calls as ranges (the vLLM adapter, roadmap-v2 M8; runner
// ruppell-accuracy/plans/02_ADAPTER_COST.md, B3), shared by the platform tools. Header-only,
// included once by each.
//
// The adapter marks a call's entry and exit (ruppell_call_enter with the call's kind, a
// "collective" definition; ruppell_call_exit), and the tool writes the call to calls.bin when it
// exits, as a ruppell_range_rec: t0 and t1 its entry and exit by the tool's clock, name its kind,
// depth the calls open around it on the thread. The engine gives a kernel the innermost call open
// on its launching thread at its launch point (eager, or when its graph was captured), as it does
// a range's (M7). No external correlation is pushed, so a kernel keeps its module scope as its
// external id; the push and pop that marked a call before cost 3-4 us a call inside a served model
// (B200, plan 02 B1).
#pragma once

#include "records.h"

#include <stdio_ext.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace calls
{
inline FILE*                 f = nullptr;
inline std::mutex&           m = *new std::mutex();  // leaked, as the tools' globals are
inline std::atomic<uint64_t> n_calls{0}, n_unmatched{0};

struct open_call
{
    uint64_t kind;
    uint64_t t0;
};
inline thread_local std::vector<open_call> tl_open;  // calls open on this thread, innermost last

inline void
open(const std::string& dir)
{
    f = std::fopen((dir + "/calls.bin").c_str(), "wb");
    // written on the calling thread: a large buffer keeps its writes rare (ranges.h says why)
    if(f) std::setvbuf(f, nullptr, _IOFBF, 1u << 20);
}

inline void
enter(uint64_t kind, uint64_t now)
{
    tl_open.push_back({kind, now});
}

inline void
exit(uint64_t now, uint64_t tid)
{
    if(tl_open.empty())
    {
        ++n_unmatched;
        return;
    }
    const open_call c = tl_open.back();
    tl_open.pop_back();
    const ruppell_range_rec r{c.t0, now, tid, tid, c.kind, 0, RUPPELL_RANGE_PUSH, (uint32_t) tl_open.size(), 0, 0};
    std::lock_guard<std::mutex> lk(m);
    if(f) std::fwrite(&r, sizeof(r), 1, f);
    ++n_calls;
}

// for meta.json: calls written, and exits with none open
inline std::string
meta_json()
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "{\"recorded\": %llu, \"unmatched_exits\": %llu}", (unsigned long long) n_calls,
                  (unsigned long long) n_unmatched);
    return buf;
}

inline void
flush()
{
    std::lock_guard<std::mutex> lk(m);
    if(f) std::fflush(f);
}

inline void
close()
{
    std::lock_guard<std::mutex> lk(m);
    if(f)
    {
        std::fclose(f);
        f = nullptr;
    }
}

// a stream inherited across fork() is dropped unwritten (as ranges::forget_in_child)
inline void
forget_in_child()
{
    if(!f) return;
    __fpurge(f);
    std::fclose(f);
    f = nullptr;
}
}  // namespace calls
